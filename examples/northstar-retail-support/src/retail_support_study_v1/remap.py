from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest_file, digest_value, load_json, write_json_atomic
from .constants import (
    PREPARED_DIR,
    PRIVATE_DIR,
    SOURCE_LOCK_PATH,
    TASK_COUNT,
    TAU3_COMMIT,
    TAU3_LICENSE,
    TAU3_REPOSITORY,
    TAU3_TAG_OBJECT,
    TAU3_VERSION,
)

_FIRST_NAMES = (
    "Avery",
    "Blair",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jamie",
    "Kai",
    "Lane",
    "Morgan",
    "Noel",
    "Oakley",
    "Parker",
    "Quinn",
    "Reese",
    "Sage",
    "Taylor",
)

_LAST_NAMES = (
    "Arbor",
    "Brook",
    "Cedar",
    "Dale",
    "Elm",
    "Field",
    "Grove",
    "Harbor",
    "Ivory",
    "Juniper",
    "Knoll",
    "Lake",
    "Meadow",
    "North",
    "Orchard",
    "Pine",
    "Quarry",
    "River",
    "Stone",
    "Thorne",
    "Umber",
    "Vale",
    "Willow",
    "Yarrow",
    "Zephyr",
)

_ID_FIELDS = {
    "user_id",
    "order_id",
    "product_id",
    "item_id",
    "payment_method_id",
    "tracking_id",
}
_ID_LIST_FIELDS = {
    "orders",
    "item_ids",
    "new_item_ids",
    "tracking_ids",
}
_ADDRESS_REDACTION_FIELDS = {"address1", "address2", "zip"}
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PROSE_ID_PATTERNS = {
    "order": re.compile(r"#W\d+"),
    "payment": re.compile(r"(?:credit_card|gift_card|paypal)_\d+"),
    "user": re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b"),
}
_RUNTIME_INTERNAL_ID_PATTERN = re.compile(
    r"\b(?:retail_user_\d{4}|item_\d{5}|prod_\d{4})\b"
)
_CITY_ZIP_CODES: dict[tuple[str, str], str] = {
    ("Austin", "TX"): "78701",
    ("Charlotte", "NC"): "28202",
    ("Chicago", "IL"): "60602",
    ("Columbus", "OH"): "43201",
    ("Dallas", "TX"): "75204",
    ("Denver", "CO"): "80202",
    ("Fort Worth", "TX"): "76102",
    ("Houston", "TX"): "77002",
    ("Indianapolis", "IN"): "46204",
    ("Jacksonville", "FL"): "32202",
    ("Los Angeles", "CA"): "90012",
    ("New York", "NY"): "10002",
    ("Philadelphia", "PA"): "19103",
    ("Phoenix", "AZ"): "85004",
    ("San Antonio", "TX"): "78205",
    ("San Diego", "CA"): "92102",
    ("San Francisco", "CA"): "94103",
    ("San Jose", "CA"): "95116",
    ("Seattle", "WA"): "98101",
    ("Springfield", "IL"): "62702",
    ("Washington", "DC"): "20001",
}
_UNBOUND_ZIP_CODES = ("02108", "33101")
_SENSITIVE_SOURCE_PATTERNS = (
    _EMAIL_PATTERN,
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
)
_PREPARED_SOURCE_FILES = (
    "db.json",
    "policy.md",
    "task_splits.json",
    "tasks.runtime.json",
    "user-guidelines.md",
)
_PRIVATE_SOURCE_FILES = ("replacements.json", "task-truth.json")


def _stable_index(value: str, modulo: int) -> int:
    raw = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % modulo


def _git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_paths(tau_root: Path) -> dict[str, Path]:
    base = tau_root / "data" / "tau2" / "domains" / "retail"
    return {
        "database": base / "db.json",
        "tasks": base / "tasks.json",
        "task_splits": base / "split_tasks.json",
        "policy": base / "policy.md",
        "user_guidelines": (
            tau_root / "data" / "tau2" / "user_simulator" / "simulation_guidelines.md"
        ),
    }


def _add(mapping: dict[str, str], old: str, new: str) -> None:
    if not old or old == new:
        return
    prior = mapping.get(old)
    if prior is not None and prior != new:
        raise ValueError(f"source value has conflicting replacements: {old!r}")
    mapping[old] = new


def _walk_strings(
    value: Any, key: str | None = None
) -> Iterable[tuple[str | None, str]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_strings(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child, key)
    elif isinstance(value, str):
        yield key, value


def _unique_number_map(
    values: Iterable[str],
    *,
    namespace: str,
    lower: int,
    upper: int,
    forbidden: Iterable[str] = (),
) -> dict[str, int]:
    """Assign stable, collision-free numbers while preserving equality."""

    if upper < lower:
        raise ValueError("invalid replacement range")
    width = upper - lower + 1
    unique_values = sorted(set(values))
    blocked = {int(value) for value in forbidden if str(value).isdigit()}
    if len(unique_values) + len(blocked) > width:
        raise ValueError(f"not enough values in the {namespace} replacement range")

    assigned: dict[str, int] = {}
    used = set(blocked)
    for value in unique_values:
        candidate = lower + _stable_index(f"{namespace}:{value}", width)
        while candidate in used:
            candidate = lower + ((candidate - lower + 1) % width)
        assigned[value] = candidate
        used.add(candidate)
    return assigned


def _collect_address_values(
    db: dict[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, set[str]]:
    values = {field: set() for field in _ADDRESS_REDACTION_FIELDS}
    for key, value in _walk_strings((db, tasks)):
        if key in values and value:
            values[key].add(value)
    return values


def _assert_task_reference_coverage(
    tasks: list[dict[str, Any]], replacements: dict[str, str]
) -> None:
    """Reject identity-shaped task references that escaped the remap ledger."""

    missing_structured: list[str] = []
    for key, value in _walk_strings(tasks):
        if key in _ID_FIELDS | _ID_LIST_FIELDS and value not in replacements:
            missing_structured.append(f"{key}:{value}")
    encoded = json.dumps(tasks, ensure_ascii=False, sort_keys=True)
    missing_prose = [
        f"{label}:{match}"
        for label, pattern in _PROSE_ID_PATTERNS.items()
        for match in sorted(set(pattern.findall(encoded)))
        if match not in replacements
    ]
    missing_emails = sorted(
        email
        for email in set(_EMAIL_PATTERN.findall(encoded))
        if email not in replacements
    )
    missing = (
        missing_structured
        + missing_prose
        + [f"email:{email}" for email in missing_emails]
    )
    if missing:
        raise ValueError(f"task identity references lack replacements: {missing[:10]}")


def _add_supplemental_task_replacements(
    tasks: list[dict[str, Any]], mapping: dict[str, str]
) -> None:
    """Remap deliberately invalid or stale task identifiers without making them valid."""

    by_kind: dict[str, set[str]] = {
        "user_id": set(),
        "order_id": set(),
        "product_id": set(),
        "item_id": set(),
        "payment_method_id": set(),
        "tracking_id": set(),
    }
    list_kind = {
        "orders": "order_id",
        "item_ids": "item_id",
        "new_item_ids": "item_id",
        "tracking_ids": "tracking_id",
    }
    for key, value in _walk_strings(tasks):
        normalized_key = list_kind.get(key or "", key or "")
        if normalized_key in by_kind and value not in mapping:
            by_kind[normalized_key].add(value)

    target_order_suffixes = {
        replacement.removeprefix("#R")
        for source, replacement in mapping.items()
        if source.startswith("#W") and replacement.startswith("#R")
    }
    unrelated_orders = [
        order_id
        for order_id in sorted(by_kind["order_id"])
        if not (order_id.startswith("#") and f"#W{order_id[1:]}" in mapping)
    ]
    unrelated_numbers = _unique_number_map(
        unrelated_orders,
        namespace="missing_order",
        lower=1_000_000,
        upper=9_999_999,
        forbidden=target_order_suffixes,
    )
    for order_id in sorted(by_kind["order_id"]):
        corresponding_order = f"#W{order_id[1:]}" if order_id.startswith("#") else ""
        if corresponding_order in mapping:
            target_suffix = mapping[corresponding_order].removeprefix("#R")
        else:
            target_suffix = f"{unrelated_numbers[order_id]:07d}"
        _add(mapping, order_id, f"#{target_suffix}")

    missing_prefixes = {
        "user_id": "missing_user",
        "product_id": "missing_product",
        "item_id": "missing_item",
        "payment_method_id": "missing_payment",
        "tracking_id": "missing_tracking",
    }
    for kind, prefix in missing_prefixes.items():
        for index, value in enumerate(sorted(by_kind[kind]), 1):
            _add(mapping, value, f"{prefix}_{index:04d}")

    encoded = json.dumps(tasks, ensure_ascii=False, sort_keys=True)
    order_replacements = [
        (source, target)
        for source, target in list(mapping.items())
        if source.startswith("#W") or source in by_kind["order_id"]
    ]
    for source, target in order_replacements:
        source_without_hash = source.removeprefix("#")
        target_without_hash = target.removeprefix("#")
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(source_without_hash)}"
            rf"(?![A-Za-z0-9])",
            encoded,
        ):
            _add(mapping, source_without_hash, target_without_hash)
        source_suffix = source_without_hash.removeprefix("W")
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(source_suffix)}(?![A-Za-z0-9])",
            encoded,
        ):
            target_suffix = target_without_hash.removeprefix("R")
            _add(mapping, source_suffix, target_suffix)
    if "forgot the 'W'" in encoded:
        _add(mapping, "forgot the 'W'", "forgot the 'R'")


def _build_replacements(
    db: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[
    dict[str, str],
    dict[str, dict[str, str]],
    dict[tuple[str, str, str], tuple[str, str, str]],
]:
    mapping: dict[str, str] = {}
    field_mapping: dict[str, dict[str, str]] = {"last_four": {}, "zip": {}}
    identity_triples: dict[tuple[str, str, str], tuple[str, str, str]] = {}

    product_ids = sorted(db["products"])
    item_ids = sorted(
        item_id
        for product in db["products"].values()
        for item_id in product["variants"]
    )
    user_ids = sorted(db["users"])
    order_ids = sorted(db["orders"])
    tracking_ids = sorted(
        tracking_id
        for order in db["orders"].values()
        for fulfillment in order.get("fulfillments", [])
        for tracking_id in fulfillment.get("tracking_id", [])
    )
    payment_ids = sorted(
        payment_id
        for user in db["users"].values()
        for payment_id in user["payment_methods"]
    )
    source_last_fours = sorted(
        {
            str(payment["last_four"])
            for user in db["users"].values()
            for payment in user["payment_methods"].values()
            if payment.get("last_four") is not None
        }
    )

    address_values = _collect_address_values(db, tasks)
    street_numbers = _unique_number_map(
        address_values["address1"],
        namespace="address1",
        lower=1000,
        upper=9999,
    )
    unit_numbers = _unique_number_map(
        address_values["address2"],
        namespace="address2",
        lower=1,
        upper=9999,
    )
    for value, number in street_numbers.items():
        _add(mapping, value, f"{number} Lantern Way")
    for value, number in unit_numbers.items():
        _add(mapping, value, f"Unit {number:04d}")

    zip_locations: dict[str, tuple[str, str]] = {}
    for node in _walk_dicts((db, tasks)):
        if not {"city", "state", "zip"}.issubset(node):
            continue
        source_zip = str(node["zip"])
        location = (str(node["city"]), str(node["state"]))
        prior = zip_locations.get(source_zip)
        if prior is not None and prior != location:
            raise ValueError(
                f"source ZIP {source_zip!r} maps to conflicting locations"
            )
        zip_locations[source_zip] = location
    if not set(zip_locations) <= address_values["zip"]:
        raise ValueError("structured address ZIP collection drifted")
    unknown_locations = sorted(set(zip_locations.values()) - set(_CITY_ZIP_CODES))
    if unknown_locations:
        raise ValueError(f"unreviewed source locations: {unknown_locations}")
    unbound_zips = sorted(address_values["zip"] - set(zip_locations))
    if len(unbound_zips) > len(_UNBOUND_ZIP_CODES):
        raise ValueError(f"too many reviewed unbound ZIP clues: {unbound_zips}")
    zip_targets = {
        **{
            source_zip: _CITY_ZIP_CODES[location]
            for source_zip, location in zip_locations.items()
        },
        **dict(zip(unbound_zips, _UNBOUND_ZIP_CODES, strict=False)),
    }
    for source_zip, target_zip in sorted(zip_targets.items()):
        if target_zip in address_values["zip"]:
            raise ValueError(
                f"reviewed public ZIP {target_zip} overlaps a source ZIP"
            )
        _add(mapping, source_zip, target_zip)
    field_mapping["zip"] = {
        value: mapping[value] for value in sorted(address_values["zip"])
    }

    remapped_last_fours = _unique_number_map(
        source_last_fours,
        namespace="last_four",
        lower=1000,
        upper=9999,
        forbidden=source_last_fours,
    )
    field_mapping["last_four"] = {
        value: f"{number:04d}" for value, number in remapped_last_fours.items()
    }

    for index, value in enumerate(product_ids, 1):
        _add(mapping, value, f"prod_{index:04d}")
    for index, value in enumerate(item_ids, 1):
        _add(mapping, value, f"item_{index:05d}")
    for index, value in enumerate(order_ids, 1):
        _add(mapping, value, f"#R{index:07d}")
    for index, value in enumerate(tracking_ids, 1):
        _add(mapping, value, f"track_{index:06d}")
    for index, value in enumerate(payment_ids, 1):
        prefix = next(
            (
                candidate
                for candidate in ("credit_card", "gift_card", "paypal")
                if value.startswith(candidate + "_")
            ),
            "payment",
        )
        _add(mapping, value, f"{prefix}_{index:06d}")

    for index, user_id in enumerate(user_ids, 1):
        user = db["users"][user_id]
        proposed_first = _FIRST_NAMES[(index - 1) % len(_FIRST_NAMES)]
        proposed_last = _LAST_NAMES[
            ((index - 1) // len(_FIRST_NAMES)) % len(_LAST_NAMES)
        ]
        new_user_id = f"retail_user_{index:04d}"
        _add(mapping, user_id, new_user_id)
        original_first = user["name"]["first_name"]
        original_last = user["name"]["last_name"]
        original_zip = str(user["address"]["zip"])
        original_full = f"{original_first} {original_last}"
        if original_full in mapping:
            first, last = mapping[original_full].split(" ", 1)
        else:
            first, last = proposed_first, proposed_last
            _add(mapping, original_full, f"{first} {last}")
        remapped_zip = mapping[original_zip]
        identity_triples[(original_first, original_last, original_zip)] = (
            first,
            last,
            remapped_zip,
        )
        _add(mapping, user["email"], f"customer-{index:04d}@example.invalid")

    _add_supplemental_task_replacements(tasks, mapping)

    all_task_emails = sorted(
        set(_EMAIL_PATTERN.findall(json.dumps(tasks, ensure_ascii=False)))
    )
    supplemental_emails = [email for email in all_task_emails if email not in mapping]
    for index, email in enumerate(supplemental_emails, 1):
        _add(
            mapping,
            email,
            f"supplemental-contact-{index:04d}@example.invalid",
        )

    _assert_task_reference_coverage(tasks, mapping)
    return mapping, field_mapping, identity_triples


def _compile_replacement_pattern(
    replacements: dict[str, str],
) -> re.Pattern[str]:
    ordered_keys = sorted(replacements, key=len, reverse=True)

    def replacement_expression(key: str) -> str:
        prefix = r"(?<![A-Za-z0-9])" if key[0].isalnum() else ""
        suffix = r"(?![A-Za-z0-9])" if key[-1].isalnum() else ""
        return f"{prefix}{re.escape(key)}{suffix}"

    return re.compile(
        "|".join(replacement_expression(key) for key in ordered_keys if key)
    )


def _rewrite(
    value: Any,
    replacements: dict[str, str],
    field_replacements: dict[str, dict[str, str]],
    identity_triples: dict[tuple[str, str, str], tuple[str, str, str]],
    *,
    compiled_pattern: re.Pattern[str] | None = None,
    replace_embedded: bool = True,
) -> Any:
    replacement_pattern = compiled_pattern
    if replace_embedded and replacement_pattern is None:
        replacement_pattern = _compile_replacement_pattern(replacements)

    def rewrite_string(text: str) -> str:
        if text in replacements:
            return replacements[text]
        if not replace_embedded:
            return text
        assert replacement_pattern is not None
        return replacement_pattern.sub(lambda match: replacements[match.group(0)], text)

    def visit(node: Any, key: str | None = None) -> Any:
        if isinstance(node, dict):
            rewritten = {
                rewrite_string(str(child_key)): visit(child, str(child_key))
                for child_key, child in node.items()
            }
            if {"first_name", "last_name"}.issubset(rewritten):
                original_name = (
                    str(node["first_name"]),
                    str(node["last_name"]),
                )
                remapped_names = {
                    (target[0], target[1])
                    for source, target in identity_triples.items()
                    if source[:2] == original_name
                }
                if len(remapped_names) == 1:
                    rewritten["first_name"], rewritten["last_name"] = next(
                        iter(remapped_names)
                    )
            return rewritten
        if isinstance(node, list):
            return [visit(child, key) for child in node]
        if isinstance(node, str):
            replacement = field_replacements.get(key or "", {}).get(node)
            if replacement is not None:
                return replacement
            return rewrite_string(node)
        return node

    return visit(value)


def _infer_primary_user_id(task: dict[str, Any], db: dict[str, Any]) -> str:
    """Resolve the one customer whose identity grounds a retail task."""

    encoded = json.dumps(task, ensure_ascii=False, sort_keys=True)
    payment_owners = {
        payment_id: user_id
        for user_id, user in db["users"].items()
        for payment_id in user["payment_methods"]
    }
    email_owners = {user["email"]: user_id for user_id, user in db["users"].items()}
    candidates = (
        {user_id for user_id in db["users"] if user_id in encoded}
        | {
            order["user_id"]
            for order_id, order in db["orders"].items()
            if order_id in encoded
        }
        | {
            owner
            for payment_id, owner in payment_owners.items()
            if payment_id in encoded
        }
        | {owner for email, owner in email_owners.items() if email in encoded}
    )

    if not candidates:
        known_info = str(
            task.get("user_scenario", {}).get("instructions", {}).get("known_info")
            or ""
        )
        candidates = {
            user_id
            for user_id, user in db["users"].items()
            if (
                f"{user['name']['first_name']} {user['name']['last_name']}"
                in known_info
                and str(user["address"]["zip"]) in known_info
            )
        }
    if len(candidates) != 1:
        raise ValueError(
            f"task {task.get('id')} must resolve one customer; found "
            f"{sorted(candidates)}"
        )
    return next(iter(candidates))


def _align_task_user_name(
    source_task: dict[str, Any],
    remapped_task: dict[str, Any],
    source_db: dict[str, Any],
    remapped_db: dict[str, Any],
    replacements: dict[str, str],
) -> None:
    """Keep first-name-only task clues aligned with the remapped account."""

    source_user_id = _infer_primary_user_id(source_task, source_db)
    remapped_user_id = replacements[source_user_id]
    source_first = str(source_db["users"][source_user_id]["name"]["first_name"])
    remapped_first = str(remapped_db["users"][remapped_user_id]["name"]["first_name"])
    instructions = remapped_task["user_scenario"]["instructions"]
    known_info = instructions.get("known_info")
    if isinstance(known_info, str):
        instructions["known_info"] = re.sub(
            rf"\b{re.escape(source_first)}\b",
            remapped_first,
            known_info,
        )


def _align_database_user_names(
    source_db: dict[str, Any],
    remapped_db: dict[str, Any],
    replacements: dict[str, str],
    identity_triples: dict[tuple[str, str, str], tuple[str, str, str]],
) -> None:
    """Apply the tuple-aware name remap to database user records."""

    for source_user_id, source_user in source_db["users"].items():
        target_user_id = replacements[source_user_id]
        original = (
            source_user["name"]["first_name"],
            source_user["name"]["last_name"],
            source_user["address"]["zip"],
        )
        first, last, _ = identity_triples[original]
        remapped_db["users"][target_user_id]["name"] = {
            "first_name": first,
            "last_name": last,
        }


def _humanize_runtime_scenario(
    scenario: dict[str, Any],
    *,
    authenticated_user_id: str,
    public_user_name: str,
    item_descriptions: dict[str, str],
    product_descriptions: dict[str, str],
) -> dict[str, Any]:
    """Remove database-shaped identity and benchmark scripting from user prose."""

    user_id = re.escape(authenticated_user_id)

    def rewrite_text(text: str) -> str:
        # Some source tasks identify the simulated customer by an internal user
        # ID. Keep that ID at the MCP account boundary, but give the simulator
        # the fictional customer's public name instead.
        text = re.sub(
            rf"\bYou just created your user id {user_id}\b",
            "You just created your account",
            text,
        )
        text = re.sub(
            rf"\byour user id {user_id}\b",
            "your account",
            text,
        )
        text = re.sub(
            rf"\bYou are [A-Z][A-Za-z'-]* \({user_id}\)",
            f"You are {public_user_name}",
            text,
        )
        text = re.sub(
            rf"\bYou are user {user_id}\b",
            f"You are {public_user_name}",
            text,
        )
        text = re.sub(
            rf"\bYou are {user_id}\b",
            f"You are {public_user_name}",
            text,
        )
        text = re.sub(
            rf"\b{user_id}\b",
            public_user_name,
            text,
        )

        # Preserve the deliberate missing-prefix interaction without forcing a
        # simulator to recite a benchmark answer word for word.
        text = re.sub(
            r"respond exactly with ['\"]I forgot the ['\"]?[WR]['\"]? "
            r"at the beginning['\"]",
            "say you left off the R prefix and provide the corrected order number",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"say that you forgot the ['\"]?[WR]['\"]? at the beginning",
            "say you left off the R prefix and provide the corrected order number",
            text,
            flags=re.IGNORECASE,
        )

        # One source task supplied a hidden item-ID oracle. Preserve the user's
        # actual request—match the hose in the pending order—without giving the
        # simulator an evaluator-like answer key.
        text = re.sub(
            r"The correct garden hose item id is item_\d+ "
            r"\(do not share this with the agent\)\. If the agent proposes a "
            r"different item id for the garden hose, tell them that's not the "
            r"right one and to keep looking in your pending order\.",
            (
                "For the garden hose, ask for the same version that appears in "
                "your pending order. If the agent proposes a different hose, say "
                "it does not match the pending one and ask them to check again."
            ),
            text,
            flags=re.IGNORECASE,
        )

        # Customer prose should describe catalog choices, not database keys.
        # The private reference actions retain the exact IDs used for scoring.
        for item_id, description in item_descriptions.items():
            text = re.sub(
                rf"\bitem (?:id|number) {re.escape(item_id)}\b",
                description,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"\b{re.escape(item_id)}\b",
                description,
                text,
                flags=re.IGNORECASE,
            )
        for product_id, description in product_descriptions.items():
            text = re.sub(
                rf"\b(?:item|product) (?:id )?{re.escape(product_id)}\b",
                description,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"\b{re.escape(product_id)}\b",
                description,
                text,
                flags=re.IGNORECASE,
            )
        text = text.replace(
            (
                "IF the agent cannot find item with ID the mechanical keyboard, "
                "mention that it could be a product ID."
            ),
            (
                "If the agent cannot identify the keyboard, clarify that you mean "
                "the mechanical keyboard rather than a specific variant."
            ),
        )
        text = text.replace(
            "You are not familiar with the domain and might confuse product and item ids",
            (
                "You are not sure how the store distinguishes product types from "
                "specific variants"
            ),
        )
        text = text.replace(
            (
                "Also, you want to exchange the garden hose you received for the "
                "type that is in your pending order. For the garden hose, ask for "
                "the same version that appears in your pending order."
            ),
            (
                "Also, ask to exchange the garden hose you received for the same "
                "version listed in your pending order."
            ),
        )
        return text

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: visit(child) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child) for child in value]
        if isinstance(value, str):
            return rewrite_text(value)
        return value

    result = visit(deepcopy(scenario))
    assert isinstance(result, dict)
    return result


def _catalog_reference_descriptions(
    remapped_db: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    def option_phrase(key: str, raw_value: Any) -> str:
        value = str(raw_value)
        if key in {"color", "material"}:
            return value
        if key == "size":
            return f"size {value}"
        if key == "sole":
            return f"{value}-soled"
        if key == "backlight":
            return "no-backlight" if value == "none" else f"{value}-backlit"
        if key == "switch type":
            return f"{value}-switch"
        if key == "length":
            return f"{value}-long"
        if key == "screen size":
            return value
        if key == "processor":
            return f"{value}-processor"
        if key == "ram":
            return f"{value}-RAM"
        if key == "storage":
            return f"{value}-storage"
        return f"{key} {value}"

    item_descriptions: dict[str, str] = {}
    product_descriptions: dict[str, str] = {}
    for product_id, product in remapped_db["products"].items():
        product_name = str(product["name"]).lower()
        product_descriptions[str(product_id)] = f"the {product_name}"
        for item_id, item in product["variants"].items():
            options = [
                option_phrase(str(key), value)
                for key, value in sorted(item.get("options", {}).items())
            ]
            qualifier = ", ".join(options)
            item_descriptions[str(item_id)] = (
                f"the {qualifier} {product_name}"
                if qualifier
                else f"the {product_name}"
            )
    return item_descriptions, product_descriptions


def _assert_runtime_scenarios_humanized(runtime_tasks: list[dict[str, Any]]) -> None:
    encoded = json.dumps(
        [task["user_scenario"] for task in runtime_tasks],
        ensure_ascii=False,
        sort_keys=True,
    )
    internal_ids = sorted(set(_RUNTIME_INTERNAL_ID_PATTERN.findall(encoded)))
    if internal_ids:
        raise ValueError(
            "runtime user scenarios expose internal user IDs: "
            f"{internal_ids[:10]}"
        )
    scripted_phrases = (
        "respond exactly",
        "forgot the 'R' at the beginning",
        "forgot the R at the beginning",
        "do not share this with the agent",
        "correct garden hose item id",
    )
    residual = [phrase for phrase in scripted_phrases if phrase.lower() in encoded.lower()]
    if residual:
        raise ValueError(f"runtime user scenarios retain benchmark scripting: {residual}")


def _runtime_task(
    task: dict[str, Any],
    *,
    authenticated_user_id: str,
    public_user_name: str,
    item_descriptions: dict[str, str],
    product_descriptions: dict[str, str],
) -> dict[str, Any]:
    scenario = task.get("user_scenario")
    if not scenario:
        raise ValueError(f"task {task.get('id')} has no user_scenario")
    return {
        "id": str(task["id"]),
        # Server-only account boundary. The runner does not place this field in
        # either model prompt; it supplies it when starting the task MCP server.
        "authenticated_user_id": authenticated_user_id,
        "description": task.get("description"),
        "user_scenario": _humanize_runtime_scenario(
            scenario,
            authenticated_user_id=authenticated_user_id,
            public_user_name=public_user_name,
            item_descriptions=item_descriptions,
            product_descriptions=product_descriptions,
        ),
        "initial_state": task.get("initial_state"),
    }


def _reference_action_semantics(
    source_db: dict[str, Any],
    source_tasks: list[dict[str, Any]],
    remapped_db: dict[str, Any],
    remapped_tasks: list[dict[str, Any]],
    replacements: dict[str, str],
    field_replacements: dict[str, dict[str, str]],
    identity_triples: dict[tuple[str, str, str], tuple[str, str, str]],
) -> dict[str, Any]:
    """Verify remapping preserves reference results and resulting database state."""

    # Import only inside the local preparation boundary. mcp_protocol sets this
    # before importing tau3 as well; repeating it here keeps this path pure.
    import os

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from tau2.domains.retail.data_model import RetailDB

    from .retail_tools import StudyRetailTools

    remapped_by_id = {str(task["id"]): task for task in remapped_tasks}
    outcomes: list[dict[str, Any]] = []
    mismatches: list[str] = []
    action_count = 0
    successful_result_count = 0
    error_action_count = 0
    result_mismatch_count = 0
    error_class_mismatch_count = 0
    final_state_mismatch_count = 0
    compiled_pattern = _compile_replacement_pattern(replacements)

    def jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return json.loads(json.dumps(value, ensure_ascii=False))

    for source_task in source_tasks:
        task_id = str(source_task["id"])
        remapped_task = remapped_by_id[task_id]
        source_actions = source_task.get("evaluation_criteria", {}).get("actions") or []
        remapped_actions = (
            remapped_task.get("evaluation_criteria", {}).get("actions") or []
        )
        if len(source_actions) != len(remapped_actions):
            mismatches.append(f"{task_id}:action_count")
            continue
        source_tools = StudyRetailTools(RetailDB.model_validate(source_db))
        remapped_tools = StudyRetailTools(RetailDB.model_validate(remapped_db))
        for index, (source_action, remapped_action) in enumerate(
            zip(source_actions, remapped_actions, strict=True)
        ):
            action_count += 1
            pair: dict[str, Any] = {
                "task": task_id,
                "index": index,
                "source_tool": source_action["name"],
                "remapped_tool": remapped_action["name"],
            }
            if source_action["name"] != remapped_action["name"]:
                pair["source_outcome"] = "not_run"
                pair["remapped_outcome"] = "not_run"
                mismatches.append(f"{task_id}:{index}:tool_name")
                outcomes.append(pair)
                continue

            def execute(tools: Any, action: dict[str, Any]) -> tuple[str, Any | None]:
                try:
                    result = tools.use_tool(
                        action["name"], **action.get("arguments", {})
                    )
                except Exception as exc:  # reference actions can intentionally fail
                    return f"error:{type(exc).__name__}", None
                return "success", jsonable(result)

            source_outcome, source_result = execute(source_tools, source_action)
            remapped_outcome, remapped_result = execute(remapped_tools, remapped_action)
            pair["source_outcome"] = source_outcome
            pair["remapped_outcome"] = remapped_outcome
            if pair["source_outcome"] != pair["remapped_outcome"]:
                mismatches.append(f"{task_id}:{index}:outcome")
                if source_outcome.startswith("error:") or remapped_outcome.startswith(
                    "error:"
                ):
                    error_class_mismatch_count += 1
            elif source_outcome == "success":
                successful_result_count += 1
                projected_source_result = _rewrite(
                    source_result,
                    replacements,
                    field_replacements,
                    identity_triples,
                    compiled_pattern=compiled_pattern,
                )
                pair["source_result_digest"] = digest_value(projected_source_result)
                pair["remapped_result_digest"] = digest_value(remapped_result)
                if projected_source_result != remapped_result:
                    result_mismatch_count += 1
                    mismatches.append(f"{task_id}:{index}:result")
            else:
                error_action_count += 1
            outcomes.append(pair)

        projected_source_state = _rewrite(
            source_tools.db.model_dump(mode="json"),
            replacements,
            field_replacements,
            identity_triples,
            replace_embedded=False,
        )
        _align_database_user_names(
            source_db,
            projected_source_state,
            replacements,
            identity_triples,
        )
        remapped_state = remapped_tools.db.model_dump(mode="json")
        source_state_digest = digest_value(projected_source_state)
        remapped_state_digest = digest_value(remapped_state)
        outcomes.append(
            {
                "task": task_id,
                "final_source_state_digest": source_state_digest,
                "final_remapped_state_digest": remapped_state_digest,
            }
        )
        if projected_source_state != remapped_state:
            final_state_mismatch_count += 1
            mismatches.append(f"{task_id}:final_state")

    if mismatches:
        raise ValueError(
            f"remapped reference-action semantics drifted: {mismatches[:10]}"
        )
    return {
        "tasks_checked": len(source_tasks),
        "reference_actions_checked": action_count,
        "successful_results_checked": successful_result_count,
        "intentional_errors_checked": error_action_count,
        "result_mismatches": result_mismatch_count,
        "error_class_mismatches": error_class_mismatch_count,
        "final_states_checked": len(source_tasks),
        "final_state_mismatches": final_state_mismatch_count,
        "semantic_mismatches": 0,
        "outcome_digest": digest_value(outcomes),
    }


def _assert_no_source_identity(
    values: Iterable[Any], replacements: dict[str, str]
) -> None:
    encoded = json.dumps(list(values), ensure_ascii=False, sort_keys=True)
    residual = []
    for old in replacements:
        if not old or (old.isdigit() and len(old) <= 5):
            continue
        prefix = r"(?<![A-Za-z0-9])" if old[0].isalnum() else ""
        suffix = r"(?![A-Za-z0-9])" if old[-1].isalnum() else ""
        if re.search(f"{prefix}{re.escape(old)}{suffix}", encoded):
            residual.append(old)
    if residual:
        sample = sorted(residual, key=len, reverse=True)[:5]
        raise ValueError(f"source identifiers remain after remap: {sample}")
    for pattern in _SENSITIVE_SOURCE_PATTERNS:
        for match in pattern.finditer(encoded):
            if pattern is _EMAIL_PATTERN and match.group(0).endswith(".invalid"):
                continue
            raise ValueError(f"source-shaped identity remains: {match.group(0)!r}")


def _assert_field_replacements_applied(
    source: Any,
    remapped: Any,
    field_replacements: dict[str, dict[str, str]],
) -> None:
    for field, replacements in field_replacements.items():
        source_values = [value for key, value in _walk_strings(source) if key == field]
        remapped_values = [
            value for key, value in _walk_strings(remapped) if key == field
        ]
        if len(source_values) != len(remapped_values):
            raise ValueError(f"{field} field count changed during remap")
        for original, rewritten in zip(source_values, remapped_values, strict=True):
            expected = replacements.get(original)
            if expected is None or rewritten != expected:
                raise ValueError(f"{field} replacement drifted for {original!r}")


def _assert_source_names_removed(
    source_db: dict[str, Any],
    source_tasks: list[dict[str, Any]],
    remapped_db: dict[str, Any],
    runtime_tasks: list[dict[str, Any]],
    private_truth: list[dict[str, Any]],
    replacements: dict[str, str],
) -> None:
    source_names = {
        (str(user["name"]["first_name"]), str(user["name"]["last_name"]))
        for user in source_db["users"].values()
    }
    encoded = json.dumps(
        (remapped_db, runtime_tasks, private_truth),
        ensure_ascii=False,
        sort_keys=True,
    )
    residual_full_names = [
        f"{first} {last}"
        for first, last in sorted(source_names)
        if re.search(
            rf"\b{re.escape(first)}\s+{re.escape(last)}\b",
            encoded,
        )
    ]
    structured_names = {
        (str(node["first_name"]), str(node["last_name"]))
        for value in (remapped_db, runtime_tasks, private_truth)
        for node in _walk_dicts(value)
        if {"first_name", "last_name"}.issubset(node)
    }
    residual_structured = sorted(source_names & structured_names)
    if residual_full_names or residual_structured:
        raise ValueError(
            "source names remain after remap: "
            f"{(residual_full_names + [str(value) for value in residual_structured])[:10]}"
        )

    remapped_by_id = {str(task["id"]): task for task in runtime_tasks}
    for source_task in source_tasks:
        source_user_id = _infer_primary_user_id(source_task, source_db)
        target_user_id = replacements[source_user_id]
        source_first = str(source_db["users"][source_user_id]["name"]["first_name"])
        target_first = str(remapped_db["users"][target_user_id]["name"]["first_name"])
        if source_first == target_first:
            continue
        known_info = str(
            remapped_by_id[str(source_task["id"])]
            .get("user_scenario", {})
            .get("instructions", {})
            .get("known_info")
            or ""
        )
        if re.search(rf"\b{re.escape(source_first)}\b", known_info):
            raise ValueError(
                f"task {source_task['id']} retains source first name {source_first!r}"
            )


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_dicts(child)


def prepare_source(tau_root: Path) -> dict[str, Any]:
    tau_root = tau_root.resolve()
    if _git_head(tau_root) != TAU3_COMMIT:
        raise ValueError(
            f"tau3 checkout must be exactly {TAU3_COMMIT}; found {_git_head(tau_root)}"
        )
    paths = _source_paths(tau_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned tau3 inputs: {missing}")

    source_db = load_json(paths["database"])
    source_tasks = load_json(paths["tasks"])
    if len(source_tasks) != TASK_COUNT:
        raise ValueError(
            f"expected {TASK_COUNT} retail tasks, found {len(source_tasks)}"
        )
    replacements, field_replacements, identity_triples = _build_replacements(
        source_db, source_tasks
    )
    remapped_db = _rewrite(
        source_db, replacements, field_replacements, identity_triples
    )
    # User name fields in the database do not include ZIP beside them, so set
    # them from the source user tuple after the general foreign-key rewrite.
    _align_database_user_names(
        source_db,
        remapped_db,
        replacements,
        identity_triples,
    )
    remapped_tasks = _rewrite(
        source_tasks, replacements, field_replacements, identity_triples
    )
    for source_task, remapped_task in zip(source_tasks, remapped_tasks, strict=True):
        _align_task_user_name(
            source_task,
            remapped_task,
            source_db,
            remapped_db,
            replacements,
        )
    _assert_field_replacements_applied(
        (source_db, source_tasks),
        (remapped_db, remapped_tasks),
        field_replacements,
    )
    semantic_validation = _reference_action_semantics(
        source_db,
        source_tasks,
        remapped_db,
        remapped_tasks,
        replacements,
        field_replacements,
        identity_triples,
    )
    item_descriptions, product_descriptions = _catalog_reference_descriptions(
        remapped_db
    )
    runtime_tasks = []
    for source_task, remapped_task in zip(source_tasks, remapped_tasks, strict=True):
        authenticated_user_id = replacements[
            _infer_primary_user_id(source_task, source_db)
        ]
        public_name = remapped_db["users"][authenticated_user_id]["name"]
        runtime_tasks.append(
            _runtime_task(
                remapped_task,
                authenticated_user_id=authenticated_user_id,
                public_user_name=(
                    f"{public_name['first_name']} {public_name['last_name']}"
                ),
                item_descriptions=item_descriptions,
                product_descriptions=product_descriptions,
            )
        )
    _assert_runtime_scenarios_humanized(runtime_tasks)
    private_truth = [
        {
            "id": str(task["id"]),
            "evaluation_criteria": task.get("evaluation_criteria"),
        }
        for task in remapped_tasks
    ]
    remapped_policy = _rewrite(
        paths["policy"].read_text(encoding="utf-8"),
        replacements,
        field_replacements,
        identity_triples,
    )
    user_guidelines = paths["user_guidelines"].read_text(encoding="utf-8")
    task_splits = load_json(paths["task_splits"])

    _assert_source_names_removed(
        source_db,
        source_tasks,
        remapped_db,
        runtime_tasks,
        private_truth,
        replacements,
    )
    _assert_no_source_identity(
        (remapped_db, runtime_tasks, private_truth, remapped_policy), replacements
    )

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(PREPARED_DIR / "db.json", remapped_db)
    write_json_atomic(PREPARED_DIR / "tasks.runtime.json", runtime_tasks)
    write_json_atomic(PREPARED_DIR / "task_splits.json", task_splits)
    (PREPARED_DIR / "policy.md").write_text(remapped_policy, encoding="utf-8")
    (PREPARED_DIR / "user-guidelines.md").write_text(user_guidelines, encoding="utf-8")
    write_json_atomic(PRIVATE_DIR / "task-truth.json", private_truth)
    replacement_ledger = {
        "global": replacements,
        "fields": field_replacements,
    }
    write_json_atomic(PRIVATE_DIR / "replacements.json", replacement_ledger)

    source_files = {
        label: {
            "path": path.relative_to(tau_root).as_posix(),
            "sha256": digest_file(path),
        }
        for label, path in paths.items()
    }
    prepared_files = {
        name: digest_file(PREPARED_DIR / name) for name in _PREPARED_SOURCE_FILES
    }
    private_files = {
        name: digest_file(PRIVATE_DIR / name) for name in _PRIVATE_SOURCE_FILES
    }
    from .retail_tools import TAU3_COMPATIBILITY_FIXES

    lock = {
        "schema_version": 2,
        "source": {
            "repository": TAU3_REPOSITORY,
            "version": TAU3_VERSION,
            "tag_object": TAU3_TAG_OBJECT,
            "commit": TAU3_COMMIT,
            "license": TAU3_LICENSE,
            "files": source_files,
        },
        "remap": {
            "algorithm": "retail-support-public-safe-v3",
            "replacement_count": len(replacements)
            + sum(len(values) for values in field_replacements.values()),
            "replacement_digest": digest_value(replacement_ledger),
        },
        "runtime_compatibility": list(TAU3_COMPATIBILITY_FIXES),
        "counts": {
            "tasks": len(runtime_tasks),
            "users": len(remapped_db["users"]),
            "orders": len(remapped_db["orders"]),
            "products": len(remapped_db["products"]),
        },
        "validation": semantic_validation,
        "prepared_files": prepared_files,
        "private_files": private_files,
    }
    lock["source_lock_sha256"] = digest_value(lock)
    write_json_atomic(SOURCE_LOCK_PATH, lock)
    return lock


def verify_prepared_source() -> dict[str, Any]:
    lock = load_json(SOURCE_LOCK_PATH)
    claimed = lock.get("source_lock_sha256", "")
    unsigned = {**lock}
    unsigned.pop("source_lock_sha256", None)
    if claimed != digest_value(unsigned):
        raise ValueError("source lock digest mismatch")
    from .retail_tools import TAU3_COMPATIBILITY_FIXES

    if lock.get("schema_version") != 2:
        raise ValueError("source lock schema is not the reviewed compatibility schema")
    if lock.get("runtime_compatibility") != list(TAU3_COMPATIBILITY_FIXES):
        raise ValueError("source lock compatibility fixes differ from runtime")
    expected_identity = {
        "repository": TAU3_REPOSITORY,
        "version": TAU3_VERSION,
        "tag_object": TAU3_TAG_OBJECT,
        "commit": TAU3_COMMIT,
        "license": TAU3_LICENSE,
    }
    actual_identity = {
        key: lock.get("source", {}).get(key) for key in expected_identity
    }
    if actual_identity != expected_identity:
        raise ValueError("source identity does not match the pinned tau3 release")
    if set(lock.get("prepared_files", {})) != set(_PREPARED_SOURCE_FILES):
        raise ValueError("source lock prepared-file membership mismatch")
    if set(lock.get("private_files", {})) != set(_PRIVATE_SOURCE_FILES):
        raise ValueError("source lock private-file membership mismatch")
    for name, expected in lock["prepared_files"].items():
        path = PREPARED_DIR / name
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"prepared file drift: {name}")
    for name, expected in lock["private_files"].items():
        path = PRIVATE_DIR / name
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"private file drift: {name}")
    if lock["counts"]["tasks"] != TASK_COUNT:
        raise ValueError("source lock task count mismatch")
    return lock
