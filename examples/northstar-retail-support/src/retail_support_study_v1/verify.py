from __future__ import annotations

import math
import re
from typing import Any

from .canonical import digest_value, load_json, require_hex_digest
from .constants import (
    CANARY_COORDINATE_COUNT,
    CANARY_COST_CAP_USD,
    CANARY_TASK_GROUPS,
    CANARY_TASK_IDS,
    FORBIDDEN_PUBLIC_KEY_FRAGMENTS,
    FULL_COORDINATE_COUNT,
    FULL_COST_CAP_USD,
    LEGACY_CANARY_COORDINATE_COUNT,
    MODEL_ROUTE_COUNTS_CANARY,
    MODEL_ROUTE_COUNTS_FULL,
    RUNS_DIR,
    STUDY_ID,
    TASK_COUNT,
    TRIAL_INDICES,
)
from .cost_policy import (
    bundle_has_disallowed_model_evidence,
    continues_frozen_attempt_failures,
)
from .mcp_protocol import MCPProcessClient
from .preview import coordinate_matrix, verify_preview
from .recovery import reviewed_canary_recovery_contract
from .remap import verify_prepared_source
from .runtime_snapshot import verify_runtime_snapshot
from .schemas import (
    AgentProfile,
    BehaviorProfile,
    ConversationBundle,
    Coordinate,
    ModelRoute,
    RunLedger,
)
from .traffic_config import AGENT_ROUTER_POLICY, MODEL_PROBE_POLICY

_SECRET_PATTERNS = (
    re.compile(r"\b(?:api|secret|token|sk)[_-][A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
)
_SIMULATOR_FALSE_NEGATIVE_REASON = (
    "RuntimeError: simulator failed the bounded scenario-conformance repair"
)
_SIMULATOR_CONDITIONAL_DISCLOSURE_REASON = (
    "simulator_conditional_clause_revealed_before_confirmation_boundary"
)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _scan_public_value(value: Any, *, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(str(key))
            if any(
                _normalized_key(fragment) in normalized
                for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            ):
                errors.append(f"forbidden public key at {path}.{key}")
            errors.extend(_scan_public_value(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_scan_public_value(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                errors.append(f"secret-shaped value at {path}")
    return errors


def _costs_match(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return abs(actual - expected) <= 1e-12


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonnegative_cost(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be a nonnegative finite number")
    return parsed


def _verify_canary_execution_lineage(
    preview: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Verify and summarize the retained identity of a replacement canary."""

    execution = preview.get("canary_execution")
    if execution is None and allow_legacy:
        return {
            "execution_generation": 1,
            "is_replacement": False,
        }
    if not isinstance(execution, dict) or execution.get("schema_version") != 1:
        raise ValueError("preview has no valid canary execution lineage")
    claimed = require_hex_digest(
        str(execution.get("lock_sha256") or ""),
        label="canary execution lock",
    )
    if claimed != digest_value({**execution, "lock_sha256": ""}):
        raise ValueError("preview canary execution lock digest mismatch")
    if execution.get("study_id") != STUDY_ID or execution.get("cohort") != "canary":
        raise ValueError("preview canary execution identity mismatch")
    generation = _nonnegative_int(
        execution.get("execution_generation"),
        label="canary execution generation",
    )
    if generation < 2:
        raise ValueError("replacement canary execution generation must be at least 2")
    reason = execution.get("reason")
    recovery_contract = reviewed_canary_recovery_contract(
        generation=generation,
        reason=reason,
    )
    if execution.get("mode") != recovery_contract["mode"]:
        raise ValueError("canary recovery mode mismatch")

    replacement = execution.get("replacement_policy")
    expected_replacement = {
        "planned_coordinate_count": LEGACY_CANARY_COORDINATE_COUNT,
        "reuse_prior_attempts": False,
        "reuse_prior_bundles": False,
        "scope": "rerun_all_36_canary_coordinates",
    }
    if replacement != expected_replacement:
        raise ValueError("canary replacement policy mismatch")

    supersedes = execution.get("supersedes")
    if (
        not isinstance(supersedes, dict)
        or supersedes.get("status") != recovery_contract["predecessor_status"]
    ):
        raise ValueError(
            "replacement canary predecessor status does not match recovery"
        )

    superseded_generation_raw = supersedes.get("execution_generation")
    if superseded_generation_raw is None:
        # Generation 2 was sealed before predecessor generation and lock fields
        # were added to the lineage schema. Retain that exact preview as a
        # verifiable transition from the implicit generation 1.
        if generation != 2:
            raise ValueError(
                "replacement canary must name its superseded execution generation"
            )
        superseded_generation = 1
        superseded_execution_lock = None
    else:
        superseded_generation = _nonnegative_int(
            superseded_generation_raw,
            label="superseded canary execution generation",
        )
        if superseded_generation != generation - 1:
            raise ValueError("canary execution generations are not monotonic")
        superseded_execution_lock = require_hex_digest(
            str(supersedes.get("execution_lock_sha256") or ""),
            label="superseded canary execution lock",
        )
        if superseded_execution_lock == claimed:
            raise ValueError("replacement canary cannot supersede its own lock")

    superseded_execution_lock_file: str | None = None
    uncertainty_reasons: list[str] = []
    if generation >= 3:
        raw_lock_file_digest = supersedes.get("execution_lock_file_sha256")
        if not isinstance(raw_lock_file_digest, str):
            raise ValueError(
                "superseded canary execution-lock file digest is required"
            )
        superseded_execution_lock_file = require_hex_digest(
            raw_lock_file_digest,
            label="superseded canary execution-lock file",
        )
        if raw_lock_file_digest != superseded_execution_lock_file:
            raise ValueError(
                "superseded canary execution-lock file must use lowercase SHA-256"
            )
        raw_uncertainty_reasons = supersedes.get("cost_uncertainty_reasons")
        if not isinstance(raw_uncertainty_reasons, list) or not all(
            isinstance(item, str) and item.strip()
            for item in raw_uncertainty_reasons
        ):
            raise ValueError(
                "superseded canary cost-uncertainty reasons are incomplete"
            )
        uncertainty_reasons = list(raw_uncertainty_reasons)
        if reason == "provider_timeout_created_ambiguous_cost_reserve":
            if not uncertainty_reasons:
                raise ValueError(
                    "superseded canary cost-uncertainty reasons are incomplete"
                )
            if not any("APITimeoutError" in item for item in uncertainty_reasons):
                raise ValueError(
                    "timeout recovery has no retained APITimeoutError evidence"
                )
        if reason == "provider_internal_server_error_after_bounded_retries":
            if not any(
                "InternalServerError" in item for item in uncertainty_reasons
            ):
                raise ValueError(
                    "internal-server recovery has no retained "
                    "InternalServerError uncertainty evidence"
                )

    digest_fields = (
        "preview_sha256",
        "runtime_manifest_sha256",
        "runtime_manifest_file_sha256",
        "ledger_sha256",
        "ledger_file_sha256",
    )
    for field in digest_fields:
        require_hex_digest(
            str(supersedes.get(field) or ""),
            label=f"superseded canary {field}",
        )
    if supersedes["preview_sha256"] == preview.get("preview_sha256"):
        raise ValueError("replacement canary cannot supersede its own preview")
    planned = _nonnegative_int(
        supersedes.get("planned_coordinate_count"),
        label="superseded canary planned-coordinate count",
    )
    if planned != LEGACY_CANARY_COORDINATE_COUNT:
        raise ValueError("superseded canary planned-coordinate count mismatch")
    completed = _nonnegative_int(
        supersedes.get("completed_coordinate_count"),
        label="superseded canary completed-coordinate count",
    )
    unavailable = supersedes.get("unavailable")
    unavailable_ids = supersedes.get("unavailable_coordinate_ids")
    unavailable_count = _nonnegative_int(
        supersedes.get("unavailable_coordinate_count"),
        label="superseded canary unavailable-coordinate count",
    )
    if (
        not isinstance(unavailable, dict)
        or not isinstance(unavailable_ids, list)
        or unavailable_ids != sorted(unavailable)
        or unavailable_count != len(unavailable)
        or not all(isinstance(key, str) and key for key in unavailable)
        or not all(isinstance(reason, str) and reason for reason in unavailable.values())
    ):
        raise ValueError("superseded canary unavailable evidence mismatch")
    terminal = completed + unavailable_count
    if terminal > planned:
        raise ValueError("superseded canary terminal-coordinate count exceeds plan")
    observed = _nonnegative_cost(
        supersedes.get("observed_cost_usd"),
        label="superseded canary observed cost",
    )
    reserve = _nonnegative_cost(
        supersedes.get("ambiguous_cost_reserve_usd"),
        label="superseded canary ambiguous cost reserve",
    )
    exposure = _nonnegative_cost(
        supersedes.get("cost_exposure_upper_bound_usd"),
        label="superseded canary cost exposure",
    )
    if not _costs_match(round(observed + reserve, 12), exposure):
        raise ValueError("superseded canary cost evidence does not reconcile")
    if reserve == 0 and unavailable_count == 0:
        raise ValueError("superseded canary has no retained uncertainty evidence")
    if reason == "simulator_contract_false_negative_after_visible_correction":
        if completed != 5 or unavailable_count != 1:
            raise ValueError(
                "simulator-contract recovery predecessor must retain 5 complete "
                "and 1 unavailable coordinate"
            )
        if reserve != 0 or uncertainty_reasons:
            raise ValueError(
                "simulator-contract recovery cannot retain cost uncertainty"
            )
        if set(unavailable.values()) != {_SIMULATOR_FALSE_NEGATIVE_REASON}:
            raise ValueError(
                "simulator-contract recovery has no exact false-negative evidence"
            )
    elif reason == _SIMULATOR_CONDITIONAL_DISCLOSURE_REASON:
        if completed != 29 or unavailable_count != 1 or planned - terminal != 6:
            raise ValueError(
                "conditional-disclosure recovery predecessor must retain 29 "
                "complete, 1 unavailable, and 6 pending coordinates"
            )
        if reserve != 0 or uncertainty_reasons:
            raise ValueError(
                "conditional-disclosure recovery cannot retain cost uncertainty"
            )
        if set(unavailable.values()) != {_SIMULATOR_FALSE_NEGATIVE_REASON}:
            raise ValueError(
                "conditional-disclosure recovery has no exact simulator evidence"
            )
    elif reason == "provider_internal_server_error_after_bounded_retries":
        if not any("InternalServerError" in item for item in unavailable.values()):
            raise ValueError(
                "internal-server recovery has no matching unavailable evidence"
            )

    return {
        "execution_generation": generation,
        "is_replacement": True,
        "lock_sha256": claimed,
        "mode": execution["mode"],
        "reason": reason,
        "replacement_scope": replacement["scope"],
        "planned_coordinates": replacement["planned_coordinate_count"],
        "reuses_prior_attempts": replacement["reuse_prior_attempts"],
        "reuses_prior_bundles": replacement["reuse_prior_bundles"],
        "superseded_execution_generation": superseded_generation,
        "superseded_execution_lock_sha256": superseded_execution_lock,
        "superseded_execution_lock_file_sha256": (
            superseded_execution_lock_file
        ),
        "superseded_preview_sha256": supersedes["preview_sha256"],
        "superseded_runtime_manifest_sha256": supersedes[
            "runtime_manifest_sha256"
        ],
        "superseded_ledger_sha256": supersedes["ledger_sha256"],
        "superseded_status": supersedes["status"],
        "superseded_completed_coordinates": completed,
        "superseded_unavailable_coordinates": unavailable_count,
        "superseded_terminal_coordinates": terminal,
        "superseded_pending_coordinates": planned - terminal,
        "superseded_observed_cost_usd": observed,
        "superseded_ambiguous_cost_reserve_usd": reserve,
        "superseded_cost_exposure_upper_bound_usd": exposure,
        "superseded_cost_uncertainty_reasons": uncertainty_reasons,
    }


def verify_static() -> dict[str, Any]:
    source = verify_prepared_source()
    preview = verify_preview()
    runtime = verify_runtime_snapshot(preview["preview_sha256"])
    canary_execution = _verify_canary_execution_lineage(
        preview,
        allow_legacy=True,
    )
    if dict(runtime.preview).get("canary_execution") != preview.get("canary_execution"):
        raise ValueError("runtime snapshot canary execution lineage mismatch")
    coordinates = preview["cohorts"]["full"]["coordinates"]
    if len(coordinates) != FULL_COORDINATE_COUNT:
        raise ValueError("full coordinate count mismatch")
    if len({row["coordinate_id"] for row in coordinates}) != FULL_COORDINATE_COUNT:
        raise ValueError("coordinate IDs are not unique")
    task_ids = [str(task["id"]) for task in runtime.tasks]
    if len(set(task_ids)) != TASK_COUNT:
        raise ValueError("logical task count mismatch")
    expected_full = [
        coordinate.model_dump(mode="json")
        for coordinate in coordinate_matrix("full", task_ids)
    ]
    if coordinates != expected_full:
        raise ValueError("full coordinates do not match the exact trial matrix")

    canary = preview["cohorts"]["canary"]["coordinates"]
    expected_canary = [
        coordinate.model_dump(mode="json")
        for coordinate in coordinate_matrix("canary", list(CANARY_TASK_IDS))
    ]
    if canary != expected_canary:
        raise ValueError("canary coordinates do not match the exact trial assignment")
    if len(canary) != CANARY_COORDINATE_COUNT:
        raise ValueError("canary coordinate count mismatch")
    if preview["cohorts"]["full"].get("trial_indices") != list(TRIAL_INDICES):
        raise ValueError("full trial registration mismatch")
    if preview["cohorts"]["canary"].get("trial_indices") != list(TRIAL_INDICES):
        raise ValueError("canary trial registration mismatch")

    traffic = preview.get("traffic")
    if not isinstance(traffic, dict):
        raise ValueError("preview has no sealed traffic contract")
    agent_profiles = [
        AgentProfile.model_validate(value)
        for value in traffic.get("agent_profiles", [])
    ]
    if len(agent_profiles) != 3 or len(
        {profile.profile_id for profile in agent_profiles}
    ) != 3:
        raise ValueError("preview must bind exactly three agent profiles")
    if traffic.get("agent_router") != AGENT_ROUTER_POLICY:
        raise ValueError("preview agent-router policy mismatch")
    model_routes = [
        ModelRoute.model_validate(value)
        for value in traffic.get("model_routes", [])
    ]
    if traffic.get("model_probe_policy") != MODEL_PROBE_POLICY:
        raise ValueError("preview model-probe policy mismatch")
    BehaviorProfile.model_validate(traffic.get("behavior_profile"))
    route_ids = {route.route_id for route in model_routes}
    if len(model_routes) != 5 or len(route_ids) != 5:
        raise ValueError("preview must bind exactly five model routes")

    expected_route_counts = {
        "full": dict(MODEL_ROUTE_COUNTS_FULL),
        "canary": dict(MODEL_ROUTE_COUNTS_CANARY),
    }
    for cohort, rows in (("full", coordinates), ("canary", canary)):
        observed = {
            route_id: sum(row.get("model_route_id") == route_id for row in rows)
            for route_id in route_ids
        }
        if observed != expected_route_counts[cohort]:
            raise ValueError(f"{cohort} model-route counts mismatch")
        if preview["cohorts"][cohort].get("model_route_counts") != observed:
            raise ValueError(f"{cohort} model-route summary mismatch")
    if preview["cohorts"]["canary"].get("selection_policy", {}).get(
        "task_group_counts"
    ) != {group: len(ids) for group, ids in CANARY_TASK_GROUPS.items()}:
        raise ValueError("canary agent-stratum counts mismatch")
    if preview.get("cost_caps_usd") != {
        "canary": CANARY_COST_CAP_USD,
        "full": FULL_COST_CAP_USD,
    }:
        raise ValueError("preview cost caps mismatch")

    with MCPProcessClient(
        db_path=runtime.db_path,
        policy_path=runtime.policy_path,
        package_source_dir=runtime.package_source_dir,
    ) as mcp:
        tools = mcp.list_openai_tools()
    tool_names = sorted(tool["function"]["name"] for tool in tools)
    if tools != preview["mcp"]["tools"]:
        raise ValueError("live MCP tool schemas do not match preview")
    if tool_names != sorted(preview["mcp"]["tool_names"]):
        raise ValueError("live MCP manifest does not match preview")

    return {
        "source_lock_sha256": source["source_lock_sha256"],
        "preview_sha256": preview["preview_sha256"],
        "runtime_snapshot_manifest_sha256": runtime.manifest["manifest_sha256"],
        "logical_tasks": len(task_ids),
        "trials": len(TRIAL_INDICES),
        "full_coordinates": len(coordinates),
        "canary_coordinates": len(canary),
        "agent_profiles": len(agent_profiles),
        "providers": len({route.provider_name for route in model_routes}),
        "model_routes": len(model_routes),
        "mcp_tools": len(tool_names),
        "canary_execution": canary_execution,
        "status": "ready_for_canary",
    }


def verify_cohort(
    cohort: str,
    *,
    preview_sha256: str | None = None,
) -> dict[str, Any]:
    if preview_sha256 is None:
        preview = verify_preview()
        runtime = verify_runtime_snapshot(preview["preview_sha256"])
    else:
        # An explicit selector is a read-only audit of that retained generation.
        # It must re-hash the sealed bytes, but it need not equal today's source.
        runtime = verify_runtime_snapshot(
            preview_sha256,
            require_current_code=False,
        )
        preview = dict(runtime.preview)
    canary_execution = _verify_canary_execution_lineage(
        preview,
        allow_legacy=True,
    )
    continue_frozen_attempt_failures = continues_frozen_attempt_failures(preview)
    cohort_dir = RUNS_DIR / cohort / preview["preview_sha256"]
    ledger_path = cohort_dir / "ledger.json"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("run ledger must be a regular file")
    ledger = RunLedger.model_validate(load_json(ledger_path))
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("run ledger content digest mismatch")
    if ledger.study_id != STUDY_ID or ledger.cohort != cohort:
        raise ValueError("run ledger identity mismatch")
    if ledger.preview_sha256 != preview["preview_sha256"]:
        raise ValueError("run ledger preview mismatch")
    coordinates = [
        Coordinate.model_validate(row)
        for row in preview["cohorts"][cohort]["coordinates"]
    ]
    expected_order = [row.coordinate_id for row in coordinates]
    expected = set(expected_order)
    coordinates_by_id = {row.coordinate_id: row for row in coordinates}
    if ledger.planned_coordinate_ids != expected_order:
        raise ValueError("run ledger planned coordinates mismatch")
    if len(ledger.completed_coordinate_ids) != len(
        set(ledger.completed_coordinate_ids)
    ):
        raise ValueError("run ledger repeats a completed coordinate")
    if not set(ledger.completed_coordinate_ids) <= expected:
        raise ValueError("run ledger contains an unexpected completed coordinate")
    if not set(ledger.unavailable) <= expected:
        raise ValueError("run ledger contains an unexpected unavailable coordinate")
    if set(ledger.completed_coordinate_ids) & set(ledger.unavailable):
        raise ValueError("coordinate is both complete and unavailable")

    errors: list[str] = []
    bundle_dir = cohort_dir / "bundles"
    attempt_dir = cohort_dir / "attempts"
    bundle_files = (
        {path.stem: path for path in bundle_dir.glob("*.json")}
        if bundle_dir.is_dir()
        else {}
    )
    attempt_files = (
        {path.stem: path for path in attempt_dir.glob("*.json")}
        if attempt_dir.is_dir()
        else {}
    )
    if not set(bundle_files) <= expected:
        raise ValueError("bundle directory contains an unexpected coordinate")
    if not set(attempt_files) <= expected:
        raise ValueError("attempt directory contains an unexpected coordinate")
    if set(ledger.bundle_digests) != set(bundle_files):
        raise ValueError("ledger bundle digests do not match frozen bundle files")

    observed_cost = 0.0
    ambiguous_cost = 0.0
    uncertainty_reasons: list[str] = []
    invalid_responses: list[str] = []
    disallowed_model_evidence: list[str] = []
    for coordinate_id in expected_order:
        path = bundle_files.get(coordinate_id)
        if path is None:
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"bundle is not a regular file: {coordinate_id}")
            continue
        claimed = ledger.bundle_digests[coordinate_id]
        bundle = ConversationBundle.model_validate(load_json(path))
        if bundle.coordinate != coordinates_by_id[coordinate_id]:
            errors.append(f"bundle coordinate mismatch: {coordinate_id}")
        if bundle.bundle_sha256 != claimed:
            errors.append(f"bundle ledger digest mismatch: {coordinate_id}")
        if bundle.bundle_sha256 != digest_value(bundle.unsigned_payload()):
            errors.append(f"bundle content digest mismatch: {coordinate_id}")
        if bundle.status == "complete":
            if coordinate_id not in ledger.completed_coordinate_ids:
                errors.append(f"complete bundle absent from ledger: {coordinate_id}")
        else:
            if ledger.unavailable.get(coordinate_id) != bundle.unavailable_reason:
                errors.append(f"unavailable bundle reason mismatch: {coordinate_id}")

        observed_values = [
            float(call.usage.observed_cost_usd)
            for call in bundle.model_calls
            if call.usage.observed_cost_usd is not None
        ]
        call_observed = round(sum(observed_values), 12)
        call_ambiguous = round(
            sum(call.ambiguous_cost_reserve_usd for call in bundle.model_calls),
            12,
        )
        call_reasons = [
            reason
            for call in bundle.model_calls
            for reason in call.cost_uncertainty_reasons
        ]
        observed_cost = round(observed_cost + call_observed, 12)
        ambiguous_cost = round(ambiguous_cost + call_ambiguous, 12)
        uncertainty_reasons.extend(call_reasons)
        invalid_responses.extend(
            call.response_validation_error
            for call in bundle.model_calls
            if call.response_validation_error
        )
        if bundle_has_disallowed_model_evidence(bundle):
            disallowed_model_evidence.append(coordinate_id)
        expected_bundle_cost = call_observed if observed_values else None
        if not _costs_match(
            bundle.usage.observed_cost_usd,
            expected_bundle_cost,
        ):
            errors.append(f"bundle observed cost mismatch: {coordinate_id}")

        attempt_path = attempt_files.get(coordinate_id)
        if attempt_path is None:
            errors.append(f"bundle has no attempt receipt: {coordinate_id}")
        elif attempt_path.is_symlink() or not attempt_path.is_file():
            errors.append(f"attempt receipt is not a regular file: {coordinate_id}")
        else:
            attempt = load_json(attempt_path)
            claimed_attempt = str(attempt.get("attempt_sha256") or "")
            if claimed_attempt != digest_value({**attempt, "attempt_sha256": ""}):
                errors.append(f"attempt receipt digest mismatch: {coordinate_id}")
            if attempt.get("study_id") != STUDY_ID:
                errors.append(f"attempt study mismatch: {coordinate_id}")
            if attempt.get("preview_sha256") != preview["preview_sha256"]:
                errors.append(f"attempt preview mismatch: {coordinate_id}")
            if attempt.get("coordinate_id") != coordinate_id:
                errors.append(f"attempt coordinate ID mismatch: {coordinate_id}")
            if attempt.get("coordinate") != bundle.coordinate.model_dump(mode="json"):
                errors.append(f"attempt coordinate mismatch: {coordinate_id}")
            expected_attempt_status = bundle.status
            if (
                not continue_frozen_attempt_failures
                and (
                    call_ambiguous
                    or any(
                        call.response_validation_error
                        for call in bundle.model_calls
                    )
                )
            ) or coordinate_id in disallowed_model_evidence:
                expected_attempt_status = "uncertain"
            expected_attempt_cost = round(call_observed + call_ambiguous, 12)
            for field, expected_value in (
                ("status", expected_attempt_status),
                ("bundle_sha256", bundle.bundle_sha256),
                ("observed_cost_usd", call_observed),
                ("ambiguous_cost_reserve_usd", call_ambiguous),
                ("cost_exposure_upper_bound_usd", expected_attempt_cost),
                ("cost_uncertainty_reasons", call_reasons),
            ):
                if attempt.get(field) != expected_value:
                    errors.append(f"attempt/bundle {field} mismatch: {coordinate_id}")
        public_projection = {
            "conversation_id": bundle.conversation_id,
            "messages": bundle.messages,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "error_type": call.error_type,
                    "error_message": call.error_message,
                }
                for call in bundle.tool_calls
            ],
        }
        errors.extend(_scan_public_value(public_projection))

    terminal = set(ledger.completed_coordinate_ids) | set(ledger.unavailable)
    unbundled_unavailable = set(ledger.unavailable) - set(bundle_files)
    unknown_cost_coordinates: list[str] = []
    unknown_cost_reasons: list[str] = []
    for coordinate_id in sorted(unbundled_unavailable):
        reason = ledger.unavailable[coordinate_id]
        attempt_path = attempt_files.get(coordinate_id)
        if "prior paid attempt has no frozen bundle" in reason:
            if attempt_path is None:
                errors.append(
                    f"uncertain coordinate has no attempt receipt: {coordinate_id}"
                )
                continue
            attempt = load_json(attempt_path)
            if attempt.get("attempt_sha256") != digest_value(
                {**attempt, "attempt_sha256": ""}
            ):
                errors.append(f"attempt receipt digest mismatch: {coordinate_id}")
            if attempt.get("study_id") != STUDY_ID:
                errors.append(f"attempt study mismatch: {coordinate_id}")
            if attempt.get("preview_sha256") != preview["preview_sha256"]:
                errors.append(f"attempt preview mismatch: {coordinate_id}")
            if attempt.get("coordinate_id") != coordinate_id:
                errors.append(f"attempt coordinate ID mismatch: {coordinate_id}")
            if attempt.get("coordinate") != coordinates_by_id[coordinate_id].model_dump(
                mode="json"
            ):
                errors.append(f"attempt coordinate mismatch: {coordinate_id}")
            if attempt.get("status") != "uncertain" or not attempt.get(
                "cost_exposure_unknown"
            ):
                errors.append(
                    f"unfrozen paid attempt is not marked uncertain: {coordinate_id}"
                )
            unknown_cost_coordinates.append(coordinate_id)
            unknown_cost_reasons.append(
                "unreconciled paid attempt reserved to the remaining "
                f"cohort cap: {coordinate_id}"
            )
        elif reason == "cohort cost cap reached":
            if attempt_path is not None:
                errors.append(
                    f"budget-blocked coordinate has an attempt: {coordinate_id}"
                )
        else:
            errors.append(f"unavailable coordinate has no bundle: {coordinate_id}")

    for coordinate_id, attempt_path in attempt_files.items():
        if coordinate_id in bundle_files or coordinate_id in unbundled_unavailable:
            continue
        attempt = load_json(attempt_path)
        if not (ledger.status == "running" and attempt.get("status") == "started"):
            errors.append(f"attempt receipt has no ledger outcome: {coordinate_id}")

    if unknown_cost_coordinates:
        cap = float(preview["cost_caps_usd"][cohort])
        expected_ambiguous = round(cap - observed_cost, 12)
        if ledger.cost_exposure_upper_bound_usd != cap:
            errors.append("unknown paid attempt is not reserved to the cohort cap")
    else:
        expected_ambiguous = ambiguous_cost
    if abs(ledger.observed_cost_usd - observed_cost) > 1e-12:
        errors.append("ledger observed cost does not reconcile frozen model calls")
    if abs(ledger.ambiguous_cost_reserve_usd - expected_ambiguous) > 1e-12:
        errors.append("ledger ambiguous reserve does not reconcile paid attempts")
    if ledger.cost_uncertainty_reasons != (uncertainty_reasons + unknown_cost_reasons):
        errors.append("ledger cost uncertainty reasons do not reconcile model calls")

    if ledger.status == "complete":
        if terminal != expected:
            errors.append("complete ledger does not reconcile every planned coordinate")
        if (
            (
                not continue_frozen_attempt_failures
                and (ledger.ambiguous_cost_reserve_usd or invalid_responses)
            )
            or disallowed_model_evidence
        ):
            errors.append("complete ledger contains uncertain model evidence")
    elif ledger.status == "paused":
        if terminal == expected:
            errors.append("paused ledger has no remaining coordinate")
        if (
            (
                not continue_frozen_attempt_failures
                and (ledger.ambiguous_cost_reserve_usd or invalid_responses)
            )
            or disallowed_model_evidence
        ):
            errors.append("paused ledger contains uncertain model evidence")
    elif ledger.status == "blocked_budget":
        if not any(
            reason == "cohort cost cap reached"
            or "cost cap would be exceeded" in reason
            for reason in ledger.unavailable.values()
        ):
            errors.append("blocked-budget ledger has no budget boundary")
        if ledger.ambiguous_cost_reserve_usd and (
            not continue_frozen_attempt_failures
            or bool(disallowed_model_evidence)
        ):
            errors.append("blocked-budget ledger contains ambiguous cost")
    elif ledger.status == "uncertain":
        if not (
            (
                not continue_frozen_attempt_failures
                and (ledger.ambiguous_cost_reserve_usd or invalid_responses)
            )
            or disallowed_model_evidence
            or unknown_cost_coordinates
        ):
            errors.append("uncertain ledger has no uncertainty evidence")
    elif ledger.status == "running" and terminal == expected:
        errors.append("running ledger already reconciles the entire plan")

    if errors:
        raise ValueError(f"cohort verification failed: {errors[:10]}")
    return {
        "cohort": cohort,
        "preview_sha256": preview["preview_sha256"],
        "runtime_snapshot_manifest_sha256": runtime.manifest["manifest_sha256"],
        "planned": len(expected),
        "completed": len(ledger.completed_coordinate_ids),
        "unavailable": len(ledger.unavailable),
        "observed_cost_usd": ledger.observed_cost_usd,
        "ambiguous_cost_reserve_usd": ledger.ambiguous_cost_reserve_usd,
        "cost_exposure_upper_bound_usd": ledger.cost_exposure_upper_bound_usd,
        "ledger_sha256": ledger.ledger_sha256,
        "canary_execution": canary_execution,
        "emission_eligible": ledger.status == "complete",
        "status": ledger.status,
    }
