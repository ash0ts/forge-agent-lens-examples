from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools

from retail_support_study_v1 import constants, preview, remap, review, runner, verify
from retail_support_study_v1.canonical import digest_value, load_json
from retail_support_study_v1.pricing import (
    CostBudget,
    conservative_request_cost_usd,
)
from retail_support_study_v1.schemas import RunLedger

ROOT = constants.PACKAGE_ROOT


def _all_true_review_row(coordinate_id: str) -> dict[str, Any]:
    return {
        "coordinate_id": coordinate_id,
        "run_status": "complete",
        "natural_progressive_dialogue": True,
        "no_hidden_truth_leakage": True,
        "tool_calling_valid": True,
        "mutation_confirmation_valid": True,
        "simulator_conformance_repair_count": 0,
        "simulator_conformance_valid": True,
        "terminal_response_valid": True,
        "review_notes": "reviewed",
    }


def test_source_lock_names_exact_annotated_tag_and_peeled_commit() -> None:
    lock = load_json(ROOT / "source.lock.json")

    assert constants.TAU3_VERSION == "1.0.1"
    assert constants.TAU3_TAG_OBJECT == ("b711c1ead46f55111bf765cf44d5da8bacc2d28c")
    assert constants.TAU3_COMMIT == ("fc0055dc4e0a316c3f83133267fbd6faaa770992")
    assert constants.TAU3_TAG_OBJECT != constants.TAU3_COMMIT
    assert lock["schema_version"] == 2
    assert lock["runtime_compatibility"] == [
        {
            "id": "tau3-v1.0.1-multi-item-variant-binding",
            "tool": "modify_pending_order_items",
            "upstream_version": "1.0.1",
            "behavior": (
                "bind each replacement item's price and options to its own variant "
                "instead of reusing the final validated variant"
            ),
        }
    ]
    assert lock["source"] == {
        "repository": constants.TAU3_REPOSITORY,
        "version": constants.TAU3_VERSION,
        "tag_object": constants.TAU3_TAG_OBJECT,
        "commit": constants.TAU3_COMMIT,
        "license": "MIT",
        "files": lock["source"]["files"],
    }
    assert lock["source"]["files"]["tasks"]["sha256"] == (
        "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8"
    )
    assert lock["source"]["files"]["database"]["sha256"] == (
        "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"
    )


@pytest.mark.parametrize("field", ["tag_object", "commit"])
def test_prepared_source_rejects_rehashed_source_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    prepared_dir = tmp_path / "prepared"
    private_dir = tmp_path / "private"
    prepared_dir.mkdir()
    private_dir.mkdir()

    lock = load_json(ROOT / "source.lock.json")
    for name in lock["prepared_files"]:
        (prepared_dir / name).write_bytes((ROOT / "prepared" / name).read_bytes())
    for name in lock["private_files"]:
        (private_dir / name).write_bytes((ROOT / "private" / name).read_bytes())

    lock["source"][field] = "0" * 40
    unsigned = {**lock}
    unsigned.pop("source_lock_sha256", None)
    lock["source_lock_sha256"] = digest_value(unsigned)
    lock_path = tmp_path / "source.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    monkeypatch.setattr(remap, "SOURCE_LOCK_PATH", lock_path)
    monkeypatch.setattr(remap, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(remap, "PRIVATE_DIR", private_dir)

    with pytest.raises(ValueError, match="source identity"):
        remap.verify_prepared_source()


def test_remapped_database_foreign_keys_and_task_actions_validate() -> None:
    raw_db = load_json(ROOT / "prepared" / "db.json")
    db = RetailDB.model_validate(raw_db)
    runtime_tasks = load_json(ROOT / "prepared" / "tasks.runtime.json")
    truth_by_id = {
        row["id"]: row for row in load_json(ROOT / "private" / "task-truth.json")
    }

    assert len(runtime_tasks) == constants.TASK_COUNT == 114
    assert set(truth_by_id) == {str(task["id"]) for task in runtime_tasks}

    variants = {
        item_id: product_id
        for product_id, product in db.products.items()
        for item_id in product.variants
    }
    for user_id, user in db.users.items():
        assert user.user_id == user_id
        for order_id in user.orders:
            assert order_id in db.orders
            assert db.orders[order_id].user_id == user_id

    for order_id, order in db.orders.items():
        assert order.order_id == order_id
        assert order.user_id in db.users
        user = db.users[order.user_id]
        assert order_id in user.orders
        for item in order.items:
            assert item.product_id in db.products
            assert variants[item.item_id] == item.product_id
        for payment in order.payment_history:
            assert payment.payment_method_id in user.payment_methods

    tools = RetailTools(db)
    known_tools = set(preview._mcp_manifest()["tool_names"])
    parsed_tasks = []
    for runtime_task in runtime_tasks:
        assert "evaluation_criteria" not in runtime_task
        authenticated_user_id = runtime_task["authenticated_user_id"]
        assert authenticated_user_id in db.users
        task = Task.model_validate(
            {
                **runtime_task,
                "evaluation_criteria": truth_by_id[str(runtime_task["id"])][
                    "evaluation_criteria"
                ],
            }
        )
        parsed_tasks.append(task)
        for action in task.evaluation_criteria.actions or []:
            assert action.name in known_tools
            # Some reference trajectories intentionally include calls that return an
            # error. Binding the reviewed arguments verifies their tool contract
            # without rewriting those purposeful failure cases into successes.
            inspect.signature(getattr(tools, action.name)).bind(**action.arguments)
            if action.compare_args is not None:
                assert set(action.compare_args) <= set(action.arguments)
    assert len(parsed_tasks) == 114


def test_remapped_source_is_public_safe_without_collapsing_identity_relations() -> None:
    raw_db = load_json(ROOT / "prepared" / "db.json")
    runtime_tasks = load_json(ROOT / "prepared" / "tasks.runtime.json")
    private_truth = load_json(ROOT / "private" / "task-truth.json")
    replacement_ledger = load_json(ROOT / "private" / "replacements.json")
    lock = load_json(ROOT / "source.lock.json")

    encoded_outputs = json.dumps(
        (raw_db, runtime_tasks, private_truth), ensure_ascii=False, sort_keys=True
    )
    emails = remap._EMAIL_PATTERN.findall(encoded_outputs)
    assert emails
    assert all(email.endswith(".invalid") for email in emails)
    assert "sofia.thomas3019@example.com" not in encoded_outputs
    assert "aarav.santos8321@example.com" not in encoded_outputs

    task_106 = next(task for task in runtime_tasks if str(task["id"]) == "106")
    known_info = task_106["user_scenario"]["instructions"]["known_info"]
    assert "supplemental-contact-" in known_info
    assert "@example.invalid" in known_info
    assert "sofia.thomas" not in known_info.lower()
    authenticated_user = raw_db["users"][task_106["authenticated_user_id"]]
    assert authenticated_user["email"] in known_info

    last_four_mapping = replacement_ledger["fields"]["last_four"]
    remapped_last_fours = [
        payment["last_four"]
        for user in raw_db["users"].values()
        for payment in user["payment_methods"].values()
        if "last_four" in payment
    ]
    assert len(remapped_last_fours) == 239
    assert len(set(remapped_last_fours)) == len(last_four_mapping) == 235
    assert set(remapped_last_fours) == set(last_four_mapping.values())
    assert set(last_four_mapping).isdisjoint(last_four_mapping.values())

    zip_mapping = replacement_ledger["fields"]["zip"]
    remapped_zips = {
        value
        for key, value in remap._walk_strings((raw_db, runtime_tasks, private_truth))
        if key == "zip"
    }
    assert remapped_zips <= set(zip_mapping.values())
    assert set(zip_mapping).isdisjoint(remapped_zips)
    assert len({user["address"]["city"] for user in raw_db["users"].values()}) > 10
    assert len({user["address"]["state"] for user in raw_db["users"].values()}) > 10

    assert set(lock["prepared_files"]) == set(remap._PREPARED_SOURCE_FILES)
    assert set(lock["private_files"]) == set(remap._PRIVATE_SOURCE_FILES)

    scenario_text = json.dumps(
        [task["user_scenario"] for task in runtime_tasks],
        ensure_ascii=False,
        sort_keys=True,
    )
    assert not remap._RUNTIME_INTERNAL_ID_PATTERN.search(scenario_text)
    assert "respond exactly" not in scenario_text.lower()
    assert "do not share this with the agent" not in scenario_text.lower()

    task_45 = next(task for task in runtime_tasks if str(task["id"]) == "45")
    task_45_user = raw_db["users"][task_45["authenticated_user_id"]]
    task_45_name = (
        f"{task_45_user['name']['first_name']} "
        f"{task_45_user['name']['last_name']}"
    )
    assert task_45_name in task_45["user_scenario"]["instructions"]["known_info"]
    assert task_45["authenticated_user_id"] not in json.dumps(
        task_45["user_scenario"], sort_keys=True
    )

    for task in runtime_tasks:
        user = raw_db["users"][task["authenticated_user_id"]]
        known_info = str(
            task["user_scenario"]["instructions"].get("known_info") or ""
        )
        full_name = f"{user['name']['first_name']} {user['name']['last_name']}"
        has_email = user["email"] in known_info
        has_name_and_zip = full_name in known_info and user["address"]["zip"] in known_info
        assert has_email or has_name_and_zip, task["id"]

    for node in remap._walk_dicts((raw_db, runtime_tasks, private_truth)):
        if not {"city", "state", "zip"}.issubset(node):
            continue
        location = (str(node["city"]), str(node["state"]))
        assert str(node["zip"]) == remap._CITY_ZIP_CODES[location]

    task_29 = next(task for task in runtime_tasks if str(task["id"]) == "29")
    task_29_reason = task_29["user_scenario"]["instructions"]["reason_for_call"]
    assert "same version listed in your pending order" in task_29_reason
    assert "item_" not in task_29_reason
    assert lock["validation"] == {
        "tasks_checked": 114,
        "reference_actions_checked": 550,
        "successful_results_checked": 532,
        "intentional_errors_checked": 18,
        "result_mismatches": 0,
        "error_class_mismatches": 0,
        "final_states_checked": 114,
        "final_state_mismatches": 0,
        "semantic_mismatches": 0,
        "outcome_digest": lock["validation"]["outcome_digest"],
    }


def test_invalid_identity_clues_stay_invalid_but_are_fully_remapped() -> None:
    db = load_json(ROOT / "prepared" / "db.json")
    runtime_by_id = {
        str(task["id"]): task
        for task in load_json(ROOT / "prepared" / "tasks.runtime.json")
    }
    truth_by_id = {
        str(task["id"]): task
        for task in load_json(ROOT / "private" / "task-truth.json")
    }

    task_39_actions = truth_by_id["39"]["evaluation_criteria"]["actions"]
    expected_user = db["users"][task_39_actions[2]["arguments"]["user_id"]]
    expected_name = expected_user["name"]
    assert task_39_actions[0]["arguments"]["first_name"] == expected_name["first_name"]
    assert task_39_actions[0]["arguments"]["last_name"] == expected_name["last_name"]
    assert task_39_actions[1]["arguments"]["first_name"] == expected_name["first_name"]
    assert task_39_actions[1]["arguments"]["last_name"] == expected_name["last_name"]
    assert task_39_actions[0]["arguments"]["zip"] != expected_user["address"]["zip"]
    assert task_39_actions[1]["arguments"]["zip"] == expected_user["address"]["zip"]

    task_46_actions = truth_by_id["46"]["evaluation_criteria"]["actions"]
    reason = runtime_by_id["46"]["user_scenario"]["instructions"]["reason_for_call"]
    assert task_46_actions[1]["arguments"]["order_id"].startswith("#")
    assert not task_46_actions[1]["arguments"]["order_id"].startswith("#R")
    assert task_46_actions[2]["arguments"]["order_id"] == "#0000940"
    assert task_46_actions[3]["arguments"]["order_id"] == "#R0000940"
    assert "left off the R prefix" in reason
    assert "respond exactly" not in reason.lower()
    assert "9502126" not in reason
    assert "9502127" not in reason

    task_2_actions = truth_by_id["2"]["evaluation_criteria"]["actions"]
    assert task_2_actions[1]["arguments"]["product_id"].startswith("missing_product_")

    replacement_ledger = load_json(ROOT / "private" / "replacements.json")
    source_full_names = {
        source
        for source, target in replacement_ledger["global"].items()
        if len(target.split()) == 2
        and target.split()[0] in remap._FIRST_NAMES
        and target.split()[1] in remap._LAST_NAMES
    }
    encoded = json.dumps((runtime_by_id, truth_by_id), sort_keys=True)
    assert not [name for name in source_full_names if name in encoded]

    source_name_pairs = {tuple(name.split(" ", 1)) for name in source_full_names}
    output_name_pairs = {
        (str(node["first_name"]), str(node["last_name"]))
        for value in (runtime_by_id, truth_by_id)
        for node in remap._walk_dicts(value)
        if {"first_name", "last_name"}.issubset(node)
    }
    assert source_name_pairs.isdisjoint(output_name_pairs)

    task_67_actions = truth_by_id["67"]["evaluation_criteria"]["actions"]
    expected_user = db["users"][task_67_actions[3]["arguments"]["user_id"]]
    expected_name = expected_user["name"]
    assert {
        (
            action["arguments"]["first_name"],
            action["arguments"]["last_name"],
        )
        for action in task_67_actions[:3]
    } == {(expected_name["first_name"], expected_name["last_name"])}
    assert task_67_actions[0]["arguments"]["zip"] != expected_user["address"]["zip"]
    assert task_67_actions[1]["arguments"]["zip"] != expected_user["address"]["zip"]
    assert task_67_actions[2]["arguments"]["zip"] == expected_user["address"]["zip"]


def test_privacy_scan_checks_every_email_not_only_the_first_match() -> None:
    with pytest.raises(ValueError, match="source-shaped identity"):
        remap._assert_no_source_identity(
            [
                {
                    "safe": "customer-0001@example.invalid",
                    "unsafe": "source.person@example.com",
                }
            ],
            {},
        )


def test_preview_is_deterministic_private_and_has_exact_coordinate_matrix() -> None:
    first = preview.build_preview()
    second = preview.build_preview()
    assert first == second
    assert first["preview_sha256"] == digest_value(
        {key: value for key, value in first.items() if key != "preview_sha256"}
    )

    full = first["cohorts"]["full"]["coordinates"]
    canary = first["cohorts"]["canary"]["coordinates"]
    selection_policy = first["cohorts"]["canary"]["selection_policy"]
    assert first["schema_version"] == 10
    assert first["study_id"] == "agent-lens-retail-support-study-v10"
    assert first["telemetry"]["mode"] == "live_application_instrumentation"
    assert first["telemetry"]["destinations"] == {
        "canary": "wandb/agent-lens-retail-support-live-canary-v10-r4",
        "full": "wandb/agent-lens-retail-support-live-v10",
    }
    assert first["telemetry"]["run_requires_live_agent_lens"] is True
    assert first["telemetry"]["support_agent_inference_spans"] == "live"
    assert first["telemetry"]["mcp_tool_spans"] == "live_executions_only"
    assert first["telemetry"]["simulator_spans"] == "excluded"
    assert first["telemetry"]["emit_from_frozen_bundles_only"] is False
    assert len(full) == 114 * 3 * 4 == constants.FULL_COORDINATE_COUNT
    assert len(canary) == 20 * 3 == constants.CANARY_COORDINATE_COUNT
    assert len({row["coordinate_id"] for row in full + canary}) == len(full) + len(
        canary
    )

    task_ids = [
        str(row["id"]) for row in load_json(ROOT / "prepared" / "tasks.runtime.json")
    ]
    assert full == [
        coordinate.model_dump(mode="json")
        for coordinate in preview.coordinate_matrix("full", task_ids)
    ]
    assert first["cohorts"]["full"]["trial_indices"] == [0, 1, 2, 3]
    assert first["cohorts"]["full"]["trial_coordinate_counts"] == [342] * 4
    assert first["cohorts"]["full"]["ordering"] == ("trial-major-task-profile")
    for trial_index in constants.TRIAL_INDICES:
        start = trial_index * 342
        block = full[start : start + 342]
        assert {row["trial_index"] for row in block} == {trial_index}
        assert [row["logical_task_id"] for row in block[::3]] == task_ids
        assert [row["profile"] for row in block[:3]] == list(constants.PROFILES)

    by_task: dict[str, set[tuple[str, int]]] = {}
    for row in full:
        by_task.setdefault(row["logical_task_id"], set()).add(
            (row["profile"], row["trial_index"])
        )
    assert len(by_task) == 114
    expected_replicates = {
        (profile, trial_index)
        for profile in constants.PROFILES
        for trial_index in constants.TRIAL_INDICES
    }
    assert set(map(frozenset, by_task.values())) == {frozenset(expected_replicates)}
    repeated = [
        row
        for row in full
        if row["logical_task_id"] == task_ids[0] and row["profile"] == "ordinary"
    ]
    assert [row["trial_index"] for row in repeated] == [0, 1, 2, 3]
    assert len({row["coordinate_id"] for row in repeated}) == 4
    assert len({row["seed"] for row in repeated}) == 4

    assert canary == [
        coordinate.model_dump(mode="json")
        for coordinate in preview.coordinate_matrix(
            "canary", list(constants.CANARY_TASK_IDS)
        )
    ]
    canary_by_task: dict[str, set[str]] = {}
    for row in canary:
        canary_by_task.setdefault(row["logical_task_id"], set()).add(row["profile"])
    assert set(canary_by_task) == set(constants.CANARY_TASK_IDS)
    assert set(map(frozenset, canary_by_task.values())) == {
        frozenset(constants.PROFILES)
    }
    assert first["cohorts"]["canary"]["trial_indices"] == [0, 1, 2, 3]
    assert first["cohorts"]["canary"]["trial_coordinate_counts"] == [15] * 4
    assert first["cohorts"]["canary"]["trial_assignment_policy"] == (
        "round-robin-task-profile-v1"
    )
    for trial_index in constants.TRIAL_INDICES:
        rows = [row for row in canary if row["trial_index"] == trial_index]
        assert len(rows) == 15
        assert {
            profile: sum(row["profile"] == profile for row in rows)
            for profile in constants.PROFILES
        } == {profile: 5 for profile in constants.PROFILES}
    assert selection_policy == {
        "id": "fixed-agent-stratified-v2",
        "task_group_counts": {
            "order_management": 7,
            "returns_resolution": 7,
            "generalist": 6,
        },
        "source_split_counts": {"train": 10, "test": 10},
        "reference_action_count": 105,
        "reference_tool_count": 15,
        "manifest_tool_count": 16,
        "manifest_tools_without_reference_actions": 1,
        "covers_all_reference_tools": True,
        "required_strata": [
            "human_handoff",
            "lookup_read_only",
            "mutation_confirmation",
            "no_action_communication",
            "simple_mutation_baseline",
        ],
    }
    assert first["models"]["support_agent"]["max_output_tokens"] == 4_096
    assert first["models"]["support_agent"]["selection"] == (
        "coordinate_model_route_id"
    )
    assert first["models"]["user_simulator"]["max_output_tokens"] == 512
    assert first["models"]["user_simulator"]["enable_thinking"] is False
    assert first["models"]["user_simulator"]["scenario_conformance"][
        "contract_version"
    ] == "confirmation-choreography-v8"
    assert first["run_policy"]["transport_retries"] == 2
    assert first["run_policy"]["transport_retry_delays_seconds"] == {
        "http_500_internal_server_error": [30.0, 60.0],
        "other_retryable_transport": [1.0, 2.0],
    }
    assert first["run_policy"]["full_run_approval_mode"] == (
        "direct_operator_exact_preview_sha"
    )
    assert "canary_execution" not in first
    assert "canary_execution_lock_sha256" not in first["inputs"]

    legacy_preview = load_json(ROOT / "preview.json")
    assert legacy_preview["study_id"] == "agent-lens-retail-support-study-v1"
    assert len(legacy_preview["cohorts"]["full"]["coordinates"]) == 342
    assert constants.PREVIEW_PATH == ROOT / "preview-live-v10.json"
    v9_preview = load_json(ROOT / "preview-live-v1.json")
    assert v9_preview["schema_version"] == 9
    assert v9_preview["preview_sha256"] == (
        "d1020c2cb628d0f7ee786871b339ab7c174c4d9cf6a7a7898f92eb30e68c1718"
    )
    assert v9_preview["telemetry"]["destinations"]["full"] == (
        "wandb/agent-lens-retail-support-live-v9"
    )

    raw_db = load_json(ROOT / "prepared" / "db.json")
    first_user = next(iter(raw_db["users"].values()))
    first_order_id = next(iter(raw_db["orders"]))
    task_text = load_json(ROOT / "prepared" / "tasks.runtime.json")[0]["user_scenario"][
        "instructions"
    ]["reason_for_call"]
    encoded = json.dumps(first, sort_keys=True)
    assert first_user["email"] not in encoded
    assert first_order_id not in encoded
    assert task_text not in encoded
    assert "evaluation_criteria" not in encoded


def test_canary_selection_covers_reference_tools_and_required_strata() -> None:
    runtime_by_id = {
        str(row["id"]): row
        for row in load_json(ROOT / "prepared" / "tasks.runtime.json")
    }
    truth = load_json(ROOT / "private" / "task-truth.json")
    truth_by_id = {str(row["id"]): row for row in truth}
    selected_truth = [truth_by_id[task_id] for task_id in constants.CANARY_TASK_IDS]
    selected_actions = [
        action
        for row in selected_truth
        for action in row["evaluation_criteria"]["actions"]
    ]
    selected_tools = {str(action["name"]) for action in selected_actions}
    all_reference_tools = {
        str(action["name"])
        for row in truth
        for action in row["evaluation_criteria"]["actions"]
    }
    manifest_tools = set(preview._mcp_manifest()["tool_names"])

    assert constants.CANARY_TASK_IDS == (
        "17",
        "40",
        "41",
        "87",
        "24",
        "76",
        "113",
        "18",
        "21",
        "45",
        "51",
        "53",
        "73",
        "84",
        "12",
        "16",
        "59",
        "39",
        "55",
        "68",
    )
    assert len(set(constants.CANARY_TASK_IDS)) == constants.CANARY_TASK_COUNT
    assert selected_tools == all_reference_tools
    assert constants.WRITE_TOOLS <= selected_tools
    assert "transfer_to_human_agents" in selected_tools
    assert manifest_tools - all_reference_tools == {"list_all_product_types"}

    no_action_communication = truth_by_id["24"]["evaluation_criteria"]
    assert no_action_communication["actions"] == []
    assert no_action_communication["communicate_info"]

    read_only = truth_by_id["68"]["evaluation_criteria"]
    read_only_tools = {str(action["name"]) for action in read_only["actions"]}
    assert read_only_tools
    assert read_only_tools.isdisjoint(constants.WRITE_TOOLS)
    assert read_only["communicate_info"]

    confirmation = truth_by_id["84"]["evaluation_criteria"]
    assert [action["name"] for action in confirmation["actions"]] == [
        "return_delivered_order_items"
    ]
    confirmation_prompt = runtime_by_id["84"]["user_scenario"]["instructions"][
        "reason_for_call"
    ].lower()
    assert "asks for confirmation" in confirmation_prompt

    handoff = truth_by_id["50"]["evaluation_criteria"]
    assert [action["name"] for action in handoff["actions"]] == [
        "transfer_to_human_agents"
    ]

    simple_return = truth_by_id["73"]["evaluation_criteria"]
    assert [action["name"] for action in simple_return["actions"]] == [
        "return_delivered_order_items"
    ]
    simple_cancellation = truth_by_id["113"]["evaluation_criteria"]
    assert {action["name"] for action in simple_cancellation["actions"]} == {
        "cancel_pending_order"
    }

    splits = load_json(ROOT / "prepared" / "task_splits.json")
    assert sum(task_id in splits["train"] for task_id in constants.CANARY_TASK_IDS) == 10
    assert sum(task_id in splits["test"] for task_id in constants.CANARY_TASK_IDS) == 10


def test_static_verification_rejects_trial_order_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = preview.build_preview()
    tampered = json.loads(json.dumps(planned))
    coordinates = tampered["cohorts"]["full"]["coordinates"]
    coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
    runtime = SimpleNamespace(
        preview=tampered,
        tasks=tuple(load_json(ROOT / "prepared" / "tasks.runtime.json")),
    )
    monkeypatch.setattr(verify, "verify_prepared_source", lambda: {})
    monkeypatch.setattr(verify, "verify_preview", lambda: tampered)
    monkeypatch.setattr(verify, "verify_runtime_snapshot", lambda _sha: runtime)

    with pytest.raises(ValueError, match="exact trial matrix"):
        verify.verify_static()


def test_preview_rejects_duplicate_canary_task_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preview, "CANARY_TASK_IDS", ("16",) * 20)
    with pytest.raises(ValueError, match="canary task IDs are not unique"):
        preview.build_preview()


def test_cost_budget_rejects_before_crossing_cap() -> None:
    budget = CostBudget(0.001, observed_usd=0.0009)
    with pytest.raises(RuntimeError, match="cost cap would be exceeded"):
        budget.require_capacity(
            constants.SUPPORT_MODEL,
            messages=[{"role": "user", "content": "hello"}],
            max_output_tokens=2_048,
        )
    assert budget.observed_usd == 0.0009


def test_cost_reservation_includes_the_support_tool_manifest() -> None:
    messages = [{"role": "user", "content": "hello"}]
    without_tools = conservative_request_cost_usd(
        constants.SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=32,
    )
    with_tools = conservative_request_cost_usd(
        constants.SUPPORT_MODEL,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "x" * 20_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        max_output_tokens=32,
    )
    assert with_tools > without_tools


def test_canary_review_must_cover_current_exact_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_sha = "a" * 64
    coordinate_ids = [f"coordinate-{index:02d}" for index in range(36)]
    cohort_dir = tmp_path / "runs" / "canary" / preview_sha
    cohort_dir.mkdir(parents=True)
    ledger = RunLedger(
        study_id=constants.STUDY_ID,
        cohort="canary",
        preview_sha256=preview_sha,
        planned_coordinate_ids=coordinate_ids,
        completed_coordinate_ids=coordinate_ids,
        unavailable={},
        observed_cost_usd=0.5,
        bundle_digests={coordinate_id: "c" * 64 for coordinate_id in coordinate_ids},
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    (cohort_dir / "ledger.json").write_text(
        json.dumps(ledger.model_dump(mode="json")), encoding="utf-8"
    )
    private_dir = tmp_path / "private"
    private_dir.mkdir()

    # A one-row review used to pass because only the booleans were checked. It
    # must never authorize a paid 342-cell cohort.
    value = {
        "schema_version": 1,
        "preview_sha256": preview_sha,
        "canary_ledger_sha256": "stale-ledger",
        "review_status": "approved",
        "rows": [_all_true_review_row(coordinate_ids[0])],
        "review_sha256": "",
    }
    value["review_sha256"] = digest_value(value)
    (private_dir / f"canary-review-{preview_sha}.json").write_text(
        json.dumps(value), encoding="utf-8"
    )

    monkeypatch.setattr(review, "PRIVATE_DIR", private_dir)
    monkeypatch.setattr(review, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        review,
        "verify_cohort",
        lambda cohort: {
            "status": "complete",
            "ledger_sha256": ledger.ledger_sha256,
        },
    )
    monkeypatch.setattr(
        review,
        "verify_preview",
        lambda: {
            "preview_sha256": preview_sha,
            "cohorts": {
                "canary": {
                    "coordinates": [
                        {"coordinate_id": coordinate_id}
                        for coordinate_id in coordinate_ids
                    ]
                }
            },
        },
    )

    with pytest.raises(ValueError, match="canary (review|ledger)"):
        review.verify_canary_review()


def test_canary_review_requires_full_bundle_and_attempt_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_sha = "b" * 64
    coordinate_id = "coordinate-00"
    cohort_dir = tmp_path / "runs" / "canary" / preview_sha
    cohort_dir.mkdir(parents=True)
    ledger = RunLedger(
        study_id=constants.STUDY_ID,
        cohort="canary",
        preview_sha256=preview_sha,
        planned_coordinate_ids=[coordinate_id],
        completed_coordinate_ids=[coordinate_id],
        unavailable={},
        observed_cost_usd=0.1,
        bundle_digests={coordinate_id: "c" * 64},
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    (cohort_dir / "ledger.json").write_text(
        json.dumps(ledger.model_dump(mode="json")), encoding="utf-8"
    )

    monkeypatch.setattr(review, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        review,
        "verify_preview",
        lambda: {
            "preview_sha256": preview_sha,
            "cohorts": {"canary": {"coordinates": [{"coordinate_id": coordinate_id}]}},
        },
    )
    monkeypatch.setattr(
        review,
        "verify_cohort",
        lambda cohort: {
            "status": "paused",
            "ledger_sha256": ledger.ledger_sha256,
        },
    )

    with pytest.raises(ValueError, match="verify as complete"):
        review.verify_canary_review()


def test_exact_preview_digest_authorizes_full_cohort_at_execution_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_sha = "d" * 64
    monkeypatch.setattr(
        runner,
        "verify_runtime_snapshot",
        lambda _sha: SimpleNamespace(
            preview={
                "preview_sha256": preview_sha,
                "cohorts": {"full": {"coordinates": []}},
                "cost_caps_usd": {"full": 50.0},
            },
            tasks=[],
            pricing_lock={},
            manifest={},
        ),
    )
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")

    ledger = runner.run_cohort("full", approve_preview_sha=preview_sha)
    assert ledger.status == "complete"
    assert ledger.planned_coordinate_ids == []

    with pytest.raises(ValueError, match="approval digest does not match"):
        runner.run_cohort("full", approve_preview_sha="e" * 64)
