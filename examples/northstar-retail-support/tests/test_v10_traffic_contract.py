from __future__ import annotations

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from retail_support_study_v1 import constants, preview
from retail_support_study_v1.canonical import load_json
from retail_support_study_v1.schemas import (
    AgentProfile,
    BehaviorProfile,
    Coordinate,
    ModelRoute,
)
from retail_support_study_v1.traffic_config import (
    GENERALIST_AGENT_ID,
    ORDER_AGENT_ID,
    RETURNS_AGENT_ID,
    allocate_model_route_ids,
    build_agent_profiles,
    build_behavior_profile,
    build_model_routes,
    route_agent_profile_id,
)


def test_v10_preview_binds_exact_agents_routes_behavior_and_costs() -> None:
    planned = preview.build_preview()
    traffic = planned["traffic"]

    agents = [AgentProfile.model_validate(row) for row in traffic["agent_profiles"]]
    routes = [ModelRoute.model_validate(row) for row in traffic["model_routes"]]
    behavior = BehaviorProfile.model_validate(traffic["behavior_profile"])

    assert [profile.profile_id for profile in agents] == [
        GENERALIST_AGENT_ID,
        ORDER_AGENT_ID,
        RETURNS_AGENT_ID,
    ]
    assert {route.provider_name for route in routes} == {"wandb", "anthropic"}
    assert len({route.model for route in routes}) == 5
    sonnet = next(route for route in routes if route.model == "claude-sonnet-5")
    assert sonnet.temperature is None
    assert sonnet.enable_thinking is False
    assert all(
        route.enable_thinking is False
        for route in routes
        if route.provider_name == "wandb"
    )
    assert planned["telemetry"]["destinations"] == {
        "canary": "wandb/agent-lens-retail-support-live-canary-v10-r4",
        "full": "wandb/agent-lens-retail-support-live-v10",
    }
    assert planned["cost_caps_usd"] == {"canary": 25.0, "full": 200.0}
    assert planned["run_policy"]["full_cost_projection_margin"] == 0.25
    assert behavior.fault_policy["maximum_injected_faults_per_conversation"] == 1
    assert sum(
        rule["eligible_call_rate"] for rule in behavior.fault_policy["rules"]
    ) == pytest.approx(0.0015)

    expected_counts = {
        "canary": dict(constants.MODEL_ROUTE_COUNTS_CANARY),
        "full": dict(constants.MODEL_ROUTE_COUNTS_FULL),
    }
    for cohort in ("canary", "full"):
        coordinates = planned["cohorts"][cohort]["coordinates"]
        observed = Counter(row["model_route_id"] for row in coordinates)
        assert dict(observed) == expected_counts[cohort]
        assert planned["cohorts"][cohort]["model_route_counts"] == expected_counts[
            cohort
        ]


def test_route_allocation_is_exact_deterministic_and_well_mixed() -> None:
    full_keys = [f"coordinate-{index}" for index in range(1368)]
    first = allocate_model_route_ids(cohort="full", coordinate_keys=full_keys)
    second = allocate_model_route_ids(cohort="full", coordinate_keys=full_keys)
    assert first == second
    assert Counter(first) == Counter(dict(constants.MODEL_ROUTE_COUNTS_FULL))
    assert len(set(first[:100])) >= 4

    canary_keys = [f"canary-{index}" for index in range(60)]
    canary = allocate_model_route_ids(cohort="canary", coordinate_keys=canary_keys)
    assert Counter(canary) == Counter(dict(constants.MODEL_ROUTE_COUNTS_CANARY))


def test_content_router_uses_visible_request_and_mixed_intent_falls_back() -> None:
    assert route_agent_profile_id(
        "Please cancel the pending order before it ships."
    ) == ORDER_AGENT_ID
    assert route_agent_profile_id(
        "The headphones arrived broken and I need to return them."
    ) == RETURNS_AGENT_ID
    assert route_agent_profile_id(
        "Please update the shipping address on my pending order."
    ) == ORDER_AGENT_ID
    assert route_agent_profile_id(
        "Please update the delivery addresses on my pending orders."
    ) == ORDER_AGENT_ID
    assert route_agent_profile_id(
        "Cancel the charger order and return the tablet that arrived yesterday."
    ) == GENERALIST_AGENT_ID


def test_fixed_canary_task_briefs_exercise_all_three_content_routes() -> None:
    tasks = {
        str(row["id"]): row
        for row in load_json(constants.PREPARED_DIR / "tasks.runtime.json")
    }
    expected = {
        "order_management": ORDER_AGENT_ID,
        "returns_resolution": RETURNS_AGENT_ID,
        "generalist": GENERALIST_AGENT_ID,
    }
    for group, task_ids in constants.CANARY_TASK_GROUPS.items():
        expected_count = {
            "order_management": 7,
            "returns_resolution": 7,
            "generalist": 6,
        }[group]
        assert len(task_ids) == expected_count
        for task_id in task_ids:
            text = tasks[task_id]["user_scenario"]["instructions"]["reason_for_call"]
            assert route_agent_profile_id(text) == expected[group]
    assert route_agent_profile_id(
        "Can you tell me the current balance on my gift card?"
    ) == GENERALIST_AGENT_ID


def test_specialist_profiles_are_scoped_subsets_of_generalist() -> None:
    agents = build_agent_profiles(
        instructions_sha256="a" * 64,
        manifest_tool_names=(
            "calculate",
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "get_item_details",
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
            "transfer_to_human_agents",
        ),
    )
    by_id = {profile.profile_id: profile for profile in agents}
    generalist = set(by_id[GENERALIST_AGENT_ID].tool_names)
    order = set(by_id[ORDER_AGENT_ID].tool_names)
    returns = set(by_id[RETURNS_AGENT_ID].tool_names)
    assert order < generalist
    assert returns < generalist
    assert "return_delivered_order_items" not in order
    assert "cancel_pending_order" not in returns


@pytest.mark.parametrize(
    ("builder", "model", "field", "replacement"),
    [
        (build_model_routes, ModelRoute, "model", "changed/model"),
        (
            lambda: build_agent_profiles(
                instructions_sha256="b" * 64,
                manifest_tool_names=(
                    "calculate",
                    "cancel_pending_order",
                    "exchange_delivered_order_items",
                    "find_user_id_by_email",
                    "find_user_id_by_name_zip",
                    "get_item_details",
                    "get_order_details",
                    "get_product_details",
                    "get_user_details",
                    "list_all_product_types",
                    "modify_pending_order_address",
                    "modify_pending_order_items",
                    "modify_pending_order_payment",
                    "modify_user_address",
                    "return_delivered_order_items",
                    "transfer_to_human_agents",
                ),
            ),
            AgentProfile,
            "purpose",
            "changed purpose",
        ),
    ],
)
def test_digest_bound_records_reject_drift(
    builder: object,
    model: type[AgentProfile] | type[ModelRoute],
    field: str,
    replacement: str,
) -> None:
    records = builder()  # type: ignore[operator]
    raw = records[0].model_dump(mode="json")
    raw[field] = replacement
    with pytest.raises(ValidationError, match="digest mismatch"):
        model.model_validate(raw)

    behavior = build_behavior_profile().model_dump(mode="json")
    behavior["fault_policy"]["maximum_injected_faults_per_conversation"] = 2
    with pytest.raises(ValidationError, match="digest mismatch"):
        BehaviorProfile.model_validate(behavior)


def test_coordinate_route_binding_is_new_without_rewriting_legacy_shape() -> None:
    legacy = Coordinate(
        coordinate_id="legacy",
        logical_task_id="1",
        profile="ordinary",
        seed=1,
        cohort="full",
    )
    assert "model_route_id" not in legacy.model_dump(mode="json")
    assert "trial_index" not in legacy.model_dump(mode="json")

    current = Coordinate(
        coordinate_id="current",
        logical_task_id="1",
        profile="ordinary",
        trial_index=0,
        model_route_id="wandb-glm-5-3-flash",
        seed=1,
        cohort="full",
    )
    encoded = json.loads(current.model_dump_json())
    assert encoded["model_route_id"] == "wandb-glm-5-3-flash"
    assert encoded["trial_index"] == 0
