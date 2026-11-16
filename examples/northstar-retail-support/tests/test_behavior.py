from __future__ import annotations

from retail_support_study_v1.behavior import (
    RuntimeBehaviorProfile,
    select_tool_behavior,
    tool_family,
)
from retail_support_study_v1.traffic_config import build_behavior_profile


def _profile(
    *, service_rate: float = 0.001, timeout_rate: float = 0.0005
) -> RuntimeBehaviorProfile:
    return RuntimeBehaviorProfile.from_mapping(
        {
            "profile_id": "retail-production-shaped-v1",
            "seed": "retail-production-shaped-v1-seed",
            "latency_by_family": {
                "calculation": {"p50_ms": 20, "p95_ms": 45, "p99_ms": 100},
                "identity_lookup": {
                    "p50_ms": 110,
                    "p95_ms": 300,
                    "p99_ms": 650,
                },
                "entity_read": {
                    "p50_ms": 140,
                    "p95_ms": 380,
                    "p99_ms": 900,
                },
                "catalog_list": {
                    "p50_ms": 240,
                    "p95_ms": 700,
                    "p99_ms": 1500,
                },
                "mutation": {
                    "p50_ms": 350,
                    "p95_ms": 950,
                    "p99_ms": 1800,
                },
                "human_transfer": {
                    "p50_ms": 200,
                    "p95_ms": 600,
                    "p99_ms": 1200,
                },
            },
            "service_unavailable_rate": service_rate,
            "upstream_timeout_rate": timeout_rate,
            "service_unavailable_delay_ms": [180, 450],
            "upstream_timeout_delay_ms": [1500, 2500],
        }
    )


def test_tool_families_cover_the_retail_manifest() -> None:
    assert tool_family("calculate") == "calculation"
    assert tool_family("find_user_id_by_email") == "identity_lookup"
    assert tool_family("get_order_details") == "entity_read"
    assert tool_family("list_all_product_types") == "catalog_list"
    assert tool_family("cancel_pending_order") == "mutation"
    assert tool_family("transfer_to_human_agents") == "human_transfer"


def test_latency_selection_is_deterministic_and_bounded() -> None:
    profile = _profile(service_rate=0, timeout_rate=0)
    first = select_tool_behavior(
        profile,
        conversation_id="conversation-1",
        tool_name="get_order_details",
        call_ordinal=1,
        fault_already_injected=False,
    )
    second = select_tool_behavior(
        profile,
        conversation_id="conversation-1",
        tool_name="get_order_details",
        call_ordinal=1,
        fault_already_injected=False,
    )
    assert first == second
    assert first.fault_code is None
    assert 0.070 <= first.delay_seconds <= 1.350


def test_forced_fault_occurs_only_once_and_never_on_a_write() -> None:
    profile = _profile(service_rate=0.999, timeout_rate=0)
    read = select_tool_behavior(
        profile,
        conversation_id="conversation-2",
        tool_name="get_user_details",
        call_ordinal=1,
        fault_already_injected=False,
    )
    assert read.fault_code == "service_unavailable"
    assert 0.180 <= read.delay_seconds <= 0.450

    recovered = select_tool_behavior(
        profile,
        conversation_id="conversation-2",
        tool_name="get_user_details",
        call_ordinal=2,
        fault_already_injected=True,
    )
    assert recovered.fault_code is None

    write = select_tool_behavior(
        profile,
        conversation_id="conversation-3",
        tool_name="cancel_pending_order",
        call_ordinal=1,
        fault_already_injected=False,
    )
    assert write.fault_code is None


def test_sealed_preview_behavior_profile_normalizes_to_runtime_shape() -> None:
    sealed = build_behavior_profile()
    runtime = RuntimeBehaviorProfile.from_mapping(
        sealed.model_dump(mode="json")
    )

    assert runtime.profile_id == sealed.profile_id
    assert runtime.seed == sealed.seed
    assert runtime.latency_by_family["entity_read"].p50_ms == 140.0
    assert runtime.service_unavailable_rate == 0.001
    assert runtime.upstream_timeout_rate == 0.0005
    assert runtime.service_unavailable_delay_ms == (180.0, 450.0)
    assert runtime.upstream_timeout_delay_ms == (1500.0, 2500.0)
    assert "get_order_details" in runtime.fault_eligible_tool_names
    assert "calculate" not in runtime.fault_eligible_tool_names
