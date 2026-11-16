from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

from .canonical import digest_value
from .constants import (
    FULL_COORDINATE_COUNT,
    MODEL_ROUTE_COUNTS_CANARY,
    MODEL_ROUTE_COUNTS_FULL,
    SUPPORT_MAX_OUTPUT_TOKENS,
    SUPPORT_TEMPERATURE,
    WANDB_INFERENCE_BASE_URL,
)
from .schemas import AgentProfile, BehaviorProfile, ModelRoute

GENERALIST_AGENT_ID = "retail-support-generalist"
ORDER_AGENT_ID = "retail-order-management"
RETURNS_AGENT_ID = "retail-returns-resolution"

COMMON_TOOL_NAMES = (
    "calculate",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_item_details",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "transfer_to_human_agents",
)
ORDER_WRITE_TOOL_NAMES = (
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
)
RETURNS_WRITE_TOOL_NAMES = (
    "exchange_delivered_order_items",
    "return_delivered_order_items",
)
ALL_TOOL_NAMES = tuple(
    sorted(COMMON_TOOL_NAMES + ORDER_WRITE_TOOL_NAMES + RETURNS_WRITE_TOOL_NAMES)
)

# Patterns intentionally inspect only the customer's first visible message.
# They do not have access to a benchmark task ID, expected action, or label.
ORDER_ROUTER_PATTERNS = (
    r"\bcancel(?:led|ling|ation)?\b",
    r"\bcancel(?:led|ling|ation)?\b.{0,56}\borders?\b",
    r"\borders?\b.{0,56}\bcancel(?:led|ing|ation)?\b",
    r"\b(?:change|correct|modify|update)\b.{0,72}\b(?:delivery|shipping|orders?) address(?:es)?\b",
    r"\b(?:change|modify)\b.{0,56}\bdelivered to\b",
    r"\b(?:change|modify|switch|split)\b.{0,72}\bpayment(?: method)?\b",
    r"\b(?:change|modify|remove|replace|add)\b.{0,72}\bpending orders?\b",
)
RETURNS_ROUTER_PATTERNS = (
    r"\breturn(?:ed|ing|s)?\b",
    r"\brefund(?:ed|ing|s)?\b",
    r"\bexchange(?:d|s|ing)?\b",
    r"\breplace(?:d|ment|s|ing)?\b",
    r"\b(?:damaged|broken|wrong item)\b",
)


def _router_policy() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "policy_id": "first-customer-message-keyword-router-v1",
        "input": "first_visible_customer_message_text",
        "normalization": "unicode_nfkc_casefold_whitespace_collapse",
        "order_patterns": list(ORDER_ROUTER_PATTERNS),
        "returns_patterns": list(RETURNS_ROUTER_PATTERNS),
        "resolution": {
            "order_only": ORDER_AGENT_ID,
            "returns_only": RETURNS_AGENT_ID,
            "both": GENERALIST_AGENT_ID,
            "neither": GENERALIST_AGENT_ID,
        },
        "forbidden_inputs": [
            "benchmark task ID",
            "expected actions",
            "private task truth",
            "review labels",
        ],
        "policy_sha256": "",
    }
    return {
        **unsigned,
        "policy_sha256": digest_value(unsigned),
    }


AGENT_ROUTER_POLICY = _router_policy()


def normalize_customer_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def route_agent_profile_id(first_customer_text: str) -> str:
    """Choose an agent from visible customer content, never task truth."""

    normalized = normalize_customer_text(first_customer_text)
    order = any(re.search(pattern, normalized) for pattern in ORDER_ROUTER_PATTERNS)
    returns = any(re.search(pattern, normalized) for pattern in RETURNS_ROUTER_PATTERNS)
    if order == returns:
        return GENERALIST_AGENT_ID
    return ORDER_AGENT_ID if order else RETURNS_AGENT_ID


def build_agent_profiles(
    *,
    instructions_sha256: str,
    manifest_tool_names: Iterable[str],
) -> tuple[AgentProfile, ...]:
    manifest = tuple(sorted(manifest_tool_names))
    if manifest != ALL_TOOL_NAMES:
        raise ValueError("agent profiles do not match the exact MCP manifest")
    return (
        AgentProfile.seal(
            profile_id=GENERALIST_AGENT_ID,
            agent_name=GENERALIST_AGENT_ID,
            role="generalist",
            purpose=(
                "Handle mixed, ambiguous, account-level, catalog, and unmatched "
                "retail-support requests."
            ),
            tool_names=ALL_TOOL_NAMES,
            instructions_sha256=instructions_sha256,
        ),
        AgentProfile.seal(
            profile_id=ORDER_AGENT_ID,
            agent_name=ORDER_AGENT_ID,
            role="order_management",
            purpose=(
                "Handle pending-order cancellations, delivery and address changes, "
                "payment changes, and pending-item edits."
            ),
            tool_names=tuple(sorted(COMMON_TOOL_NAMES + ORDER_WRITE_TOOL_NAMES)),
            instructions_sha256=instructions_sha256,
        ),
        AgentProfile.seal(
            profile_id=RETURNS_AGENT_ID,
            agent_name=RETURNS_AGENT_ID,
            role="returns_resolution",
            purpose=(
                "Handle delivered-item returns, refunds, exchanges, replacements, "
                "damage, and wrong-item requests."
            ),
            tool_names=tuple(sorted(COMMON_TOOL_NAMES + RETURNS_WRITE_TOOL_NAMES)),
            instructions_sha256=instructions_sha256,
        ),
    )


_MODEL_ROUTE_SPECS = {
    "wandb-glm-5-3-flash": {
        "provider_name": "wandb",
        "provider_api": "openai_compatible",
        "model": "zai-org/GLM-5.3-Flash",
        "base_url": WANDB_INFERENCE_BASE_URL,
        "server_address": "api.inference.wandb.ai",
        "credential_env_var": "WANDB_API_KEY",
        "enable_thinking": False,
    },
    "wandb-deepseek-v4-1-flash": {
        "provider_name": "wandb",
        "provider_api": "openai_compatible",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "base_url": WANDB_INFERENCE_BASE_URL,
        "server_address": "api.inference.wandb.ai",
        "credential_env_var": "WANDB_API_KEY",
        "enable_thinking": False,
    },
    "wandb-kimi-k2-6": {
        "provider_name": "wandb",
        "provider_api": "openai_compatible",
        "model": "moonshotai/Kimi-K2.6",
        "base_url": WANDB_INFERENCE_BASE_URL,
        "server_address": "api.inference.wandb.ai",
        "credential_env_var": "WANDB_API_KEY",
        "enable_thinking": False,
    },
    "anthropic-claude-haiku-4-5": {
        "provider_name": "anthropic",
        "provider_api": "anthropic_messages",
        "model": "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com",
        "server_address": "api.anthropic.com",
        "credential_env_var": "ANTHROPIC_API_KEY",
    },
    "anthropic-claude-sonnet-5": {
        "provider_name": "anthropic",
        "provider_api": "anthropic_messages",
        "model": "claude-sonnet-5",
        "base_url": "https://api.anthropic.com",
        "server_address": "api.anthropic.com",
        "credential_env_var": "ANTHROPIC_API_KEY",
        # Sonnet 5 rejects an explicit temperature as deprecated. Omitting the
        # parameter is part of this route's sealed request identity. Adaptive
        # thinking is on by default for Sonnet 5, so disable it explicitly:
        # private reasoning must not enter frozen bundles or Agent Lens spans.
        "temperature": None,
        "enable_thinking": False,
    },
}


def build_model_routes() -> tuple[ModelRoute, ...]:
    full_counts = dict(MODEL_ROUTE_COUNTS_FULL)
    canary_counts = dict(MODEL_ROUTE_COUNTS_CANARY)
    if set(full_counts) != set(_MODEL_ROUTE_SPECS) or set(canary_counts) != set(
        _MODEL_ROUTE_SPECS
    ):
        raise ValueError("model route counts and route specs do not reconcile")
    routes: list[ModelRoute] = []
    for route_id, _ in MODEL_ROUTE_COUNTS_FULL:
        spec = dict(_MODEL_ROUTE_SPECS[route_id])
        temperature = spec.pop("temperature", SUPPORT_TEMPERATURE)
        enable_thinking = spec.pop("enable_thinking", None)
        routes.append(
            ModelRoute.seal(
                route_id=route_id,
                **spec,
                temperature=temperature,
                enable_thinking=enable_thinking,
                max_output_tokens=SUPPORT_MAX_OUTPUT_TOKENS,
                full_coordinate_count=full_counts[route_id],
                canary_coordinate_count=canary_counts[route_id],
            )
        )
    return tuple(routes)


MODEL_PROBE_POLICY = {
    "schema_version": 1,
    "must_precede_preview": True,
    "required_capability": "structured_tool_calling",
    "wandb_fallback_models": [
        "openai/gpt-oss-120b",
        "Qwen/Qwen3.6-35B-A3B",
    ],
    "anthropic_failure": "block_preview",
    "substitution_after_preview": "forbidden",
}
MODEL_PROBE_POLICY = {
    **MODEL_PROBE_POLICY,
    "policy_sha256": digest_value(
        {**MODEL_PROBE_POLICY, "policy_sha256": ""}
    ),
}


def allocate_model_route_ids(
    *,
    cohort: str,
    coordinate_keys: Iterable[str],
) -> list[str]:
    """Return exact, well-mixed route allocation in input coordinate order."""

    keys = list(coordinate_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("model-route allocation keys must be unique")
    if cohort == "full":
        expected_count = FULL_COORDINATE_COUNT
        route_counts = MODEL_ROUTE_COUNTS_FULL
    elif cohort == "canary":
        expected_count = sum(count for _, count in MODEL_ROUTE_COUNTS_CANARY)
        route_counts = MODEL_ROUTE_COUNTS_CANARY
    else:
        raise ValueError("cohort must be canary or full")
    if len(keys) != expected_count or sum(count for _, count in route_counts) != len(
        keys
    ):
        raise ValueError("model-route allocation count mismatch")

    ranked_positions = sorted(
        range(len(keys)),
        key=lambda index: hashlib.sha256(
            f"model-route-v1:{cohort}:{keys[index]}".encode("utf-8")
        ).digest(),
    )
    allocated = [""] * len(keys)
    cursor = 0
    for route_id, count in route_counts:
        for position in ranked_positions[cursor : cursor + count]:
            allocated[position] = route_id
        cursor += count
    if cursor != len(keys) or any(not route_id for route_id in allocated):
        raise ValueError("model-route allocation is incomplete")
    return allocated


def build_behavior_profile() -> BehaviorProfile:
    latency = {
        "calculation": {"p50": 20.0, "p95": 45.0, "p99": 100.0},
        "identity_lookup": {"p50": 110.0, "p95": 300.0, "p99": 650.0},
        "entity_read": {"p50": 140.0, "p95": 380.0, "p99": 900.0},
        "catalog_list": {"p50": 240.0, "p95": 700.0, "p99": 1500.0},
        "mutation": {"p50": 350.0, "p95": 950.0, "p99": 1800.0},
        "human_transfer": {"p50": 200.0, "p95": 600.0, "p99": 1200.0},
    }
    tool_families = {
        "calculate": "calculation",
        "find_user_id_by_email": "identity_lookup",
        "find_user_id_by_name_zip": "identity_lookup",
        "get_item_details": "entity_read",
        "get_order_details": "entity_read",
        "get_product_details": "entity_read",
        "get_user_details": "entity_read",
        "list_all_product_types": "catalog_list",
        "cancel_pending_order": "mutation",
        "exchange_delivered_order_items": "mutation",
        "modify_pending_order_address": "mutation",
        "modify_pending_order_items": "mutation",
        "modify_pending_order_payment": "mutation",
        "modify_user_address": "mutation",
        "return_delivered_order_items": "mutation",
        "transfer_to_human_agents": "human_transfer",
    }
    fault_eligible = tuple(
        sorted(
            name
            for name, family in tool_families.items()
            if family in {"identity_lookup", "entity_read", "catalog_list"}
        )
    )
    return BehaviorProfile.seal(
        profile_id="production-shaped-retail-tools-v1",
        seed="retail-tools-latency-fault-v1",
        latency_percentiles_ms=latency,
        tool_families=tool_families,
        sampling_key=(
            "behavior_profile_seed",
            "conversation_id",
            "tool_name",
            "tool_call_ordinal",
        ),
        fault_policy={
            "placement": "after_validation_before_state_access",
            "maximum_injected_faults_per_conversation": 1,
            "eligible_tool_names": list(fault_eligible),
            "write_tools_excluded": True,
            "rules": [
                {
                    "id": "service_unavailable",
                    "eligible_call_rate": 0.001,
                    "delay_ms": {"minimum": 180.0, "maximum": 450.0},
                    "mcp_is_error": True,
                    "span_error_type": "tool_error",
                },
                {
                    "id": "upstream_timeout",
                    "eligible_call_rate": 0.0005,
                    "delay_ms": {"minimum": 1500.0, "maximum": 2500.0},
                    "mcp_is_error": True,
                    "span_error_type": "tool_error",
                },
            ],
            "actual_transport_timeout_span_error_type": "timeout",
            "intermediate_fault_determines_conversation_outcome": False,
        },
    )
