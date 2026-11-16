from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .constants import WRITE_TOOLS


@dataclass(frozen=True)
class LatencyQuantiles:
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        values = (self.p50_ms, self.p95_ms, self.p99_ms)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("latency quantiles must be finite and positive")
        if not self.p50_ms <= self.p95_ms <= self.p99_ms:
            raise ValueError("latency quantiles must be monotonic")


@dataclass(frozen=True)
class RuntimeBehaviorProfile:
    profile_id: str
    seed: str
    latency_by_family: Mapping[str, LatencyQuantiles]
    service_unavailable_rate: float
    upstream_timeout_rate: float
    service_unavailable_delay_ms: tuple[float, float]
    upstream_timeout_delay_ms: tuple[float, float]
    fault_eligible_tool_names: frozenset[str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeBehaviorProfile:
        # Authoring/tests use the compact runtime shape. An approved preview
        # carries the sealed public BehaviorProfile shape. Normalize either
        # representation before the isolated MCP subprocess starts.
        raw_latency = value.get("latency_by_family")
        sealed_shape = raw_latency is None
        if sealed_shape:
            raw_latency = value.get("latency_percentiles_ms")
        if not isinstance(raw_latency, Mapping):
            raise ValueError("behavior profile requires latency distributions")
        latency: dict[str, LatencyQuantiles] = {}
        for family, quantiles in raw_latency.items():
            if not isinstance(family, str) or not isinstance(quantiles, Mapping):
                raise ValueError("malformed latency family")
            latency[family] = LatencyQuantiles(
                p50_ms=float(quantiles["p50" if sealed_shape else "p50_ms"]),
                p95_ms=float(quantiles["p95" if sealed_shape else "p95_ms"]),
                p99_ms=float(quantiles["p99" if sealed_shape else "p99_ms"]),
            )
        if sealed_shape:
            fault_policy = value.get("fault_policy")
            if not isinstance(fault_policy, Mapping):
                raise ValueError("sealed behavior profile requires fault_policy")
            rules = fault_policy.get("rules")
            if not isinstance(rules, list):
                raise ValueError("sealed behavior profile requires fault rules")
            by_id = {
                str(rule.get("id")): rule
                for rule in rules
                if isinstance(rule, Mapping)
            }
            service_rule = by_id.get("service_unavailable", {})
            timeout_rule = by_id.get("upstream_timeout", {})
            service_rate = float(service_rule.get("eligible_call_rate", 0.0))
            timeout_rate = float(timeout_rule.get("eligible_call_rate", 0.0))
            service_delay = service_rule.get("delay_ms")
            timeout_delay = timeout_rule.get("delay_ms")
            raw_eligible = fault_policy.get("eligible_tool_names")
            if not isinstance(raw_eligible, list) or not all(
                isinstance(name, str) for name in raw_eligible
            ):
                raise ValueError(
                    "sealed behavior profile requires eligible_tool_names"
                )
            fault_eligible_tool_names = frozenset(raw_eligible)
            if isinstance(service_delay, Mapping):
                service_delay = [
                    service_delay.get("minimum"),
                    service_delay.get("maximum"),
                ]
            if isinstance(timeout_delay, Mapping):
                timeout_delay = [
                    timeout_delay.get("minimum"),
                    timeout_delay.get("maximum"),
                ]
        else:
            service_rate = float(value.get("service_unavailable_rate", 0.0))
            timeout_rate = float(value.get("upstream_timeout_rate", 0.0))
            service_delay = value.get("service_unavailable_delay_ms")
            timeout_delay = value.get("upstream_timeout_delay_ms")
            raw_eligible = value.get("fault_eligible_tool_names")
            if raw_eligible is None:
                fault_eligible_tool_names = _DEFAULT_FAULT_ELIGIBLE_TOOLS
            elif isinstance(raw_eligible, (list, tuple, set)) and all(
                isinstance(name, str) for name in raw_eligible
            ):
                fault_eligible_tool_names = frozenset(raw_eligible)
            else:
                raise ValueError("malformed fault_eligible_tool_names")
        if (
            service_rate < 0
            or timeout_rate < 0
            or service_rate + timeout_rate >= 1
        ):
            raise ValueError("fault rates must be non-negative and total less than one")
        return cls(
            profile_id=str(value.get("profile_id") or ""),
            seed=str(value.get("seed") or ""),
            latency_by_family=latency,
            service_unavailable_rate=service_rate,
            upstream_timeout_rate=timeout_rate,
            service_unavailable_delay_ms=_delay_range(
                service_delay,
                field="service_unavailable_delay_ms",
            ),
            upstream_timeout_delay_ms=_delay_range(
                timeout_delay,
                field="upstream_timeout_delay_ms",
            ),
            fault_eligible_tool_names=fault_eligible_tool_names,
        )

    def __post_init__(self) -> None:
        if not self.profile_id or not self.seed:
            raise ValueError("behavior profile requires profile_id and seed")
        missing = EXPECTED_TOOL_FAMILIES - set(self.latency_by_family)
        extra = set(self.latency_by_family) - EXPECTED_TOOL_FAMILIES
        if missing or extra:
            raise ValueError(
                f"behavior latency families drifted: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        unknown = self.fault_eligible_tool_names - _DEFAULT_FAULT_ELIGIBLE_TOOLS
        if unknown:
            raise ValueError(
                f"fault policy includes ineligible tools: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ToolBehavior:
    delay_seconds: float
    fault_code: str | None = None


EXPECTED_TOOL_FAMILIES = frozenset(
    {
        "calculation",
        "identity_lookup",
        "entity_read",
        "catalog_list",
        "mutation",
        "human_transfer",
    }
)

_IDENTITY_TOOLS = frozenset(
    {"find_user_id_by_email", "find_user_id_by_name_zip"}
)
_ENTITY_READ_TOOLS = frozenset(
    {
        "get_item_details",
        "get_order_details",
        "get_product_details",
        "get_user_details",
    }
)
_CATALOG_TOOLS = frozenset({"list_all_product_types"})
_TRANSFER_TOOLS = frozenset({"transfer_to_human_agents"})
_DEFAULT_FAULT_ELIGIBLE_TOOLS = (
    _IDENTITY_TOOLS | _ENTITY_READ_TOOLS | _CATALOG_TOOLS
)


def _delay_range(value: Any, *, field: str) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{field} must contain two numbers")
    lower, upper = float(value[0]), float(value[1])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper):
        raise ValueError(f"{field} must be a finite ascending range")
    return lower, upper


def tool_family(name: str) -> str:
    if name == "calculate":
        return "calculation"
    if name in _IDENTITY_TOOLS:
        return "identity_lookup"
    if name in _ENTITY_READ_TOOLS:
        return "entity_read"
    if name in _CATALOG_TOOLS:
        return "catalog_list"
    if name in WRITE_TOOLS:
        return "mutation"
    if name in _TRANSFER_TOOLS:
        return "human_transfer"
    raise ValueError(f"tool has no reviewed behavior family: {name}")


def _uniform(*parts: object) -> float:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    # The midpoint avoids returning exactly zero while preserving determinism.
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def _piecewise_latency_ms(uniform: float, values: LatencyQuantiles) -> float:
    minimum = values.p50_ms * 0.5
    maximum = values.p99_ms * 1.5
    knots = (
        (0.0, minimum),
        (0.5, values.p50_ms),
        (0.95, values.p95_ms),
        (0.99, values.p99_ms),
        (1.0, maximum),
    )
    for (left_x, left_y), (right_x, right_y) in zip(knots, knots[1:]):
        if uniform <= right_x:
            ratio = (uniform - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return maximum


def select_tool_behavior(
    profile: RuntimeBehaviorProfile,
    *,
    conversation_id: str,
    tool_name: str,
    call_ordinal: int,
    fault_already_injected: bool,
) -> ToolBehavior:
    if not conversation_id or call_ordinal < 1:
        raise ValueError("tool behavior requires a conversation and positive ordinal")
    family = tool_family(tool_name)
    fault_draw = _uniform(
        profile.seed,
        conversation_id,
        tool_name,
        call_ordinal,
        "fault",
    )
    if (
        not fault_already_injected
        and tool_name in profile.fault_eligible_tool_names
    ):
        if fault_draw < profile.service_unavailable_rate:
            return ToolBehavior(
                delay_seconds=(
                    _interpolate_range(
                        profile.service_unavailable_delay_ms,
                        _uniform(
                            profile.seed,
                            conversation_id,
                            tool_name,
                            call_ordinal,
                            "service-delay",
                        ),
                    )
                    / 1000.0
                ),
                fault_code="service_unavailable",
            )
        if fault_draw < (
            profile.service_unavailable_rate + profile.upstream_timeout_rate
        ):
            return ToolBehavior(
                delay_seconds=(
                    _interpolate_range(
                        profile.upstream_timeout_delay_ms,
                        _uniform(
                            profile.seed,
                            conversation_id,
                            tool_name,
                            call_ordinal,
                            "timeout-delay",
                        ),
                    )
                    / 1000.0
                ),
                fault_code="upstream_timeout",
            )
    latency = _piecewise_latency_ms(
        _uniform(
            profile.seed,
            conversation_id,
            tool_name,
            call_ordinal,
            "latency",
        ),
        profile.latency_by_family[family],
    )
    return ToolBehavior(delay_seconds=latency / 1000.0)


def _interpolate_range(value: tuple[float, float], uniform: float) -> float:
    return value[0] + uniform * (value[1] - value[0])
