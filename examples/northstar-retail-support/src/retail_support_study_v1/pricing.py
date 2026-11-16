from __future__ import annotations

import math
from typing import Any

from .canonical import load_json
from .constants import PACKAGE_ROOT


def pricing_lock() -> dict[str, Any]:
    return load_json(PACKAGE_ROOT / "model-pricing.lock.json")


def observed_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    price_lock: dict[str, Any] | None = None,
) -> float:
    locked = pricing_lock() if price_lock is None else price_lock
    prices = locked["models"].get(model)
    if prices is None:
        raise ValueError(f"model has no locked price: {model}")
    if cached_input_tokens < 0 or cached_input_tokens > input_tokens:
        raise ValueError("cached input tokens must be within total input tokens")
    uncached_input_tokens = input_tokens - cached_input_tokens
    return round(
        (uncached_input_tokens * float(prices["input_per_million_tokens"])) / 1_000_000
        + (cached_input_tokens * float(prices["cached_input_per_million_tokens"]))
        / 1_000_000
        + (output_tokens * float(prices["output_per_million_tokens"])) / 1_000_000,
        12,
    )


def conservative_request_cost_usd(
    model: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_output_tokens: int,
    price_lock: dict[str, Any] | None = None,
) -> float:
    # A UTF-8 byte is a deliberately conservative upper bound for token count
    # in this English workload. It can over-reserve, but cannot turn missing
    # usage into a zero-dollar request.
    input_upper_bound = sum(len(str(message).encode("utf-8")) for message in messages)
    if tools:
        input_upper_bound += len(str(tools).encode("utf-8"))
    return observed_cost_usd(
        model,
        input_tokens=input_upper_bound,
        output_tokens=max_output_tokens,
        price_lock=price_lock,
    )


class CostBudget:
    def __init__(
        self,
        cap_usd: float,
        *,
        observed_usd: float = 0.0,
        ambiguous_cost_reserve_usd: float = 0.0,
        uncertainty_reasons: list[str] | None = None,
        price_lock: dict[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("cost cap must be positive")
        if not math.isfinite(observed_usd) or observed_usd < 0:
            raise ValueError("observed cost must be finite and non-negative")
        if (
            not math.isfinite(ambiguous_cost_reserve_usd)
            or ambiguous_cost_reserve_usd < 0
        ):
            raise ValueError("ambiguous cost reserve must be finite and non-negative")
        self.cap_usd = float(cap_usd)
        self.observed_usd = float(observed_usd)
        self.ambiguous_cost_reserve_usd = float(ambiguous_cost_reserve_usd)
        self._uncertainty_reasons = list(uncertainty_reasons or [])
        self._price_lock = pricing_lock() if price_lock is None else price_lock
        if self.ambiguous_cost_reserve_usd and not self._uncertainty_reasons:
            self._uncertainty_reasons.append("prior request cost could not be observed")

    @property
    def cost_exposure_upper_bound_usd(self) -> float:
        return round(
            self.observed_usd + self.ambiguous_cost_reserve_usd,
            12,
        )

    @property
    def is_uncertain(self) -> bool:
        return self.ambiguous_cost_reserve_usd > 0

    @property
    def uncertainty_reasons(self) -> tuple[str, ...]:
        return tuple(self._uncertainty_reasons)

    def require_capacity(
        self,
        model: str,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int,
        max_attempts: int = 1,
    ) -> float:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        reserve_per_attempt = conservative_request_cost_usd(
            model,
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            price_lock=self._price_lock,
        )
        request_reserve = round(reserve_per_attempt * max_attempts, 12)
        if self.cost_exposure_upper_bound_usd + request_reserve > self.cap_usd:
            raise RuntimeError(
                f"cost cap would be exceeded: observed={self.observed_usd:.6f}, "
                "ambiguous_reserve="
                f"{self.ambiguous_cost_reserve_usd:.6f}, "
                f"reserve={request_reserve:.6f}, cap={self.cap_usd:.2f}"
            )
        return reserve_per_attempt

    def record(self, cost_usd: float) -> None:
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("observed cost must be finite and non-negative")
        self.observed_usd = round(self.observed_usd + float(cost_usd), 12)

    def record_ambiguous(self, reserve_usd: float, *, reason: str) -> None:
        if not math.isfinite(reserve_usd) or reserve_usd < 0:
            raise ValueError("ambiguous reserve must be finite and non-negative")
        if not reason.strip():
            raise ValueError("ambiguous reserve requires a reason")
        self.ambiguous_cost_reserve_usd = round(
            self.ambiguous_cost_reserve_usd + float(reserve_usd),
            12,
        )
        self._uncertainty_reasons.append(reason.strip())
