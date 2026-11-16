from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from retail_support_study_v1.inference import Completion
from retail_support_study_v1.model_probe import probe_model_routes
from retail_support_study_v1.pricing import CostBudget
from retail_support_study_v1.schemas import ModelCall, ModelRoute, Usage


@dataclass
class _ProbeClient:
    route: ModelRoute
    fail: bool = False

    def __post_init__(self) -> None:
        self.provider_name = self.route.provider_name
        self.server_address = self.route.server_address
        self.budget = CostBudget(5.0)

    def complete(self, **request: Any) -> Completion:
        if self.fail:
            raise RuntimeError("safe test failure")
        output = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"probe-{self.route.route_id}",
                    "type": "function",
                    "function": {
                        "name": "inspect_order",
                        "arguments": json.dumps({"order_id": "R1234567"}),
                    },
                }
            ],
        }
        return Completion(
            message=output,
            record=ModelCall(
                call_id=f"call-{self.route.route_id}",
                participant="support_agent",
                requested_model=self.route.model,
                returned_model=self.route.model,
                finish_reason="tool_calls",
                temperature=0.0,
                requested_max_tokens=int(request["max_tokens"]),
                input_messages=list(request["messages"]),
                output_message=output,
                usage=Usage(
                    input_tokens=20,
                    output_tokens=8,
                    total_tokens=28,
                    observed_cost_usd=0.001,
                ),
            ),
        )


def test_probe_covers_all_five_routes_and_seals_success(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    output = tmp_path / "probe.json"
    monkeypatch.setattr(
        "retail_support_study_v1.model_probe.MODEL_PROBE_PATH",
        output,
    )

    result = probe_model_routes(client_factory=lambda route: _ProbeClient(route))

    assert result["status"] == "ready_for_preview"
    assert len(result["routes"]) == 5
    assert {row["provider_name"] for row in result["routes"]} == {
        "wandb",
        "anthropic",
    }
    assert all(row["status"] == "passed" for row in result["routes"])
    assert json.loads(output.read_text()) == result


def test_failed_wandb_route_reports_fallbacks_without_substituting(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.model_probe.MODEL_PROBE_PATH",
        tmp_path / "probe.json",
    )

    def factory(route: ModelRoute) -> _ProbeClient:
        return _ProbeClient(
            route,
            fail=route.route_id == "wandb-deepseek-v4-1-flash",
        )

    result = probe_model_routes(client_factory=factory)
    failed = next(
        row for row in result["routes"] if row["status"] == "failed"
    )

    assert result["status"] == "blocked"
    assert failed["requested_model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert failed["wandb_fallback_candidates"] == [
        "openai/gpt-oss-120b",
        "Qwen/Qwen3.6-35B-A3B",
    ]
