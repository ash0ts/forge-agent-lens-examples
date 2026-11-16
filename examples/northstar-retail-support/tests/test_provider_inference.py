from __future__ import annotations

from typing import Any

import httpx
import pytest

from retail_support_study_v1 import inference
from retail_support_study_v1.inference import (
    ANTHROPIC_MESSAGES_URL,
    AnthropicInferenceClient,
    InferenceCallError,
    WandbInferenceClient,
)
from retail_support_study_v1.pricing import CostBudget

MODEL = "claude-haiku-4-5-20251001"


def _price_lock() -> dict[str, Any]:
    return {
        "models": {
            MODEL: {
                "input_per_million_tokens": 1.0,
                "cached_input_per_million_tokens": 0.1,
                "output_per_million_tokens": 5.0,
            }
        }
    }


def _response(
    *,
    status_code: int = 200,
    payload: dict[str, Any] | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", ANTHROPIC_MESSAGES_URL)
    if payload is None:
        payload = {
            "id": "msg-reviewed-001",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "I found the order."}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 15,
            },
        }
    return httpx.Response(status_code, request=request, json=payload)


class _FakeHTTPClient:
    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self._outcomes = iter(outcomes)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
        self.requests.append({"url": url, "json": json})
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[httpx.Response | Exception],
) -> tuple[AnthropicInferenceClient, _FakeHTTPClient]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-logged")
    transport = _FakeHTTPClient(outcomes)
    client = AnthropicInferenceClient(
        cost_cap_usd=5.0,
        price_lock=_price_lock(),
        http_client=transport,  # type: ignore[arg-type]
    )
    return client, transport


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Read one order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _complete(
    client: AnthropicInferenceClient,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> Any:
    return client.complete(
        participant="support_agent",
        model=MODEL,
        temperature=0.0,
        seed=None,
        messages=messages
        or [
            {"role": "system", "content": "Help with retail support."},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Where is order W-123?"},
        ],
        max_tokens=256,
        tools=_tools(),
    )


def test_anthropic_text_completion_is_normalized_and_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(monkeypatch, [_response()])

    completion = _complete(client)

    assert completion.message == {
        "role": "assistant",
        "content": "I found the order.",
    }
    assert completion.record.requested_model == MODEL
    assert completion.record.returned_model == MODEL
    assert completion.record.response_id == "msg-reviewed-001"
    assert completion.record.finish_reason == "stop"
    assert completion.record.usage.input_tokens == 130
    assert completion.record.usage.cached_input_tokens == 20
    assert completion.record.usage.output_tokens == 15
    assert completion.record.usage.total_tokens == 145
    assert completion.record.usage.observed_cost_usd is not None
    assert client.budget.observed_usd > 0
    assert client.provider_name == "anthropic"
    assert client.server_address == "api.anthropic.com"

    request = transport.requests[0]
    assert request["url"] == ANTHROPIC_MESSAGES_URL
    assert request["json"]["model"] == MODEL
    assert request["json"]["system"] == "Help with retail support."
    assert request["json"]["messages"][0] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi! How can I help?"}],
    }
    assert request["json"]["tools"][0]["name"] == "get_order_details"
    assert request["json"]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_anthropic_omits_temperature_when_route_does_not_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(monkeypatch, [_response()])

    completion = client.complete(
        participant="support_agent",
        model=MODEL,
        temperature=None,
        seed=None,
        messages=[{"role": "user", "content": "Where is my order?"}],
        max_tokens=256,
        tools=_tools(),
    )

    assert "temperature" not in transport.requests[0]["json"]
    assert completion.record.temperature is None


def test_anthropic_can_disable_adaptive_thinking_without_recording_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(monkeypatch, [_response()])

    completion = client.complete(
        participant="support_agent",
        model=MODEL,
        temperature=None,
        seed=None,
        enable_thinking=False,
        messages=[{"role": "user", "content": "Where is my order?"}],
        max_tokens=256,
        tools=_tools(),
    )

    assert transport.requests[0]["json"]["thinking"] == {"type": "disabled"}
    assert completion.record.requested_enable_thinking is False


def test_anthropic_tool_use_and_tool_result_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "msg-tool-001",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {"type": "text", "text": "I will check that."},
            {
                "type": "tool_use",
                "id": "toolu_001",
                "name": "get_order_details",
                "input": {"order_id": "W-123"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 25},
    }
    client, transport = _client(monkeypatch, [_response(payload=payload)])
    messages = [
        {"role": "system", "content": "Help with retail support."},
        {"role": "user", "content": "Where is order W-123?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "toolu_prior",
                    "type": "function",
                    "function": {
                        "name": "get_order_details",
                        "arguments": '{"order_id":"W-122"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_prior",
            "content": '{"error":"service_unavailable"}',
        },
    ]

    completion = _complete(client, messages=messages)

    assert completion.message == {
        "role": "assistant",
        "content": "I will check that.",
        "tool_calls": [
            {
                "id": "toolu_001",
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "arguments": '{"order_id":"W-123"}',
                },
            }
        ],
    }
    assert completion.record.finish_reason == "tool_calls"
    converted = transport.requests[0]["json"]["messages"]
    assert converted[-2]["content"][0] == {
        "type": "tool_use",
        "id": "toolu_prior",
        "name": "get_order_details",
        "input": {"order_id": "W-122"},
    }
    assert converted[-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_prior",
        "content": '{"error":"service_unavailable"}',
        "is_error": True,
    }


def test_malformed_anthropic_tool_input_freezes_paid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "msg-malformed-tool",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_bad",
                "name": "get_order_details",
                "input": "not-an-object",
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    client, _transport = _client(monkeypatch, [_response(payload=payload)])

    with pytest.raises(InferenceCallError, match="could not be parsed") as error:
        _complete(client)

    assert error.value.record.response_id == "msg-malformed-tool"
    assert error.value.record.response_validation_error
    assert "stage=content" in error.value.record.response_validation_error
    assert error.value.record.usage.observed_cost_usd is not None
    assert client.budget.observed_usd > 0
    assert client.budget.ambiguous_cost_reserve_usd == 0


def test_anthropic_thinking_block_fails_without_leaking_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_reasoning = "private-reasoning-must-not-enter-the-receipt"
    payload = {
        "id": "msg-thinking-block",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {
                "type": "thinking",
                "thinking": private_reasoning,
                "signature": "opaque-signature",
            },
            {
                "type": "tool_use",
                "id": "toolu_thinking",
                "name": "get_order_details",
                "input": {"order_id": "W-123"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 25},
    }
    client, _transport = _client(monkeypatch, [_response(payload=payload)])

    with pytest.raises(InferenceCallError, match="stage=content") as error:
        _complete(client)

    reason = error.value.record.response_validation_error or ""
    assert "error=ValueError" in reason
    assert private_reasoning not in reason
    assert private_reasoning not in str(error.value)
    assert error.value.record.output_message == {}
    assert error.value.record.usage.observed_cost_usd is not None


def test_anthropic_usage_parse_failure_identifies_stage_without_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_value = "private-invalid-token-value"
    payload = {
        "id": "msg-invalid-usage",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Completed."}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": private_value,
            "output_tokens": 10,
        },
    }
    client, _transport = _client(monkeypatch, [_response(payload=payload)])

    with pytest.raises(InferenceCallError, match="stage=usage") as error:
        _complete(client)

    reason = error.value.record.response_validation_error or ""
    assert "error=ValueError" in reason
    assert private_value not in reason
    assert private_value not in str(error.value)


def test_anthropic_refusal_is_frozen_and_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "msg-refusal",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "I cannot help with that."}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    client, _transport = _client(monkeypatch, [_response(payload=payload)])

    with pytest.raises(InferenceCallError, match="stop reason") as error:
        _complete(client)

    assert error.value.record.finish_reason == "refusal"
    assert error.value.record.output_message["content"] == "I cannot help with that."
    assert client.budget.observed_usd > 0


def test_anthropic_retries_same_route_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(inference.time, "sleep", sleeps.append)
    client, transport = _client(
        monkeypatch,
        [_response(status_code=503, payload={"type": "error"}), _response()],
    )

    completion = _complete(client)

    assert len(transport.requests) == 2
    assert {request["json"]["model"] for request in transport.requests} == {MODEL}
    assert completion.record.transport_attempt_count == 2
    assert completion.record.ambiguous_cost_reserve_usd > 0
    assert len(completion.record.cost_uncertainty_reasons) == 1
    assert sleeps == [1.0]


def test_anthropic_rejects_unsupported_call_modes_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(monkeypatch, [_response()])
    request = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    }

    with pytest.raises(ValueError, match="reserved for support-agent"):
        client.complete(participant="user_simulator", seed=None, **request)
    with pytest.raises(ValueError, match="do not support a request seed"):
        client.complete(participant="support_agent", seed=7, **request)
    with pytest.raises(ValueError, match="thinking is not enabled"):
        client.complete(
            participant="support_agent",
            seed=None,
            enable_thinking=True,
            **request,
        )

    assert transport.requests == []


def test_anthropic_requires_credential_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required") as error:
        AnthropicInferenceClient(
            cost_cap_usd=1.0,
            price_lock=_price_lock(),
        )

    assert "test-key" not in str(error.value)


def test_provider_clients_can_share_one_coordinate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test-key-never-logged")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key-never-logged")
    shared = CostBudget(5.0, price_lock=_price_lock())

    wandb = WandbInferenceClient(budget=shared)
    anthropic = AnthropicInferenceClient(
        budget=shared,
        http_client=_FakeHTTPClient([_response()]),  # type: ignore[arg-type]
    )

    assert wandb.budget is shared
    assert anthropic.budget is shared


@pytest.mark.parametrize(
    "client_type,credential",
    [
        (WandbInferenceClient, "WANDB_API_KEY"),
        (AnthropicInferenceClient, "ANTHROPIC_API_KEY"),
    ],
)
def test_shared_budget_rejects_duplicate_numeric_state(
    monkeypatch: pytest.MonkeyPatch,
    client_type: type[WandbInferenceClient] | type[AnthropicInferenceClient],
    credential: str,
) -> None:
    monkeypatch.setenv(credential, "test-key-never-logged")
    shared = CostBudget(5.0, price_lock=_price_lock())

    with pytest.raises(ValueError, match="mutually exclusive"):
        client_type(budget=shared, cost_cap_usd=5.0)
