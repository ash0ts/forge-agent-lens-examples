from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APITimeoutError, InternalServerError

from retail_support_study_v1 import inference
from retail_support_study_v1.constants import (
    SUPPORT_MODEL,
    TRANSPORT_RETRIES,
    USER_MODEL,
)
from retail_support_study_v1.inference import (
    InferenceCallError,
    WandbInferenceClient,
)
from retail_support_study_v1.pricing import (
    CostBudget,
    conservative_request_cost_usd,
)
from retail_support_study_v1.schemas import RunLedger, Usage


class _FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = iter(outcomes)
        self.call_count = 0
        self.last_request: dict[str, Any] | None = None

    def create(self, **request: Any) -> Any:
        self.call_count += 1
        self.last_request = request
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(
    *,
    model: str = SUPPORT_MODEL,
    response_id: str = "resp-reviewed-001",
    finish_reason: str = "stop",
    system_fingerprint: str | None = "fp-reviewed-runtime",
    usage: Any = ...,
) -> Any:
    if usage is ...:
        usage = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=24,
            total_tokens=144,
            cost=None,
        )
    return SimpleNamespace(
        id=response_id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="I found the matching order.",
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _client(outcomes: list[Any], *, cap_usd: float = 1.0) -> WandbInferenceClient:
    client = object.__new__(WandbInferenceClient)
    completions = _FakeCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.budget = CostBudget(cap_usd)
    return client


def _complete(client: WandbInferenceClient) -> Any:
    return client.complete(
        participant="support_agent",
        model=SUPPORT_MODEL,
        temperature=0.0,
        seed=17,
        messages=[{"role": "user", "content": "Where is my order?"}],
        max_tokens=128,
    )


def _timeout() -> APITimeoutError:
    return APITimeoutError(
        httpx.Request("POST", "https://api.inference.wandb.ai/v1/chat/completions")
    )


def _internal_server_error() -> InternalServerError:
    request = httpx.Request(
        "POST", "https://api.inference.wandb.ai/v1/chat/completions"
    )
    return InternalServerError(
        "Internal server error",
        response=httpx.Response(500, request=request),
        body=None,
    )


def test_completion_freezes_provider_identity_and_transport_receipt() -> None:
    client = _client([_response(response_id="resp-123", finish_reason="tool_calls")])

    completion = _complete(client)

    assert completion.record.response_id == "resp-123"
    assert completion.record.system_fingerprint == "fp-reviewed-runtime"
    assert completion.record.finish_reason == "tool_calls"
    assert completion.record.transport_attempt_count == 1
    assert completion.record.requested_model == SUPPORT_MODEL
    assert completion.record.returned_model == SUPPORT_MODEL
    assert completion.record.requested_max_tokens == 128
    assert client._client.chat.completions.last_request["max_tokens"] == 128
    assert completion.record.usage.observed_cost_usd is not None
    assert client.budget.observed_usd > 0
    assert not client.budget.is_uncertain


def test_completion_can_disable_reasoning_for_supported_model() -> None:
    client = _client([_response(model=USER_MODEL)])

    completion = client.complete(
        participant="user_simulator",
        model=USER_MODEL,
        temperature=0.4,
        seed=17,
        messages=[{"role": "user", "content": "Reply as the customer."}],
        max_tokens=512,
        enable_thinking=False,
    )

    assert completion.record.requested_enable_thinking is False
    assert client._client.chat.completions.last_request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_completion_rejects_a_returned_model_alias() -> None:
    client = _client([_response(model=f"{SUPPORT_MODEL}-alias")])

    with pytest.raises(InferenceCallError, match="returned a different model") as error:
        _complete(client)

    # The request was already paid, so its observed usage remains accounted
    # even though the identity mismatch makes the completion unusable.
    assert client.budget.observed_usd > 0
    assert error.value.record.returned_model == f"{SUPPORT_MODEL}-alias"
    assert error.value.record.response_id == "resp-reviewed-001"
    assert error.value.record.response_validation_error


def test_completion_rejects_unusable_finish_reason_with_frozen_evidence() -> None:
    client = _client([_response(finish_reason="length")])

    with pytest.raises(InferenceCallError, match="finish reason") as error:
        _complete(client)

    assert error.value.record.finish_reason == "length"
    assert error.value.record.response_id == "resp-reviewed-001"
    assert error.value.record.usage.observed_cost_usd is not None
    assert client.budget.observed_usd > 0


def test_missing_usage_preserves_response_and_marks_budget_uncertain() -> None:
    messages = [{"role": "user", "content": "Where is my order?"}]
    reserve = conservative_request_cost_usd(
        SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=128,
    )
    client = _client([_response(usage=None)])

    completion = _complete(client)

    assert completion.message["content"] == "I found the matching order."
    assert completion.record.response_id == "resp-reviewed-001"
    assert completion.record.usage.observed_cost_usd is None
    assert completion.record.ambiguous_cost_reserve_usd == reserve
    assert len(completion.record.cost_uncertainty_reasons) == 1
    assert client.budget.observed_usd == 0
    assert client.budget.ambiguous_cost_reserve_usd == reserve
    assert client.budget.cost_exposure_upper_bound_usd == reserve
    assert client.budget.is_uncertain
    assert "omitted usage" in client.budget.uncertainty_reasons[0]


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(
            prompt_tokens="not-an-integer",
            completion_tokens=24,
            total_tokens=144,
            cost=None,
        ),
        SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=24,
            total_tokens=999,
            cost=None,
        ),
        SimpleNamespace(),
    ],
)
def test_malformed_paid_usage_is_frozen_and_reserved_as_uncertain(
    usage: Any,
) -> None:
    messages = [{"role": "user", "content": "Where is my order?"}]
    reserve = conservative_request_cost_usd(
        SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=128,
    )
    client = _client([_response(usage=usage)])

    with pytest.raises(InferenceCallError, match="could not be parsed") as error:
        _complete(client)

    assert error.value.record.response_id == "resp-reviewed-001"
    assert error.value.record.finish_reason == "response_parse_error"
    assert error.value.record.response_validation_error
    assert error.value.record.ambiguous_cost_reserve_usd == reserve
    assert client.budget.observed_usd == 0
    assert client.budget.ambiguous_cost_reserve_usd == reserve
    assert client.budget.is_uncertain


def test_malformed_message_keeps_valid_observed_usage_before_stopping() -> None:
    response = _response()
    response.choices[0].message = SimpleNamespace(content="missing role")
    client = _client([response])

    with pytest.raises(InferenceCallError, match="could not be parsed") as error:
        _complete(client)

    assert error.value.record.response_id == "resp-reviewed-001"
    assert error.value.record.usage.observed_cost_usd is not None
    assert error.value.record.ambiguous_cost_reserve_usd == 0
    assert client.budget.observed_usd > 0
    assert client.budget.ambiguous_cost_reserve_usd == 0


def test_retry_reserves_ambiguous_attempt_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference.time, "sleep", lambda _seconds: None)
    messages = [{"role": "user", "content": "Where is my order?"}]
    reserve = conservative_request_cost_usd(
        SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=128,
    )
    client = _client([_timeout(), _response()])

    completion = _complete(client)

    assert completion.record.transport_attempt_count == 2
    assert completion.record.ambiguous_cost_reserve_usd == reserve
    assert len(completion.record.cost_uncertainty_reasons) == 1
    assert client.budget.observed_usd > 0
    assert client.budget.ambiguous_cost_reserve_usd == reserve
    assert client.budget.cost_exposure_upper_bound_usd == round(
        client.budget.observed_usd + reserve,
        12,
    )
    assert client.budget.cost_exposure_upper_bound_usd <= client.budget.cap_usd
    assert client.budget.is_uncertain


def test_request_must_fit_every_allowed_transport_attempt() -> None:
    messages = [{"role": "user", "content": "Where is my order?"}]
    reserve = conservative_request_cost_usd(
        SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=128,
    )
    client = _client(
        [_response()],
        cap_usd=reserve * (TRANSPORT_RETRIES + 0.5),
    )

    with pytest.raises(RuntimeError, match="cost cap would be exceeded"):
        _complete(client)

    assert client._client.chat.completions.call_count == 0
    assert client.budget.observed_usd == 0
    assert client.budget.ambiguous_cost_reserve_usd == 0


def test_exhausted_retries_keep_each_ambiguous_attempt_within_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference.time, "sleep", lambda _seconds: None)
    messages = [{"role": "user", "content": "Where is my order?"}]
    reserve = conservative_request_cost_usd(
        SUPPORT_MODEL,
        messages=messages,
        tools=None,
        max_output_tokens=128,
    )
    attempts = TRANSPORT_RETRIES + 1
    client = _client(
        [_timeout() for _ in range(attempts)],
        cap_usd=reserve * attempts,
    )

    with pytest.raises(InferenceCallError, match="request unavailable after") as error:
        _complete(client)

    assert client.budget.observed_usd == 0
    assert client.budget.ambiguous_cost_reserve_usd == round(
        reserve * attempts,
        12,
    )
    assert client.budget.cost_exposure_upper_bound_usd <= client.budget.cap_usd
    assert len(client.budget.uncertainty_reasons) == attempts
    assert error.value.record.transport_attempt_count == attempts
    assert error.value.record.ambiguous_cost_reserve_usd == round(
        reserve * attempts,
        12,
    )
    assert error.value.record.finish_reason == "transport_error"


def test_internal_server_retries_wait_30_then_60_and_not_after_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(inference.time, "sleep", sleeps.append)
    attempts = TRANSPORT_RETRIES + 1
    client = _client([_internal_server_error() for _ in range(attempts)])

    with pytest.raises(InferenceCallError, match="request unavailable after"):
        _complete(client)

    assert client._client.chat.completions.call_count == 3
    assert sleeps == [30.0, 60.0]


def test_cached_tokens_use_cached_rate_and_provider_cost_is_preserved() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1_000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=600),
        completion_tokens=100,
        total_tokens=1_100,
        cost=0.123456,
    )
    client = _client([_response(usage=usage)])

    completion = _complete(client)

    assert completion.record.usage.cached_input_tokens == 600
    assert completion.record.usage.provider_reported_cost_usd == 0.123456
    assert completion.record.usage.observed_cost_usd == 0.123456
    assert client.budget.observed_usd == 0.123456


def test_cached_tokens_are_priced_at_locked_cached_rate_without_provider_cost() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1_000,
        prompt_tokens_details={"cached_tokens": 600},
        completion_tokens=100,
        total_tokens=1_100,
        cost=None,
    )
    client = _client([_response(usage=usage)])

    completion = _complete(client)

    expected = inference.observed_cost_usd(
        SUPPORT_MODEL,
        input_tokens=1_000,
        cached_input_tokens=600,
        output_tokens=100,
    )
    assert completion.record.usage.cached_input_tokens == 600
    assert completion.record.usage.provider_reported_cost_usd is None
    assert completion.record.usage.observed_cost_usd == expected


def test_run_ledger_defaults_cost_exposure_for_existing_callers() -> None:
    ledger = RunLedger(
        study_id="study-v1",
        cohort="canary",
        preview_sha256="a" * 64,
        planned_coordinate_ids=[],
        completed_coordinate_ids=[],
        unavailable={},
        observed_cost_usd=1.25,
        bundle_digests={},
        status="complete",
    )

    assert ledger.ambiguous_cost_reserve_usd == 0
    assert ledger.cost_exposure_upper_bound_usd == 1.25


def test_run_ledger_rejects_inconsistent_cost_exposure() -> None:
    with pytest.raises(ValueError, match="cost exposure must equal"):
        RunLedger(
            study_id="study-v1",
            cohort="canary",
            preview_sha256="a" * 64,
            planned_coordinate_ids=[],
            completed_coordinate_ids=[],
            unavailable={},
            observed_cost_usd=1.25,
            ambiguous_cost_reserve_usd=0.5,
            cost_uncertainty_reasons=["missing usage"],
            cost_exposure_upper_bound_usd=1.25,
            bundle_digests={},
            status="uncertain",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_usage_rejects_non_finite_costs(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        Usage(observed_cost_usd=value)
