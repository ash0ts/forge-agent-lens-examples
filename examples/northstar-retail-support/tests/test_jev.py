from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from retail_support_study_v1.jev import (
    ASSESSMENT_QUESTION_SET_SHA256,
    DEFAULT_JEV_MODEL,
    ROUTE_QUESTION_SET_SHA256,
    JevBackend,
    JevConfig,
    JevDecisionEngine,
    JevMode,
    JevSource,
)
from retail_support_study_v1.traffic_config import (
    GENERALIST_AGENT_ID,
    ORDER_AGENT_ID,
    RETURNS_AGENT_ID,
)


def _route_response(
    choice: str,
    confidence: float,
    *,
    model: str = DEFAULT_JEV_MODEL,
) -> Any:
    probabilities = {
        "order_management": 0.05,
        "returns_resolution": 0.05,
        "generalist": 0.05,
    }
    probabilities[choice] = confidence
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=21, output_tokens=0),
        choices={
            "agent_profile": SimpleNamespace(
                choice=choice,
                confidence=confidence,
                probabilities=probabilities,
            )
        },
        nouls={},
    )


def _assessment_response(**probabilities: float) -> Any:
    return SimpleNamespace(
        model=DEFAULT_JEV_MODEL,
        usage=SimpleNamespace(input_tokens=55, output_tokens=0),
        choices={},
        nouls={
            name: SimpleNamespace(noul=probability)
            for name, probability in probabilities.items()
        },
    )


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "state": state,
                "questions": questions,
                "model": model,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def test_off_mode_is_deterministic_and_never_calls_jev() -> None:
    client = _FakeClient([])
    engine = JevDecisionEngine(JevConfig(mode="off"), client=client)

    decision = engine.route("Return the damaged mug.", RETURNS_AGENT_ID)

    assert decision.selected_profile_id == RETURNS_AGENT_ID
    assert decision.applied_profile_id == RETURNS_AGENT_ID
    assert decision.source == JevSource.DETERMINISTIC
    assert decision.returned_model is None
    assert engine.assess_turn(
        user_message="Return it.",
        assistant_message="Which order?",
        recent_messages=[],
        tool_events=[],
    ) is None
    assert client.calls == []


def test_shadow_records_jev_choice_but_applies_deterministic_route() -> None:
    client = _FakeClient([_route_response("order_management", 0.91)])
    calls: list[dict[str, Any]] = []

    def call_runner(
        *,
        operation: str,
        state: Any,
        questions: Mapping[str, Any],
        backend: JevBackend,
        requested_model: str,
        timeout_seconds: float,
        invoke: Callable[[], Any],
    ) -> Any:
        calls.append(
            {
                "operation": operation,
                "state": state,
                "questions": questions,
                "backend": backend,
                "model": requested_model,
                "timeout": timeout_seconds,
            }
        )
        return invoke()

    engine = JevDecisionEngine(
        JevConfig(mode="shadow"),
        client=client,
        call_runner=call_runner,
    )
    decision = engine.route(
        "I need to return this and also change a pending order.",
        GENERALIST_AGENT_ID,
    )

    assert decision.selected_profile_id == ORDER_AGENT_ID
    assert decision.applied_profile_id == GENERALIST_AGENT_ID
    assert decision.confidence == pytest.approx(0.91)
    assert decision.source == JevSource.JEV
    assert decision.usage.input_tokens == 21
    assert calls[0]["operation"] == "route"
    assert calls[0]["backend"] == JevBackend.LIVE
    assert calls[0]["state"] == {
        "first_customer_message": (
            "I need to return this and also change a pending order."
        )
    }
    assert set(calls[0]["questions"]) == {"agent_profile"}
    assert calls[0]["model"] == DEFAULT_JEV_MODEL


def test_route_mode_requires_confidence_and_falls_back_safely() -> None:
    client = _FakeClient(
        [
            _route_response("order_management", 0.92),
            _route_response("order_management", 0.62),
            RuntimeError("secret provider body"),
        ]
    )
    engine = JevDecisionEngine(
        JevConfig(mode="route", route_confidence_threshold=0.75),
        client=client,
    )

    applied = engine.route("Change my pending order.", GENERALIST_AGENT_ID)
    assert applied.applied_profile_id == ORDER_AGENT_ID
    assert applied.source == JevSource.JEV

    uncertain = engine.route("Can you help with this?", GENERALIST_AGENT_ID)
    assert uncertain.selected_profile_id == ORDER_AGENT_ID
    assert uncertain.applied_profile_id == GENERALIST_AGENT_ID
    assert uncertain.source == JevSource.DETERMINISTIC_FALLBACK
    assert uncertain.fallback_reason == "confidence_below_threshold"

    failed = engine.route("Return the broken item.", RETURNS_AGENT_ID)
    assert failed.applied_profile_id == RETURNS_AGENT_ID
    assert failed.source == JevSource.DETERMINISTIC_FALLBACK
    assert failed.fallback_reason == "route_failed:RuntimeError"
    assert "secret" not in failed.model_dump_json()


def test_post_turn_assessment_uses_three_independent_nouls() -> None:
    client = _FakeClient(
        [
            _assessment_response(
                redundant_confirmation=0.94,
                partial_completion=0.12,
                unsupported_definitive_answer=0.31,
            )
        ]
    )
    engine = JevDecisionEngine(JevConfig(mode="shadow"), client=client)

    assessment = engine.assess_turn(
        user_message="Yes, change that address.",
        assistant_message="Please confirm the same address change.",
        recent_messages=[
            {"role": "assistant", "content": "Should I change it?"},
            {"role": "user", "content": "Yes, change that address."},
            {"role": "assistant", "content": "Please confirm it again."},
        ],
        tool_events=[{"name": "get_order_details", "status": "executed"}],
    )

    assert assessment is not None
    assert assessment.source == JevSource.JEV
    assert assessment.probabilities == {
        "partial_completion": pytest.approx(0.12),
        "redundant_confirmation": pytest.approx(0.94),
        "unsupported_definitive_answer": pytest.approx(0.31),
    }
    questions = client.calls[0]["questions"]
    assert set(questions) == {
        "redundant_confirmation",
        "partial_completion",
        "unsupported_definitive_answer",
    }
    assert all(question.type == "noul" for question in questions.values())


def test_question_sets_and_public_config_are_stable() -> None:
    engine = JevDecisionEngine()
    public = engine.public_config()

    assert len(ROUTE_QUESTION_SET_SHA256) == 64
    assert len(ASSESSMENT_QUESTION_SET_SHA256) == 64
    assert public["requested_model"] == "jev-1.13.0"
    assert public["route_confidence_threshold"] == pytest.approx(0.75)
    assert public["route_question_set_sha256"] == ROUTE_QUESTION_SET_SHA256
    assert (
        public["assessment_question_set_sha256"]
        == ASSESSMENT_QUESTION_SET_SHA256
    )


def test_environment_on_aliases_to_safe_shadow_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETAIL_DEMO_JEV_MODE", "on")
    monkeypatch.setenv("RETAIL_DEMO_JEV_BACKEND", "live")

    config = JevConfig.from_environment()

    assert config.mode == JevMode.SHADOW
    assert config.backend == JevBackend.LIVE


def test_mock_backend_routes_and_assesses_without_a_credential() -> None:
    engine = JevDecisionEngine(JevConfig(mode="shadow", backend="mock"))

    specialist = engine.route(
        "The headphones arrived damaged and I need to return them.",
        RETURNS_AGENT_ID,
    )
    assert specialist.source == JevSource.MOCK
    assert specialist.backend == JevBackend.MOCK
    assert specialist.selected_profile_id == RETURNS_AGENT_ID
    assert specialist.applied_profile_id == RETURNS_AGENT_ID
    assert specialist.confidence == pytest.approx(0.89)
    assert specialist.returned_model == "mock-jev-decision-v1"

    mixed = engine.route(
        "Return the broken mug and cancel my other pending order.",
        GENERALIST_AGENT_ID,
    )
    assert mixed.selected_profile_id == GENERALIST_AGENT_ID
    assert mixed.confidence == pytest.approx(0.58)

    assessment = engine.assess_turn(
        user_message="Yes, confirm that exact address change.",
        assistant_message="Please confirm again before I change it.",
        recent_messages=[
            {"role": "user", "content": "Yes, confirm that exact address change."},
        ],
        tool_events=[],
    )
    assert assessment is not None
    assert assessment.backend == JevBackend.MOCK
    assert assessment.source == JevSource.MOCK
    assert assessment.probabilities["redundant_confirmation"] == pytest.approx(0.92)

    engine.close()
