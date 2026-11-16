from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typesafe_sdk import (
    Choice,
    JSONContent,
    Noul,
    NoulCriteria,
    Question,
    SystemOneResponse,
    TypeSafeClient,
)

from .canonical import digest_value
from .traffic_config import (
    GENERALIST_AGENT_ID,
    ORDER_AGENT_ID,
    RETURNS_AGENT_ID,
)

DEFAULT_JEV_MODEL = "jev-1.13.0"
MOCK_JEV_MODEL = "mock-jev-decision-v1"
DEFAULT_ROUTE_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_JEV_TIMEOUT_SECONDS = 2.0

ROUTE_QUESTION_SET_ID = "retail-support-first-message-route-v1"
ASSESSMENT_QUESTION_SET_ID = "retail-support-turn-assessment-v1"

_ROUTE_QUESTION_SPEC: dict[str, Any] = {
    "agent_profile": {
        "type": "choice",
        "instructions": (
            "Using only `first_customer_message`, choose the support role whose "
            "tool boundary best fits the request. Choose `generalist` for mixed, "
            "ambiguous, account-level, catalog, or unmatched requests."
        ),
        "criteria": {
            "order_management": (
                "Pending-order cancellation; changing a pending order's delivery "
                "address, items, or payment; or changing the user's saved address."
            ),
            "returns_resolution": (
                "A delivered-item return, refund, exchange, replacement, damaged "
                "item, or wrong-item request."
            ),
            "generalist": (
                "A mixed request spanning both specialist roles, an ambiguous or "
                "unmatched request, or account-level and catalog support."
            ),
        },
    }
}

_ASSESSMENT_QUESTION_SPEC: dict[str, Any] = {
    "redundant_confirmation": {
        "type": "noul",
        "instructions": (
            "In `current_turn`, does the assistant ask the user to confirm the "
            "same action and target after the visible conversation already contains "
            "an explicit confirmation, without a changed action or new requirement?"
        ),
        "criteria": {
            "true": (
                "The user already confirmed this exact action and target, but the "
                "assistant asks for the same confirmation again instead of acting or "
                "explaining a genuinely new requirement."
            ),
            "false": (
                "This is the first required confirmation, the action or target "
                "changed, the prior reply was not explicit assent, or the assistant "
                "does not ask for confirmation."
            ),
        },
    },
    "partial_completion": {
        "type": "noul",
        "instructions": (
            "Does the assistant address only part of the user's current multi-part "
            "request and stop without completing, deferring, or explaining the "
            "remaining requested part?"
        ),
        "criteria": {
            "true": (
                "The current user request contains multiple requested outcomes and "
                "the assistant leaves at least one unaddressed without explanation."
            ),
            "false": (
                "The request has one outcome, every requested part is addressed, or "
                "the assistant clearly explains why a remaining part is deferred."
            ),
        },
    },
    "unsupported_definitive_answer": {
        "type": "noul",
        "instructions": (
            "Does the assistant state a definite fact, action outcome, or "
            "recommendation that is not supported by the visible messages and tool "
            "evidence in `current_turn`?"
        ),
        "criteria": {
            "true": (
                "The assistant presents an unsupported claim as settled, including "
                "using the wrong record or claiming an action succeeded without "
                "matching tool evidence."
            ),
            "false": (
                "The claim is supported by visible evidence, is clearly qualified, "
                "or no definitive claim is made."
            ),
        },
    },
}

ROUTE_QUESTION_SET_SHA256 = digest_value(
    {"question_set_id": ROUTE_QUESTION_SET_ID, **_ROUTE_QUESTION_SPEC}
)
ASSESSMENT_QUESTION_SET_SHA256 = digest_value(
    {"question_set_id": ASSESSMENT_QUESTION_SET_ID, **_ASSESSMENT_QUESTION_SPEC}
)

_ROLE_TO_PROFILE = {
    "order_management": ORDER_AGENT_ID,
    "returns_resolution": RETURNS_AGENT_ID,
    "generalist": GENERALIST_AGENT_ID,
}
_ASSESSMENT_KEYS = frozenset(_ASSESSMENT_QUESTION_SPEC)


class JevMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    ROUTE = "route"


class JevBackend(StrEnum):
    LIVE = "live"
    MOCK = "mock"


class JevSource(StrEnum):
    DETERMINISTIC = "deterministic"
    JEV = "jev"
    MOCK = "mock"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    UNAVAILABLE = "unavailable"


class JevConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: JevMode = JevMode.OFF
    backend: JevBackend = JevBackend.LIVE
    requested_model: str = DEFAULT_JEV_MODEL
    route_confidence_threshold: float = Field(
        default=DEFAULT_ROUTE_CONFIDENCE_THRESHOLD,
        ge=0,
        le=1,
    )
    timeout_seconds: float = Field(
        default=DEFAULT_JEV_TIMEOUT_SECONDS,
        gt=0,
        le=60,
    )

    @model_validator(mode="before")
    @classmethod
    def default_mock_model(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            values = dict(value)
            if values.get("backend") in {
                JevBackend.MOCK,
                JevBackend.MOCK.value,
            }:
                values["requested_model"] = MOCK_JEV_MODEL
            return values
        return value

    @model_validator(mode="after")
    def validate_model(self) -> "JevConfig":
        if not self.requested_model.strip():
            raise ValueError("Jev requested_model must not be empty")
        return self

    @classmethod
    def from_environment(cls) -> "JevConfig":
        backend = JevBackend(
            os.environ.get("RETAIL_DEMO_JEV_BACKEND", JevBackend.LIVE.value)
        )
        configured_mode = os.environ.get(
            "RETAIL_DEMO_JEV_MODE",
            JevMode.OFF.value,
        ).strip().casefold()
        # `on` is an intuitive demo setting, but it must not silently give a
        # remote decision authority over routing. Treat it as the safe
        # observational mode; applying a decision still requires `route`.
        mode = JevMode.SHADOW if configured_mode == "on" else configured_mode
        default_model = (
            MOCK_JEV_MODEL if backend == JevBackend.MOCK else DEFAULT_JEV_MODEL
        )
        return cls(
            mode=mode,
            backend=backend,
            requested_model=os.environ.get(
                "RETAIL_DEMO_JEV_MODEL",
                default_model,
            ),
            route_confidence_threshold=float(
                os.environ.get(
                    "RETAIL_DEMO_JEV_ROUTE_CONFIDENCE_THRESHOLD",
                    str(DEFAULT_ROUTE_CONFIDENCE_THRESHOLD),
                )
            ),
            timeout_seconds=float(
                os.environ.get(
                    "RETAIL_DEMO_JEV_TIMEOUT_SECONDS",
                    str(DEFAULT_JEV_TIMEOUT_SECONDS),
                )
            ),
        )


class JevUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class JevRoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: JevMode
    backend: JevBackend
    selected_profile_id: str
    applied_profile_id: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] = Field(default_factory=dict)
    requested_model: str
    returned_model: str | None = None
    usage: JevUsage = Field(default_factory=JevUsage)
    duration_ms: float = Field(ge=0)
    source: JevSource
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> "JevRoutingDecision":
        if set(self.probabilities) - set(_ROLE_TO_PROFILE):
            raise ValueError("Jev routing probabilities contain an unknown role")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("Jev routing probabilities must be between zero and one")
        return self


class JevTurnAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: JevBackend
    probabilities: dict[str, float] = Field(default_factory=dict)
    requested_model: str
    returned_model: str | None = None
    usage: JevUsage = Field(default_factory=JevUsage)
    duration_ms: float = Field(ge=0)
    source: JevSource
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> "JevTurnAssessment":
        keys = set(self.probabilities)
        available_sources = {JevSource.JEV, JevSource.MOCK}
        if self.source in available_sources and keys != _ASSESSMENT_KEYS:
            raise ValueError("Jev assessment must contain every frozen question")
        if self.source not in available_sources and keys:
            raise ValueError("unavailable Jev assessment cannot contain probabilities")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("Jev assessment probabilities must be between zero and one")
        return self


class SystemOneClient(Protocol):
    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> SystemOneResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _MockUsage:
    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True)
class _MockChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    type: str = "choice"


@dataclass(frozen=True)
class _MockNoulAnswer:
    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class _MockSystemOneResponse:
    model: str
    usage: _MockUsage
    answers: dict[str, _MockChoiceAnswer | _MockNoulAnswer]

    @property
    def choices(self) -> dict[str, _MockChoiceAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, _MockChoiceAnswer)
        }

    @property
    def nouls(self) -> dict[str, _MockNoulAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, _MockNoulAnswer)
        }


def _mock_token_count(value: Any) -> int:
    # The mock does not bill or tokenize. This bounded approximation only makes
    # the demo receipt structurally resemble a typed decision response.
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return max(1, (len(encoded) + 3) // 4)


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


class MockJevClient:
    """Deterministic local stand-in for demonstrating the Jev integration.

    It implements the same narrow System One-shaped contract used by the live
    client, but it is deliberately simple and is never represented as a
    TypeSafe result. It exists so the UI and Agent Lens trace can demonstrate
    where typed decisions fit before a TypeSafe credential is available.
    """

    _RETURN_PHRASES = (
        "return",
        "refund",
        "exchange",
        "replacement",
        "replace",
        "damaged",
        "broken",
        "wrong item",
        "arrived cracked",
    )
    _ORDER_PHRASES = (
        "cancel",
        "pending order",
        "delivery address",
        "shipping address",
        "change the address",
        "payment method",
        "add item",
        "remove item",
        "edit my order",
        "modify my order",
    )
    _CONFIRMATION_PHRASES = (
        "yes",
        "confirm",
        "confirmed",
        "go ahead",
        "proceed",
        "do it",
    )
    _ASK_CONFIRMATION_PHRASES = (
        "please confirm",
        "confirm again",
        "reply yes",
        "yes or no",
        "shall i proceed",
        "should i proceed",
    )
    _DEFINITIVE_PHRASES = (
        "has been processed",
        "has been completed",
        "was completed",
        "is complete",
        "all done",
        "successfully changed",
        "successfully cancelled",
        "successfully returned",
        "will arrive",
    )

    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> _MockSystemOneResponse:
        del timeout
        requested = model or MOCK_JEV_MODEL
        if "agent_profile" in questions:
            text = str(dict(state).get("first_customer_message", "")).lower()
            has_return = _contains_any(text, self._RETURN_PHRASES)
            has_order = _contains_any(text, self._ORDER_PHRASES)
            if has_return and has_order:
                choice = "generalist"
                probabilities = {
                    "order_management": 0.21,
                    "returns_resolution": 0.21,
                    "generalist": 0.58,
                }
            elif has_return:
                choice = "returns_resolution"
                probabilities = {
                    "order_management": 0.04,
                    "returns_resolution": 0.89,
                    "generalist": 0.07,
                }
            elif has_order:
                choice = "order_management"
                probabilities = {
                    "order_management": 0.89,
                    "returns_resolution": 0.04,
                    "generalist": 0.07,
                }
            else:
                choice = "generalist"
                probabilities = {
                    "order_management": 0.09,
                    "returns_resolution": 0.09,
                    "generalist": 0.82,
                }
            answers: dict[str, _MockChoiceAnswer | _MockNoulAnswer] = {
                "agent_profile": _MockChoiceAnswer(
                    choice=choice,
                    confidence=probabilities[choice],
                    probabilities=probabilities,
                )
            }
        else:
            payload = dict(state)
            turn = dict(payload.get("current_turn", {}))
            user = str(turn.get("user_message", "")).lower()
            assistant = str(turn.get("assistant_message", "")).lower()
            recent = payload.get("recent_visible_messages", [])
            visible_user_text = " ".join(
                str(message.get("content", "")).lower()
                for message in recent
                if isinstance(message, Mapping) and message.get("role") == "user"
            )
            tools = turn.get("tool_events", [])
            supported_tool_event = any(
                isinstance(event, Mapping)
                and str(event.get("status", "")).lower()
                in {"executed", "prepared", "completed", "success"}
                for event in tools
            )
            explicit_assent = _contains_any(
                f"{visible_user_text} {user}", self._CONFIRMATION_PHRASES
            )
            asks_again = _contains_any(assistant, self._ASK_CONFIRMATION_PHRASES)
            redundant = 0.92 if explicit_assent and asks_again else 0.08

            user_topics = sum(
                (
                    _contains_any(user, self._RETURN_PHRASES),
                    _contains_any(user, self._ORDER_PHRASES),
                )
            )
            multi_part = user_topics > 1 or bool(
                re.search(r"\b(?:both|also|and then|as well as)\b", user)
            )
            assistant_topics = sum(
                (
                    _contains_any(assistant, self._RETURN_PHRASES),
                    _contains_any(assistant, self._ORDER_PHRASES),
                )
            )
            partial = 0.82 if multi_part and assistant_topics < user_topics else 0.11

            definitive = _contains_any(assistant, self._DEFINITIVE_PHRASES)
            unsupported = 0.86 if definitive and not supported_tool_event else 0.09
            answers = {
                "redundant_confirmation": _MockNoulAnswer(redundant),
                "partial_completion": _MockNoulAnswer(partial),
                "unsupported_definitive_answer": _MockNoulAnswer(unsupported),
            }
        return _MockSystemOneResponse(
            model=MOCK_JEV_MODEL if requested == DEFAULT_JEV_MODEL else requested,
            usage=_MockUsage(input_tokens=_mock_token_count(state)),
            answers=answers,
        )

    def close(self) -> None:
        return None


JevOperation = Literal["route", "assessment"]


class JevCallRunner(Protocol):
    def __call__(
        self,
        *,
        operation: JevOperation,
        state: JSONContent,
        questions: Mapping[str, Question],
        backend: JevBackend,
        requested_model: str,
        timeout_seconds: float,
        invoke: Callable[[], SystemOneResponse],
    ) -> SystemOneResponse: ...


def _direct_call_runner(
    *,
    operation: JevOperation,
    state: JSONContent,
    questions: Mapping[str, Question],
    backend: JevBackend,
    requested_model: str,
    timeout_seconds: float,
    invoke: Callable[[], SystemOneResponse],
) -> SystemOneResponse:
    del operation, state, questions, backend, requested_model, timeout_seconds
    return invoke()


def _route_questions() -> dict[str, Question]:
    spec = _ROUTE_QUESTION_SPEC["agent_profile"]
    return {
        "agent_profile": Choice(
            instructions=spec["instructions"],
            criteria=dict(spec["criteria"]),
        )
    }


def _assessment_questions() -> dict[str, Question]:
    return {
        name: Noul(
            instructions=spec["instructions"],
            criteria=NoulCriteria(**spec["criteria"]),
        )
        for name, spec in _ASSESSMENT_QUESTION_SPEC.items()
    }


def _usage(response: SystemOneResponse) -> JevUsage:
    usage = response.usage
    return JevUsage(
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
    )


def _failure_reason(prefix: str, exc: BaseException) -> str:
    # Provider errors can carry request bodies. Keep public failure reasons
    # bounded to a stable exception type and never surface the raw message.
    return f"{prefix}:{type(exc).__name__}"


class JevDecisionEngine:
    """Optional typed decision layer around the retail support agent.

    Jev may select an already-reviewed agent profile, but it never grants tool
    access or authorizes a mutation. Tool boundaries still come from the sealed
    profile, and the existing confirmation controller remains authoritative.
    Post-turn assessments are always observational.
    """

    def __init__(
        self,
        config: JevConfig | None = None,
        *,
        client: SystemOneClient | None = None,
        call_runner: JevCallRunner = _direct_call_runner,
    ) -> None:
        self.config = config or JevConfig()
        self._client = client
        self._owns_client = False
        self._client_initialization_error: str | None = None
        self._call_runner = call_runner
        self._lock = threading.Lock()

    def public_config(self) -> dict[str, Any]:
        return {
            **self.config.model_dump(mode="json"),
            "route_question_set_id": ROUTE_QUESTION_SET_ID,
            "route_question_set_sha256": ROUTE_QUESTION_SET_SHA256,
            "assessment_question_set_id": ASSESSMENT_QUESTION_SET_ID,
            "assessment_question_set_sha256": ASSESSMENT_QUESTION_SET_SHA256,
        }

    def _get_client(self) -> SystemOneClient:
        if self._client is not None:
            return self._client
        if self._client_initialization_error is not None:
            raise RuntimeError(self._client_initialization_error)
        try:
            if self.config.backend == JevBackend.MOCK:
                self._client = MockJevClient()
            else:
                self._client = TypeSafeClient(
                    model=self.config.requested_model,
                    timeout=self.config.timeout_seconds,
                )
        except Exception as exc:
            self._client_initialization_error = _failure_reason(
                "client_initialization_failed",
                exc,
            )
            raise RuntimeError(self._client_initialization_error) from exc
        self._owns_client = True
        return self._client

    def _evaluate(
        self,
        *,
        operation: JevOperation,
        state: JSONContent,
        questions: Mapping[str, Question],
        call_runner: JevCallRunner | None = None,
    ) -> SystemOneResponse:
        runner = call_runner or self._call_runner

        # A single shared synchronous SDK client may be used by concurrent web
        # sessions. Serialize client initialization and the short System One
        # call, not the surrounding agent work.
        with self._lock:
            client = self._get_client()

            def invoke() -> SystemOneResponse:
                return client.system_one(
                    state=state,
                    questions=questions,
                    model=self.config.requested_model,
                    timeout=self.config.timeout_seconds,
                )

            return runner(
                operation=operation,
                state=state,
                questions=questions,
                backend=self.config.backend,
                requested_model=self.config.requested_model,
                timeout_seconds=self.config.timeout_seconds,
                invoke=invoke,
            )

    def route(
        self,
        first_customer_message: str,
        deterministic_profile_id: str,
        *,
        call_runner: JevCallRunner | None = None,
    ) -> JevRoutingDecision:
        if deterministic_profile_id not in _ROLE_TO_PROFILE.values():
            raise ValueError("deterministic profile is not a known retail agent")
        if self.config.mode == JevMode.OFF:
            return JevRoutingDecision(
                mode=self.config.mode,
                backend=self.config.backend,
                selected_profile_id=deterministic_profile_id,
                applied_profile_id=deterministic_profile_id,
                requested_model=self.config.requested_model,
                duration_ms=0,
                source=JevSource.DETERMINISTIC,
            )

        state: JSONContent = {
            "first_customer_message": first_customer_message,
        }
        started = time.perf_counter()
        try:
            response = self._evaluate(
                operation="route",
                state=state,
                questions=_route_questions(),
                call_runner=call_runner,
            )
            answer = response.choices["agent_profile"]
            selected = _ROLE_TO_PROFILE.get(answer.choice)
            if selected is None:
                raise ValueError("unknown_route_choice")
            confidence = float(answer.confidence)
            probabilities = {
                str(name): float(probability)
                for name, probability in answer.probabilities.items()
            }
            applied = deterministic_profile_id
            source = (
                JevSource.MOCK
                if self.config.backend == JevBackend.MOCK
                else JevSource.JEV
            )
            fallback_reason = None
            if self.config.mode == JevMode.ROUTE:
                if confidence >= self.config.route_confidence_threshold:
                    applied = selected
                else:
                    source = JevSource.DETERMINISTIC_FALLBACK
                    fallback_reason = "confidence_below_threshold"
            return JevRoutingDecision(
                mode=self.config.mode,
                backend=self.config.backend,
                selected_profile_id=selected,
                applied_profile_id=applied,
                confidence=confidence,
                probabilities=probabilities,
                requested_model=self.config.requested_model,
                returned_model=response.model,
                usage=_usage(response),
                duration_ms=(time.perf_counter() - started) * 1_000,
                source=source,
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            return JevRoutingDecision(
                mode=self.config.mode,
                backend=self.config.backend,
                selected_profile_id=deterministic_profile_id,
                applied_profile_id=deterministic_profile_id,
                requested_model=self.config.requested_model,
                duration_ms=(time.perf_counter() - started) * 1_000,
                source=JevSource.DETERMINISTIC_FALLBACK,
                fallback_reason=_failure_reason("route_failed", exc),
            )

    def assess_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        recent_messages: Sequence[Mapping[str, str]],
        tool_events: Sequence[Mapping[str, Any]],
        call_runner: JevCallRunner | None = None,
    ) -> JevTurnAssessment | None:
        if self.config.mode == JevMode.OFF:
            return None
        state: JSONContent = {
            "current_turn": {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_events": [dict(event) for event in tool_events],
            },
            "recent_visible_messages": [
                {"role": message["role"], "content": message["content"]}
                for message in recent_messages[-8:]
            ],
        }
        started = time.perf_counter()
        try:
            response = self._evaluate(
                operation="assessment",
                state=state,
                questions=_assessment_questions(),
                call_runner=call_runner,
            )
            probabilities = {
                name: float(response.nouls[name].noul)
                for name in sorted(_ASSESSMENT_KEYS)
            }
            return JevTurnAssessment(
                backend=self.config.backend,
                probabilities=probabilities,
                requested_model=self.config.requested_model,
                returned_model=response.model,
                usage=_usage(response),
                duration_ms=(time.perf_counter() - started) * 1_000,
                source=(
                    JevSource.MOCK
                    if self.config.backend == JevBackend.MOCK
                    else JevSource.JEV
                ),
            )
        except Exception as exc:
            return JevTurnAssessment(
                backend=self.config.backend,
                requested_model=self.config.requested_model,
                duration_ms=(time.perf_counter() - started) * 1_000,
                source=JevSource.UNAVAILABLE,
                fallback_reason=_failure_reason("assessment_failed", exc),
            )

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            if client is not None and self._owns_client:
                client.close()
            self._owns_client = False
