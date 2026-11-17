from __future__ import annotations

from typing import Any, Mapping

from .schemas import ConversationBundle, ModelCall

_RETRY_AMBIGUITY_PREFIX = "retryable transport outcome was ambiguous:"
_EXHAUSTED_TRANSPORT_PREFIX = "Forge Inference request unavailable after"


def continues_frozen_attempt_failures(preview: Mapping[str, Any]) -> bool:
    """Whether an exact preview allows terminal frozen failures to coexist."""

    policy = preview.get("run_policy")
    return (
        isinstance(policy, dict)
        and policy.get("frozen_attempt_failure_policy")
        == "continue_when_bundle_is_frozen"
        and policy.get("bounded_retry_cost_ambiguity")
        == "track_upper_bound_and_continue"
    )


def is_allowed_retry_cost_ambiguity(call: ModelCall) -> bool:
    """Recognize only bounded retry ambiguity with a frozen usable outcome.

    A successful response after a retry may be used when its identity and
    finish reason validate. An exhausted retry sequence may be retained as an
    unavailable coordinate. Missing usage, malformed responses, and model
    identity drift remain disallowed.
    """

    if call.ambiguous_cost_reserve_usd <= 0:
        return False
    if not call.cost_uncertainty_reasons or any(
        not reason.startswith(_RETRY_AMBIGUITY_PREFIX)
        for reason in call.cost_uncertainty_reasons
    ):
        return False
    if call.response_validation_error is None:
        return (
            call.returned_model == call.requested_model
            and call.finish_reason in {"stop", "tool_calls"}
        )
    return (
        call.finish_reason == "transport_error"
        and call.returned_model == ""
        and call.response_validation_error.startswith(
            _EXHAUSTED_TRANSPORT_PREFIX
        )
    )


def bundle_has_disallowed_model_evidence(bundle: ConversationBundle) -> bool:
    """Return true when a frozen bundle exceeds the test-project policy."""

    if bundle.status == "unavailable":
        # The unavailable bundle is the terminal evidence for this coordinate;
        # none of its model output is used as a behavioral result or emitted to
        # Agent Lens. Keep it for accounting and diagnosis, then continue the
        # large test-project cohort. Candidate identity is enforced on every
        # completed trajectory that can enter the public projection.
        return False

    for call in bundle.model_calls:
        has_ambiguity = bool(
            call.ambiguous_cost_reserve_usd
            or call.cost_uncertainty_reasons
        )
        if has_ambiguity and not is_allowed_retry_cost_ambiguity(call):
            return True
        if (
            call.response_validation_error
            and not is_allowed_retry_cost_ambiguity(call)
        ):
            return True
    return False
