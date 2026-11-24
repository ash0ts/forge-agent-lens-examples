from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .canonical import digest_value
from .constants import WRITE_TOOLS

_SCHEMA_VERSION = 1
_PUBLIC_PREPARATION_RESULT = {
    "status": "action_prepared",
    "message": (
        "No state changed. Explain the exact proposed action and its consequences, "
        "then ask the customer for a new, explicit confirmation. After an "
        "unqualified confirmation, repeat this exact tool call once."
    ),
}
_TERMINAL_PUNCTUATION = re.compile(r"[.!]+$")
_UNQUALIFIED_ASSENTS = frozenset(
    {
        "absolutely",
        "confirmed",
        "go ahead",
        "go for it",
        "i confirm",
        "ok",
        "okay",
        "okay please",
        "please do",
        "please do it",
        "please go ahead",
        "please proceed",
        "proceed",
        "sounds good",
        "sure",
        "sure please",
        "that is right",
        "that works",
        "that's right",
        "yes",
        "yes cancel that one",
        "yes do it",
        "yes go ahead",
        "yes i confirm",
        "yes please",
        "yes please go ahead",
        "yes please proceed",
        "yes proceed",
    }
)


class MutationConfirmationReceipt(BaseModel):
    """Private audit state for one exact proposed mutation.

    It intentionally stores message digests and indexes rather than message
    content. The business arguments remain in the frozen tool call; their exact
    tool-and-argument binding is ``action_sha256``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = _SCHEMA_VERSION
    conversation_id: str
    sequence: int
    intent_id: str
    tool_name: str
    action_sha256: str
    staged_tool_call_id: str
    staged_message_index: int
    state: Literal["pending", "approved", "invalidated", "superseded", "consumed"]
    proposal_message_index: int | None = None
    proposal_message_sha256: str | None = None
    decision_message_index: int | None = None
    decision_message_sha256: str | None = None
    invalidation_reason: (
        Literal[
            "non_assent",
            "missing_proposal",
            "argument_change",
            "post_confirmation_assistant_message",
            "post_confirmation_user_message",
            "proposal_mismatch",
        ]
        | None
    ) = None
    consumed_tool_call_id: str | None = None
    consumed_message_index: int | None = None
    replaced_by_intent_id: str | None = None


class WriteGateResult(BaseModel):
    """Result returned to the runner for one attempted tool call.

    ``model_dump()`` is safe to use in public telemetry: exact intent identity,
    action digests, and private receipts are excluded by default.
    """

    model_config = ConfigDict(extra="forbid")

    authorized: bool
    decision: Literal[
        "not_required", "staged", "awaiting_confirmation", "restaged", "consumed"
    ]
    public_result: dict[str, str] | None = None
    intent_id: str | None = Field(default=None, exclude=True)
    action_sha256: str | None = Field(default=None, exclude=True)
    receipt: MutationConfirmationReceipt | None = Field(default=None, exclude=True)


class UserConfirmationResult(BaseModel):
    """Private result of observing one later customer message."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["no_pending", "approved", "invalidated"]
    intent_id: str | None = Field(default=None, exclude=True)
    action_sha256: str | None = Field(default=None, exclude=True)
    receipt: MutationConfirmationReceipt | None = Field(default=None, exclude=True)


@dataclass
class _IntentState:
    receipt: MutationConfirmationReceipt
    arguments: dict[str, Any]


_ACTION_TERMS: dict[str, tuple[str, ...]] = {
    "cancel_pending_order": ("cancel", "cancellation"),
    "exchange_delivered_order_items": ("exchange", "replace", "swap"),
    "modify_pending_order_address": ("address", "ship"),
    "modify_pending_order_items": (
        "change",
        "modification",
        "modify",
        "replace",
        "swap",
    ),
    "modify_pending_order_payment": ("payment", "pay with", "switch to"),
    "modify_user_address": ("address",),
    "return_delivered_order_items": ("return", "send back"),
}
_ORDER_SCOPED_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)
_ITEM_SCOPED_WRITE_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "return_delivered_order_items",
    }
)
_PAYMENT_SCOPED_WRITE_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)
_ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")
_DOMESTIC_COUNTRY_NAMES = frozenset(
    {
        "us",
        "u.s.",
        "usa",
        "u.s.a.",
        "united states",
        "united states of america",
    }
)
_US_STATE_CODES = frozenset(
    {
        "ak",
        "al",
        "ar",
        "as",
        "az",
        "ca",
        "co",
        "ct",
        "dc",
        "de",
        "fl",
        "fm",
        "ga",
        "gu",
        "hi",
        "ia",
        "id",
        "il",
        "in",
        "ks",
        "ky",
        "la",
        "ma",
        "md",
        "me",
        "mh",
        "mi",
        "mn",
        "mo",
        "mp",
        "ms",
        "mt",
        "nc",
        "nd",
        "ne",
        "nh",
        "nj",
        "nm",
        "nv",
        "ny",
        "oh",
        "ok",
        "or",
        "pa",
        "pr",
        "pw",
        "ri",
        "sc",
        "sd",
        "tn",
        "tx",
        "ut",
        "va",
        "vi",
        "vt",
        "wa",
        "wi",
        "wv",
        "wy",
    }
)
_US_ZIP = re.compile(r"^\d{5}(?:-\d{4})?$")
_ITEM_IDENTIFIER = re.compile(r"\bitem[_-][a-z0-9_-]+\b", re.IGNORECASE)
_REASON_TERMS: dict[str, tuple[str, ...]] = {
    "no longer needed": (
        "no longer need",
        "no longer needed",
        "not needed anymore",
    ),
    "ordered by mistake": (
        "accidental order",
        "accidentally ordered",
        "ordered by mistake",
        "ordering mistake",
    ),
}


_PARAGRAPH_OR_LIST_BREAK = re.compile(
    r"(?<=[A-Za-z0-9])[ \t]*\r?\n(?:[ \t]*\r?\n+|[ \t]*(?=-[- \t]|[*•][ \t]|\d+[.)][ \t]))"
)


def _normalize_visible_text(
    value: Any, *, preserve_paragraph_breaks: bool = False
) -> str:
    raw = str(value)
    if preserve_paragraph_breaks:
        raw = _PARAGRAPH_OR_LIST_BREAK.sub(". ", raw)
    normalized = unicodedata.normalize("NFKC", raw).casefold()
    return " ".join(normalized.replace("’", "'").split())


def _contains_visible_value(normalized_text: str, value: Any) -> bool:
    return _normalize_visible_text(value) in normalized_text


def _contains_address_value(normalized_text: str, value: Any) -> bool:
    """Match a visible address field without accepting substrings in prose."""

    normalized_value = _normalize_visible_text(value)
    if not normalized_value:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])",
            normalized_text,
        )
        is not None
    )


def _domestic_state_zip_matches(
    normalized_text: str, *, state: Any, zip_code: Any
) -> tuple[re.Match[str], ...]:
    normalized_state = _normalize_visible_text(state)
    normalized_zip = _normalize_visible_text(zip_code)
    if normalized_state not in _US_STATE_CODES or not _US_ZIP.fullmatch(
        normalized_zip
    ):
        return ()
    return tuple(
        re.finditer(
            rf"(?<![a-z0-9]){re.escape(normalized_state)}(?![a-z0-9])"
            rf"[\s,.-]*"
            rf"(?<![a-z0-9]){re.escape(normalized_zip)}(?![a-z0-9])",
            normalized_text,
        )
    )


def _domestic_state_zip_is_unambiguous(
    normalized_text: str, *, state: Any, zip_code: Any
) -> bool:
    """Accept only a terminal US state/ZIP, optionally followed by a US name."""

    matches = _domestic_state_zip_matches(
        normalized_text, state=state, zip_code=zip_code
    )
    if not matches:
        return False
    for match in matches:
        suffix = normalized_text[match.end() :].lstrip()
        if not suffix or (not suffix[0].isalnum() and suffix[0] != ","):
            continue
        if suffix.startswith(","):
            suffix = suffix[1:].lstrip()
        if not suffix:
            continue
        accepted_country = False
        for country in sorted(
            _DOMESTIC_COUNTRY_NAMES, key=lambda value: (-len(value), value)
        ):
            country_match = re.match(
                rf"{re.escape(country)}(?![a-z0-9])", suffix
            )
            if country_match is None:
                continue
            remainder = suffix[country_match.end() :].lstrip()
            if not remainder or not remainder[0].isalnum():
                accepted_country = True
            break
        if not accepted_country:
            return False
    return True


def _matches_address(normalized_text: str, arguments: Mapping[str, Any]) -> bool:
    """Require the human address, never an opaque account identifier.

    A complete US street address is unambiguous without the speaker saying
    ``USA`` aloud.  International destinations still require the country so a
    confirmation cannot silently cross a national boundary.
    """

    nested = arguments.get("address")
    address = nested if isinstance(nested, Mapping) else arguments
    country = address.get("country")
    country_is_implicit = (
        _normalize_visible_text(country) in _DOMESTIC_COUNTRY_NAMES
    )
    required_fields = tuple(
        field
        for field in _ADDRESS_FIELDS
        if field not in ({"country", "state", "zip"} if country_is_implicit else set())
        and str(address.get(field) or "").strip()
    )
    if not required_fields or not all(
        _contains_address_value(normalized_text, address.get(field))
        for field in required_fields
    ):
        return False
    if not country_is_implicit:
        return True
    if not _domestic_state_zip_is_unambiguous(
        normalized_text,
        state=address.get("state"),
        zip_code=address.get("zip"),
    ):
        return False
    return True


def _matches_cancellation_reason(
    normalized_text: str, arguments: Mapping[str, Any]
) -> bool:
    reason = _normalize_visible_text(arguments.get("reason", ""))
    accepted = _REASON_TERMS.get(reason, (reason,))
    return bool(reason) and any(term in normalized_text for term in accepted)


def _payment_kind(payment_method_id: Any) -> str | None:
    normalized = _normalize_visible_text(payment_method_id)
    if normalized.startswith("paypal_"):
        return "paypal"
    if normalized.startswith("gift_card_"):
        return "gift_card"
    if normalized.startswith("credit_card_"):
        return "credit_card"
    return None


def _matches_payment_source(
    normalized_text: str, arguments: Mapping[str, Any]
) -> bool:
    """Match the customer-facing payment type without exposing its database ID."""

    payment_method_id = arguments.get("payment_method_id")
    kind = _payment_kind(payment_method_id)
    if kind == "paypal":
        return "paypal" in normalized_text
    if kind == "gift_card":
        return "gift card" in normalized_text or "store credit" in normalized_text
    if kind == "credit_card":
        # Customers and agents usually name the visible card brand or last four
        # digits, not the database's generic ``credit_card_*`` kind. The exact
        # payment method remains bound by the action digest; this matcher only
        # verifies that the proposal describes the correct public payment type.
        conflicting_source = any(
            term in normalized_text
            for term in ("paypal", "gift card", "store credit")
        )
        visible_card = any(
            term in normalized_text
            for term in (
                "credit card",
                "visa",
                "mastercard",
                "master card",
                "american express",
                "amex",
                "discover",
                "card ending",
                "card ending in",
            )
        )
        return visible_card and not conflicting_source
    # Unknown identifiers fail closed and must be stated exactly. The frozen
    # retail workload uses only the three human-readable kinds above.
    return bool(payment_method_id) and _contains_visible_value(
        normalized_text, payment_method_id
    )


def _does_not_name_wrong_item(
    normalized_text: str, arguments: Mapping[str, Any]
) -> bool:
    """Reject an explicit wrong item ID without requiring IDs in normal prose.

    The controller cannot infer that a phrase such as "the blue shirt" maps to
    a database item ID. It can, however, reject a proposal that explicitly
    names a different opaque item. Product-language accuracy remains reviewable
    in the visible conversation and exact execution remains digest-bound.
    """

    expected = {
        _normalize_visible_text(item)
        for key in ("item_ids", "new_item_ids")
        for item in arguments.get(key, [])
    }
    explicit = {
        _normalize_visible_text(match)
        for match in _ITEM_IDENTIFIER.findall(normalized_text)
        if _normalize_visible_text(match) != "item_ids"
    }
    return not explicit or explicit <= expected


def proposal_matches_action(
    text: str,
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    preserve_paragraph_breaks: bool = True,
) -> bool:
    """Check the visible action and its deterministically observable targets.

    This is deliberately mechanical, not an LLM judge. It prevents an
    unrelated assistant question followed by “yes” from authorizing a staged
    mutation. Customer-facing order numbers, addresses, payment types, and
    cancellation reasons are required where relevant. Internal item, payment,
    and user IDs are not required verbatim, although an explicitly named wrong
    item is rejected. Human-readable quality and policy completeness remain
    separate review concerns.
    """

    normalized = _normalize_visible_text(
        text, preserve_paragraph_breaks=preserve_paragraph_breaks
    )
    terms = _ACTION_TERMS.get(tool_name, ())
    if not terms or not any(term in normalized for term in terms):
        return False
    if tool_name in _ORDER_SCOPED_WRITE_TOOLS:
        order_id = arguments.get("order_id")
        if not order_id or not _contains_visible_value(normalized, order_id):
            return False
    if tool_name in {
        "modify_pending_order_address",
        "modify_user_address",
    } and not _matches_address(normalized, arguments):
        return False
    if tool_name == "cancel_pending_order" and not _matches_cancellation_reason(
        normalized, arguments
    ):
        return False
    if tool_name in _ITEM_SCOPED_WRITE_TOOLS and not _does_not_name_wrong_item(
        normalized, arguments
    ):
        return False
    if (
        tool_name in _PAYMENT_SCOPED_WRITE_TOOLS
        and arguments.get("payment_method_id")
        and not _matches_payment_source(normalized, arguments)
    ):
        return False
    return True


def _canonical_json(value: Any, *, path: str = "arguments") -> Any:
    """Return a strict JSON value suitable for exact action identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string key")
        result: dict[str, Any] = {}
        for key in sorted(value):
            result[key] = _canonical_json(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def action_sha256(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the exact canonical identity of one state-changing action."""

    return digest_value(
        {
            "tool_name": tool_name,
            "arguments": _canonical_json(arguments),
        }
    )


def public_preparation_result() -> dict[str, str]:
    """Return the only public response allowed for an intercepted write."""

    return dict(_PUBLIC_PREPARATION_RESULT)


def confirmation_intent_id(
    *, conversation_id: str, sequence: int, action_digest: str
) -> str:
    """Return the canonical private identity for a staged mutation."""

    return digest_value(
        {
            "schema_version": _SCHEMA_VERSION,
            "conversation_id": conversation_id,
            "sequence": sequence,
            "action_sha256": action_digest,
        }
    )


def is_unqualified_assent(text: str) -> bool:
    """Recognize only a complete, unqualified affirmative message.

    Target names, conditions, corrections, and negations add tokens and are
    therefore rejected instead of being interpreted by substring matching.
    """

    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    normalized = normalized.replace("’", "'")
    if "?" in normalized or ";" in normalized or ":" in normalized:
        return False
    normalized = _TERMINAL_PUNCTUATION.sub("", normalized).strip()
    normalized = normalized.replace(",", " ")
    normalized = " ".join(normalized.split())
    return normalized in _UNQUALIFIED_ASSENTS


class MutationConfirmationController:
    """Fail-closed, one-shot confirmation state for one conversation.

    The caller must pass the exact same schema-validated business arguments to
    ``guard_write`` before execution. This controller binds and compares them;
    it does not replace MCP schema validation or account authorization.

    Integration order:

    1. Call ``guard_write`` for every tool attempt. The first write is staged
       and must not execute.
    2. Call ``observe_assistant_message`` when the agent later presents the
       proposal to the customer.
    3. Call ``observe_user_message`` for the customer's next visible reply.
    4. Retry ``guard_write`` with the exact same tool and arguments. An approved
       intent is consumed before ``authorized=True`` is returned.

    Intent IDs, action digests, and receipts are private control evidence. Do
    not append them to visible messages or Agent Lens telemetry.
    """

    def __init__(self, *, conversation_id: str) -> None:
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        self.conversation_id = conversation_id
        self._sequence = 0
        self._active: _IntentState | None = None
        self._history: list[_IntentState] = []

    def guard_write(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        tool_call_id: str,
        message_index: int,
    ) -> WriteGateResult:
        if tool_name not in WRITE_TOOLS:
            return WriteGateResult(
                authorized=True,
                decision="not_required",
            )
        if not tool_call_id:
            raise ValueError("write tool_call_id must be non-empty")
        if message_index < 0:
            raise ValueError("message_index must be non-negative")
        canonical_arguments = _canonical_json(arguments)
        if not isinstance(canonical_arguments, dict):  # pragma: no cover - Mapping
            raise ValueError("tool arguments must be an object")
        action_digest = action_sha256(tool_name, canonical_arguments)

        active = self._active
        if active is None:
            state = self._stage(
                tool_name=tool_name,
                action_sha256=action_digest,
                arguments=canonical_arguments,
                tool_call_id=tool_call_id,
                message_index=message_index,
            )
            return self._blocked_result("staged", state)

        if active.receipt.action_sha256 != action_digest:
            prior = active.receipt
            prior.state = "superseded"
            prior.invalidation_reason = "argument_change"
            state = self._stage(
                tool_name=tool_name,
                action_sha256=action_digest,
                arguments=canonical_arguments,
                tool_call_id=tool_call_id,
                message_index=message_index,
            )
            prior.replaced_by_intent_id = state.receipt.intent_id
            return self._blocked_result("restaged", state)

        if active.receipt.state == "approved":
            decision_index = active.receipt.decision_message_index
            if decision_index is None:  # pragma: no cover - state invariant
                raise RuntimeError("approved intent has no customer decision")
            if message_index <= decision_index:
                raise ValueError("write retry must follow the customer confirmation")
            if tool_call_id == active.receipt.staged_tool_call_id:
                raise ValueError("write retry must use a new tool call")
            active.receipt.state = "consumed"
            active.receipt.consumed_tool_call_id = tool_call_id
            active.receipt.consumed_message_index = message_index
            self._active = None
            return WriteGateResult(
                authorized=True,
                decision="consumed",
                public_result=None,
                intent_id=active.receipt.intent_id,
                action_sha256=active.receipt.action_sha256,
                receipt=active.receipt.model_copy(deep=True),
            )

        return self._blocked_result("awaiting_confirmation", active)

    def observe_assistant_message(self, *, text: str, message_index: int) -> None:
        """Bind the latest visible agent proposal to the pending intent."""

        active = self._active
        if active is None:
            return
        if message_index <= active.receipt.staged_message_index:
            raise ValueError("assistant proposal must follow the staged write")
        if not text.strip():
            raise ValueError("assistant proposal must be non-empty")
        if active.receipt.state == "approved":
            active.receipt.state = "invalidated"
            active.receipt.invalidation_reason = "post_confirmation_assistant_message"
            self._active = None
            return
        if active.receipt.state != "pending":
            return
        if not proposal_matches_action(
            text,
            tool_name=active.receipt.tool_name,
            arguments=active.arguments,
        ):
            active.receipt.state = "invalidated"
            active.receipt.invalidation_reason = "proposal_mismatch"
            self._active = None
            return
        active.receipt.proposal_message_index = message_index
        active.receipt.proposal_message_sha256 = digest_value(text)

    def observe_user_message(
        self,
        *,
        text: str,
        message_index: int,
    ) -> UserConfirmationResult:
        """Approve only a later whole-message assent to a visible proposal."""

        active = self._active
        if active is None:
            return UserConfirmationResult(decision="no_pending")
        receipt = active.receipt
        if message_index <= receipt.staged_message_index:
            raise ValueError("customer decision must follow the staged write")
        if receipt.state == "approved":
            receipt.state = "invalidated"
            receipt.invalidation_reason = "post_confirmation_user_message"
            self._active = None
            return UserConfirmationResult(
                decision="invalidated",
                intent_id=receipt.intent_id,
                action_sha256=receipt.action_sha256,
                receipt=receipt.model_copy(deep=True),
            )

        receipt.decision_message_index = message_index
        receipt.decision_message_sha256 = digest_value(text)
        proposal_index = receipt.proposal_message_index
        if proposal_index is None or proposal_index >= message_index:
            receipt.state = "invalidated"
            receipt.invalidation_reason = "missing_proposal"
            self._active = None
            return UserConfirmationResult(
                decision="invalidated",
                intent_id=receipt.intent_id,
                action_sha256=receipt.action_sha256,
                receipt=receipt.model_copy(deep=True),
            )
        if not is_unqualified_assent(text):
            receipt.state = "invalidated"
            receipt.invalidation_reason = "non_assent"
            self._active = None
            return UserConfirmationResult(
                decision="invalidated",
                intent_id=receipt.intent_id,
                action_sha256=receipt.action_sha256,
                receipt=receipt.model_copy(deep=True),
            )

        receipt.state = "approved"
        return UserConfirmationResult(
            decision="approved",
            intent_id=receipt.intent_id,
            action_sha256=receipt.action_sha256,
            receipt=receipt.model_copy(deep=True),
        )

    def private_receipts(self) -> list[MutationConfirmationReceipt]:
        """Return immutable copies for a host-only audit ledger."""

        return [state.receipt.model_copy(deep=True) for state in self._history]

    @property
    def mutation_confirmation_ok(self) -> bool | None:
        """Summarize executed writes without treating blocked attempts as failures."""

        consumed = [
            state.receipt
            for state in self._history
            if state.receipt.state == "consumed"
        ]
        return True if consumed else None

    @property
    def has_pending_intent(self) -> bool:
        return self._active is not None

    def _stage(
        self,
        *,
        tool_name: str,
        action_sha256: str,
        arguments: Mapping[str, Any],
        tool_call_id: str,
        message_index: int,
    ) -> _IntentState:
        self._sequence += 1
        intent_id = confirmation_intent_id(
            conversation_id=self.conversation_id,
            sequence=self._sequence,
            action_digest=action_sha256,
        )
        state = _IntentState(
            receipt=MutationConfirmationReceipt(
                conversation_id=self.conversation_id,
                sequence=self._sequence,
                intent_id=intent_id,
                tool_name=tool_name,
                action_sha256=action_sha256,
                staged_tool_call_id=tool_call_id,
                staged_message_index=message_index,
                state="pending",
            ),
            arguments=dict(_canonical_json(arguments)),
        )
        self._history.append(state)
        self._active = state
        return state

    @staticmethod
    def _blocked_result(
        decision: Literal["staged", "awaiting_confirmation", "restaged"],
        state: _IntentState,
    ) -> WriteGateResult:
        return WriteGateResult(
            authorized=False,
            decision=decision,
            public_result=public_preparation_result(),
            intent_id=state.receipt.intent_id,
            action_sha256=state.receipt.action_sha256,
            receipt=state.receipt.model_copy(deep=True),
        )
