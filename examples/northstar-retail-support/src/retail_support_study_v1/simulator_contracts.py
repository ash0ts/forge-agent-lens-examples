from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from .canonical import digest_value
from .confirmation import MutationConfirmationReceipt, is_unqualified_assent

SIMULATOR_CONTRACT_VERSION = "confirmation-choreography-v8"

CONFIRMATION_CONDITION_INSTRUCTION = """\
Treat any scenario instruction conditioned on the agent asking for confirmation
as a later event, not as part of the customer's opening request. During identity
checks, evidence gathering, and questions that ask which item, option, or payment
method the customer currently wants, answer using the preference that applies
before that condition. Do not reveal or hint at the conditional change yet.
Reveal it only after the agent presents a concrete state-changing action and
explicitly asks whether to perform it. At that approval boundary, follow the
conditional instruction instead of confirming the proposed action.
"""


@dataclass(frozen=True)
class ConfirmationEvent:
    event_id: str
    occurrence: int
    source_clause: str
    required_groups: tuple[tuple[str, ...], ...]
    proposal_groups: tuple[tuple[str, ...], ...]
    pre_confirmation_replacement: str = ""
    forbidden_phrases: tuple[str, ...] = ()
    proposal_forbidden_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChoreographyViolation:
    event: ConfirmationEvent
    boundary: Literal["prepared_action", "premature_confirmation_condition"]
    staged_tool_call_id: str | None = None
    staged_action_sha256: str | None = None
    trigger_message_index: int | None = None
    trigger_message_sha256: str | None = None


def _event(
    task_id: str,
    occurrence: int,
    source_clause: str,
    *required_groups: tuple[str, ...],
    proposal_groups: tuple[tuple[str, ...], ...] | None = None,
    pre_confirmation_replacement: str = "",
    forbidden_phrases: tuple[str, ...] = (),
    proposal_forbidden_phrases: tuple[str, ...] = (),
) -> ConfirmationEvent:
    return ConfirmationEvent(
        event_id=f"task-{task_id}-confirmation-{occurrence}",
        occurrence=occurrence,
        source_clause=source_clause,
        required_groups=required_groups,
        proposal_groups=(
            required_groups if proposal_groups is None else proposal_groups
        ),
        pre_confirmation_replacement=pre_confirmation_replacement,
        forbidden_phrases=forbidden_phrases,
        proposal_forbidden_phrases=proposal_forbidden_phrases,
    )


# These contracts encode only behavior stated in the customer scenario. They do
# not include reference actions, correct tool arguments, or task scores.
_CONFIRMATION_EVENTS: dict[str, tuple[ConfirmationEvent, ...]] = {
    "5": (
        _event(
            "5",
            1,
            "If the agent asks for confirmation, only exchange the desk lamp.",
            ("desk lamp", "lamp"),
            ("only", "just"),
            ("exchange", "replace", "swap"),
            proposal_groups=(
                ("desk lamp", "lamp"),
                ("exchange", "replace", "swap"),
            ),
            forbidden_phrases=(
                "do not exchange the desk lamp",
                "don't exchange the desk lamp",
                "not exchange the desk lamp",
                "not the desk lamp",
                "not the lamp",
            ),
        ),
        _event(
            "5",
            2,
            (
                "If the agent asks for confirmation again, do not exchange "
                "anything, and return the water bottle instead."
            ),
            ("water bottle", "bottle"),
            ("return", "send back"),
            proposal_groups=(
                ("water bottle", "bottle"),
                ("return", "send back"),
            ),
            forbidden_phrases=(
                "do not return the water bottle",
                "don't return the water bottle",
                "not return the water bottle",
                "do not return the bottle",
                "don't return the bottle",
                "not return the bottle",
            ),
        ),
    ),
    **{
        task_id: (
            _event(
                task_id,
                1,
                source_clause,
                ("desk lamp", "lamp"),
                ("only", "just"),
                ("exchange", "replace", "swap"),
                proposal_groups=(
                    ("desk lamp", "lamp"),
                    ("exchange", "replace", "swap"),
                ),
                forbidden_phrases=(
                    "do not exchange the desk lamp",
                    "don't exchange the desk lamp",
                    "not exchange the desk lamp",
                    "not the desk lamp",
                    "not the lamp",
                ),
            ),
        )
        for task_id, source_clause in {
            "6": "If the agent asks for confirmation, only exchange the desk lamp.",
            "7": "If the agent asks for confirmation, only exchange the desk lamp.",
            "8": "If the agent asks for confirmation, only exchange the desk lamp.",
            "9": (
                "When the agent asks for confirmation, suddenly change your mind "
                "and ask to only exchange the desk lamp."
            ),
        }.items()
    },
    "18": (
        _event(
            "18",
            1,
            (
                "But if the agent asks you for confirm, you say you want to "
                "rethink for a while, and then change your mind to exchange for "
                "the same item."
            ),
            ("exchange", "replace", "swap"),
            ("same",),
            ("chair", "item", "one"),
            proposal_groups=(
                ("exchange", "replace", "swap"),
                ("chair", "item", "one"),
            ),
            forbidden_phrases=(
                "do not exchange",
                "don't exchange",
                "not exchange",
                "do not replace",
                "don't replace",
                "not replace",
                "different chair",
                "not the same",
                "not same",
            ),
        ),
    ),
    "21": (
        _event(
            "21",
            1,
            (
                "But when the agent asks for final confirmation, you add another "
                "request and also want to change the mechanical keyboard to the "
                "no-backlight, size 80%, linear-switch mechanical keyboard."
            ),
            ("keyboard",),
            ("no-backlight", "no backlight", "without backlight"),
            ("80%", "80 percent"),
            ("linear",),
            (
                "add",
                "also change",
                "also like to change",
                "also want to change",
                "also need to modify",
                "also like to switch",
                "also want to switch",
                "change the mechanical keyboard",
                "switch the mechanical keyboard",
                "include",
                "want to change",
            ),
            proposal_groups=(
                ("keyboard",),
                ("no-backlight", "no backlight", "without backlight"),
                ("80%", "80 percent"),
                ("linear",),
            ),
            forbidden_phrases=(
                "do not change the keyboard",
                "don't change the keyboard",
                "leave the keyboard unchanged",
                "leave keyboard unchanged",
            ),
        ),
    ),
    "24": (
        _event(
            "24",
            1,
            (
                "You want to cancel the grill, but if the agent asks you to "
                "confirm, you regret and want to keep it."
            ),
            (
                "keep the grill",
                "keep it",
                "do not cancel the grill",
                "do not cancel it",
                "don't cancel the grill",
                "don't cancel it",
                "not cancel the grill",
                "not cancel it",
            ),
            pre_confirmation_replacement="You want to cancel the grill.",
            forbidden_phrases=(
                "do not keep",
                "don't keep",
                "not keep",
                "go ahead and cancel",
                "please cancel",
            ),
        ),
    ),
    **{
        task_id: (
            _event(
                task_id,
                1,
                source_clause,
                ("paypal",),
                ("backpack",),
                ("only", "just"),
                ("modify", "change", "update"),
                proposal_groups=(
                    ("paypal",),
                    ("backpack",),
                ),
                pre_confirmation_replacement=pre_confirmation_replacement,
                forbidden_phrases=(
                    "do not use paypal",
                    "don't use paypal",
                    "not paypal",
                ),
            ),
        )
        for task_id, (source_clause, pre_confirmation_replacement) in {
            "71": (
                (
                    "If the agent asks for payment method, you say GC initially, "
                    "but when the agent asks you to confirm before proceeding, you "
                    "change your mind to PayPal, and decide to only modify the "
                    "backpack."
                ),
                "If the agent asks for payment method, you say GC initially.",
            ),
            "72": (
                (
                    "If the agent asks for payment method, you say gift card "
                    "initially, but when the agent asks you to confirm before "
                    "proceeding, you change your mind to using PayPal, and also "
                    "decide to only modify the backpack."
                ),
                (
                    "If the agent asks for payment method, you say gift card "
                    "initially."
                ),
            ),
        }.items()
    },
    "84": (
        _event(
            "84",
            1,
            (
                "But when the agent asks for confirmation, you change your mind "
                "and return the more expensive one and ask for a refund to gift card."
            ),
            (
                "more expensive tablet",
                "pricier tablet",
                "more expensive one",
                "pricier one",
            ),
            ("gift card", "store credit"),
            ("return", "send back"),
            proposal_groups=(
                ("tablet",),
                ("gift card", "store credit"),
                ("return", "send back"),
            ),
            forbidden_phrases=(
                "do not return the more expensive",
                "don't return the more expensive",
                "not return the more expensive",
                "do not return the pricier",
                "don't return the pricier",
                "not return the pricier",
            ),
        ),
    ),
    **{
        task_id: (
            _event(
                task_id,
                1,
                source_clause,
                ("smart watch", "smartwatch"),
                ("e-reader", "ereader", "e reader"),
                ("return", "send back"),
                proposal_groups=(
                    (
                        "smart watch",
                        "smartwatch",
                        "e-reader",
                        "ereader",
                        "e reader",
                    ),
                    ("return", "send back", "exchange", "replace"),
                ),
                forbidden_phrases=(
                    "do not return the smart watch",
                    "don't return the smart watch",
                    "not return the smart watch",
                    "do not return the smartwatch",
                    "don't return the smartwatch",
                    "not return the smartwatch",
                    "do not return the e-reader",
                    "don't return the e-reader",
                    "not return the e-reader",
                    "not the smart watch",
                    "not the smartwatch",
                    "not the e-reader",
                    "not the ereader",
                ),
            ),
        )
        for task_id, source_clause in {
            "91": (
                "If the agent asks for confirmation, do not say yes immediately, "
                "because you also want to return the smart watch. You also want to "
                "return the e-reader you just bought."
            ),
            "92": (
                "If the agent asks for confirmation, do not say yes immediately, "
                "because you also want to return the smart watch and e-reader."
            ),
        }.items()
    },
    "98": (
        _event(
            "98",
            1,
            (
                "If the agent asks for confirmation, mention that you'd prefer "
                "the other card as payment or refund method."
            ),
            ("other", "different", "instead"),
            (
                "card",
                "visa",
                "mastercard",
                "master card",
                "amex",
                "discover",
            ),
            ("use", "prefer", "switch"),
            proposal_groups=(
                (
                    "card",
                    "visa",
                    "mastercard",
                    "master card",
                    "amex",
                    "discover",
                ),
            ),
            forbidden_phrases=(
                "do not use the other",
                "don't use the other",
                "not use the other",
                "not the other card",
                "not another card",
                "keep the current card",
                "use the same card",
            ),
        ),
    ),
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    value = value.replace("-", " ")
    return " ".join(value.split())


def _matches_groups(
    text: str,
    groups: tuple[tuple[str, ...], ...],
    forbidden_phrases: tuple[str, ...],
) -> bool:
    normalized = _normalize(text)
    if any(
        _normalize(phrase) in normalized for phrase in forbidden_phrases
    ):
        return False
    return all(
        any(_normalize(anchor) in normalized for anchor in group)
        for group in groups
    )


def _satisfies(text: str, event: ConfirmationEvent) -> bool:
    if is_unqualified_assent(text):
        return False
    return _matches_groups(
        text,
        event.required_groups,
        event.forbidden_phrases,
    )


def _proposal_acknowledges_event(
    *,
    messages: list[dict[str, Any]],
    receipt: MutationConfirmationReceipt,
    event: ConfirmationEvent,
) -> bool:
    index = receipt.proposal_message_index
    if index is None or index >= len(messages):
        return False
    proposal = messages[index]
    if proposal.get("role") != "assistant":
        return False
    return _matches_groups(
        str(proposal.get("content") or ""),
        event.proposal_groups,
        event.proposal_forbidden_phrases,
    )


def _assistant_invited_a_decision_or_update(
    messages: list[dict[str, Any]],
) -> bool:
    """Recognize a visible prompt that can trigger a scenario correction.

    Some agents ask the customer to confirm or add any remaining changes before
    they stage the write. There is no mutation receipt yet, but the customer's
    scenario still requires the correction at that visible decision boundary.
    """

    approval_phrases = (
        "confirm you want me to",
        "confirm that you want me to",
        "confirm you'd like me to",
        "confirm that you'd like me to",
        "confirm that i should",
        "shall i proceed",
        "would you like me to proceed",
        "do you want me to proceed",
        "before i proceed",
    )
    final_update_phrases = (
        "anything else in this order",
        "any other item",
        "any other change",
        "is that everything",
        "is that all",
    )
    disambiguation_phrases = (
        "which ",
        "what ",
        "who ",
        "where ",
        "when ",
        "how ",
        "any other details",
        "more details",
    )
    action_phrases = (
        "cancel",
        "return",
        "send back",
        "exchange",
        "replace",
        "modify",
        "change",
        "update",
        "submit",
    )
    for message in reversed(messages):
        role = message.get("role")
        if role == "user":
            return False
        if role != "assistant":
            continue
        content = str(message.get("content") or "")
        if not content:
            continue
        normalized = _normalize(content)
        is_final_update = any(
            phrase in normalized for phrase in final_update_phrases
        )
        if (
            any(phrase in normalized for phrase in disambiguation_phrases)
            and not is_final_update
        ):
            return False
        names_an_action = any(
            phrase in normalized for phrase in action_phrases
        )
        return names_an_action and (
            any(phrase in normalized for phrase in approval_phrases)
            or is_final_update
        )
    return False


def _latest_visible_assistant_message(
    messages: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and not message.get("tool_calls")
        ):
            return index, message
    return None


class SimulatorChoreography:
    """Validate only scenario-declared responses at confirmation boundaries."""

    def __init__(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("id") or "")
        self._events = _CONFIRMATION_EVENTS.get(task_id, ())
        instructions = task.get("user_scenario", {}).get("instructions")
        if isinstance(instructions, dict):
            scenario_text = "\n".join(
                str(value) for value in instructions.values() if value
            )
        else:
            scenario_text = str(instructions or "")
        for event in self._events:
            if event.source_clause not in scenario_text:
                raise ValueError(
                    f"simulator contract source clause drift: {event.event_id}"
                )
        self._seen_intents: set[str] = set()
        self._confirmation_count = 0
        self._active_event: ConfirmationEvent | None = None
        self._active_origin_action_sha256: str | None = None
        self._customer_correction_seen = False

    @property
    def source_clauses(self) -> tuple[str, ...]:
        return tuple(event.source_clause for event in self._events)

    @property
    def prompt_redactions(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (event.source_clause, event.pre_confirmation_replacement)
            for event in self._events
        )

    def _event_for_occurrence(
        self, occurrence: int
    ) -> ConfirmationEvent | None:
        return next(
            (
                event
                for event in self._events
                if event.occurrence == occurrence
            ),
            None,
        )

    def source_clause_for_prompt(
        self,
        *,
        receipts: list[MutationConfirmationReceipt],
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Expose only the event that belongs to the visible decision boundary."""

        pending = [
            receipt
            for receipt in receipts
            if receipt.state == "pending"
            and receipt.proposal_message_index is not None
        ]
        if len(pending) > 1:
            raise ValueError("multiple mutation confirmations are pending")
        if pending:
            receipt = pending[0]
            if self._active_event is not None:
                if self._customer_correction_seen:
                    correction_was_applied = (
                        receipt.action_sha256
                        != self._active_origin_action_sha256
                        and _proposal_acknowledges_event(
                            messages=messages,
                            receipt=receipt,
                            event=self._active_event,
                        )
                    )
                    if correction_was_applied:
                        next_event = self._event_for_occurrence(
                            self._confirmation_count + 1
                        )
                        return (
                            next_event.source_clause
                            if next_event is not None
                            else None
                        )
                return self._active_event.source_clause
            next_event = self._event_for_occurrence(
                self._confirmation_count + 1
            )
            return next_event.source_clause if next_event is not None else None
        if _assistant_invited_a_decision_or_update(messages):
            next_event = self._event_for_occurrence(
                self._confirmation_count + 1
            )
            return next_event.source_clause if next_event is not None else None
        return None

    def evaluate_confirmation_reply(
        self,
        *,
        text: str,
        receipts: list[MutationConfirmationReceipt],
        messages: list[dict[str, Any]] | None = None,
        stopped: bool = False,
    ) -> ChoreographyViolation | None:
        pending = [
            receipt
            for receipt in receipts
            if receipt.state == "pending" and receipt.proposal_message_index is not None
        ]
        if not pending:
            if not stopped and self._active_event is None:
                next_occurrence = self._confirmation_count + 1
                next_event = self._event_for_occurrence(next_occurrence)
                if next_event is not None and _satisfies(text, next_event):
                    visible_messages = messages or []
                    if _assistant_invited_a_decision_or_update(visible_messages):
                        self._confirmation_count = next_occurrence
                        self._active_event = next_event
                        self._active_origin_action_sha256 = None
                        self._customer_correction_seen = True
                    else:
                        trigger = _latest_visible_assistant_message(visible_messages)
                        if trigger is None:
                            raise ValueError(
                                "premature simulator disclosure has no visible prompt"
                            )
                        trigger_index, trigger_message = trigger
                        return ChoreographyViolation(
                            event=next_event,
                            boundary="premature_confirmation_condition",
                            trigger_message_index=trigger_index,
                            trigger_message_sha256=digest_value(trigger_message),
                        )
            return None
        if len(pending) != 1:
            raise ValueError("multiple mutation confirmations are pending")
        receipt = pending[0]
        new_intent = receipt.intent_id not in self._seen_intents
        if new_intent:
            self._seen_intents.add(receipt.intent_id)

            if self._active_event is not None and self._customer_correction_seen:
                if receipt.action_sha256 == self._active_origin_action_sha256:
                    # The agent ignored the visible correction. Require the
                    # simulator to restate it naturally; do not turn an agent
                    # mistake into hidden benchmark assistance.
                    self._customer_correction_seen = False
                    return ChoreographyViolation(
                        event=self._active_event,
                        boundary="prepared_action",
                        staged_tool_call_id=receipt.staged_tool_call_id,
                        staged_action_sha256=receipt.action_sha256,
                    )
                if not _proposal_acknowledges_event(
                    messages=messages or [],
                    receipt=receipt,
                    event=self._active_event,
                ):
                    self._customer_correction_seen = False
                    return ChoreographyViolation(
                        event=self._active_event,
                        boundary="prepared_action",
                        staged_tool_call_id=receipt.staged_tool_call_id,
                        staged_action_sha256=receipt.action_sha256,
                    )
                self._active_event = None
                self._active_origin_action_sha256 = None
                self._customer_correction_seen = False

            if self._active_event is None:
                self._confirmation_count += 1
                self._active_event = self._event_for_occurrence(
                    self._confirmation_count
                )
                self._active_origin_action_sha256 = (
                    receipt.action_sha256 if self._active_event is not None else None
                )

        event = self._active_event
        if stopped:
            if event is None:
                event = ConfirmationEvent(
                    event_id="pending-confirmation-stop",
                    occurrence=self._confirmation_count,
                    source_clause=(
                        "The agent is waiting for a customer decision. Respond to "
                        "that pending proposal before ending the conversation."
                    ),
                    required_groups=(),
                    proposal_groups=(),
                )
            return ChoreographyViolation(
                event=event,
                boundary="prepared_action",
                staged_tool_call_id=receipt.staged_tool_call_id,
                staged_action_sha256=receipt.action_sha256,
            )

        if event is None:
            return None
        if self._customer_correction_seen:
            return ChoreographyViolation(
                event=event,
                boundary="prepared_action",
                staged_tool_call_id=receipt.staged_tool_call_id,
                staged_action_sha256=receipt.action_sha256,
            )
        if _satisfies(text, event):
            self._customer_correction_seen = True
            return None
        return ChoreographyViolation(
            event=event,
            boundary="prepared_action",
            staged_tool_call_id=receipt.staged_tool_call_id,
            staged_action_sha256=receipt.action_sha256,
        )


def simulator_repair_instruction(violation: ChoreographyViolation) -> str:
    event = violation.event
    if violation.boundary == "premature_confirmation_condition":
        return (
            "Your previous draft introduced a preference that belongs only at a "
            "later confirmation boundary. The visible agent message is still "
            "gathering information or asking you to choose among options. Answer "
            "that message using only the currently visible scenario information. "
            "Do not reveal, hint at, or apply a later conditional change yet. Do "
            "not mention hidden instructions, evaluation, an event ID, or this "
            "control message."
        )
    return (
        "Your previous draft did not follow this instruction from the customer's "
        f"scenario: {event.source_clause} Re-read the visible conversation and "
        "respond once as the customer with that correction now. Do not confirm the "
        "proposed action in the same message. Do not mention hidden instructions, "
        "evaluation, this event ID, or this control message."
    )


def confirmation_contract_inventory() -> dict[str, int]:
    return {
        "task_count": len(_CONFIRMATION_EVENTS),
        "event_count": sum(len(events) for events in _CONFIRMATION_EVENTS.values()),
    }


def confirmation_source_clauses(task: dict[str, Any]) -> tuple[str, ...]:
    return SimulatorChoreography(task).source_clauses


def confirmation_prompt_redactions(
    task: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    return SimulatorChoreography(task).prompt_redactions
