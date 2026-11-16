from __future__ import annotations

import json
import math

import pytest

from retail_support_study_v1.confirmation import (
    MutationConfirmationController,
    action_sha256,
    is_unqualified_assent,
    proposal_matches_action,
)


@pytest.mark.parametrize(
    "text",
    [
        "Yes.",
        "YES, PLEASE!",
        "Sounds good.",
        "Okay",
        "Go ahead!",
        "That’s right.",
        "I confirm.",
        "Yes, please proceed.",
        "Yes, cancel that one.",
        "Yes, go ahead.",
        "Yes, do it.",
        "Yes, I confirm.",
    ],
)
def test_whole_message_unqualified_assent_is_accepted(text: str) -> None:
    assert is_unqualified_assent(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Yes, but don't cancel it.",
        "Yes, cancel order #R2 instead.",
        "Yes if the refund is immediate.",
        "Please cancel #R1.",
        "No.",
        "Yes?",
        "Okay; use the other card.",
        "Go ahead: cancel only the blue item.",
        "Sounds good except for the address.",
    ],
)
def test_corrections_conditions_targets_and_negations_are_not_assent(
    text: str,
) -> None:
    assert is_unqualified_assent(text) is False


def test_batch_preparation_cannot_authorize_a_different_action() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-batch")
    first = {"order_id": "#R1", "reason": "no longer needed"}
    second = {"order_id": "#R2", "reason": "no longer needed"}

    controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=first,
        tool_call_id="call-first-stage",
        message_index=2,
    )
    controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=second,
        tool_call_id="call-second-stage",
        message_index=3,
    )
    controller.observe_assistant_message(
        text="I can cancel order #R2 because it is no longer needed. Proceed?",
        message_index=4,
    )
    approved = controller.observe_user_message(
        text="Yes, please proceed.",
        message_index=5,
    )
    assert approved.decision == "approved"

    # Approval for #R2 must not execute #R1; changing the action only stages #R1.
    restaged = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=first,
        tool_call_id="call-first-restage",
        message_index=6,
    )
    assert restaged.authorized is False
    assert restaged.decision == "restaged"

    controller.observe_assistant_message(
        text="I can cancel order #R1 because it is no longer needed. Proceed?",
        message_index=7,
    )
    decision = controller.observe_user_message(
        text="Yes, cancel that one.",
        message_index=8,
    )
    assert decision.decision == "approved"
    consumed = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=first,
        tool_call_id="call-first-execute",
        message_index=9,
    )
    assert consumed.authorized is True
    assert consumed.decision == "consumed"


def _stage_and_approve(
    controller: MutationConfirmationController,
    *,
    tool_name: str = "cancel_pending_order",
    arguments: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    action_arguments = arguments or {
        "order_id": "#R1",
        "reason": "no longer needed",
    }
    staged = controller.guard_write(
        tool_name=tool_name,
        arguments=action_arguments,
        tool_call_id="call-stage",
        message_index=2,
    )
    assert staged.authorized is False
    assert staged.decision == "staged"
    assert staged.intent_id is not None
    controller.observe_assistant_message(
        text=(
            f"I can {tool_name.replace('_', ' ')} with these exact values: "
            f"{json.dumps(action_arguments, sort_keys=True)}. "
            "Would you like me to proceed?"
        ),
        message_index=3,
    )
    confirmation = controller.observe_user_message(
        text="Yes, please.",
        message_index=4,
    )
    assert confirmation.decision == "approved"
    return staged.intent_id, action_arguments


def test_exact_retry_consumes_confirmation_once() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-1")
    arguments = {
        "order_id": "#R1",
        "reason": "no longer needed",
    }

    staged = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-stage",
        message_index=2,
    )
    assert staged.authorized is False
    assert staged.decision == "staged"
    assert staged.public_result == {
        "status": "action_prepared",
        "message": (
            "No state changed. Explain the exact proposed action and its "
            "consequences, then ask the customer for a new, explicit "
            "confirmation. After an unqualified confirmation, repeat this "
            "exact tool call once."
        ),
    }
    assert staged.model_dump() == {
        "authorized": False,
        "decision": "staged",
        "public_result": staged.public_result,
    }

    awaiting = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-too-early",
        message_index=3,
    )
    assert awaiting.authorized is False
    assert awaiting.decision == "awaiting_confirmation"

    controller.observe_assistant_message(
        text=(
            "I can cancel order #R1 because it is no longer needed. "
            "Would you like me to proceed?"
        ),
        message_index=4,
    )
    approved = controller.observe_user_message(
        text="Sounds good.",
        message_index=5,
    )
    assert approved.decision == "approved"
    assert approved.model_dump() == {"decision": "approved"}

    consumed = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-consume",
        message_index=6,
    )
    assert consumed.authorized is True
    assert consumed.decision == "consumed"
    assert consumed.public_result is None
    assert consumed.model_dump() == {
        "authorized": True,
        "decision": "consumed",
        "public_result": None,
    }
    assert controller.mutation_confirmation_ok is True

    receipts = controller.private_receipts()
    assert len(receipts) == 1
    assert receipts[0].state == "consumed"
    assert receipts[0].consumed_tool_call_id == "call-consume"
    assert receipts[0].consumed_message_index == 6

    second_attempt = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-second-attempt",
        message_index=7,
    )
    assert second_attempt.authorized is False
    assert second_attempt.decision == "staged"
    assert len(controller.private_receipts()) == 2
    assert controller.private_receipts()[1].sequence == 2


@pytest.mark.parametrize(
    ("tool_name", "original", "changed"),
    [
        (
            "cancel_pending_order",
            {"order_id": "#R1", "reason": "no longer needed"},
            {"order_id": "#R2", "reason": "no longer needed"},
        ),
        (
            "cancel_pending_order",
            {"order_id": "#R1", "reason": "no longer needed"},
            {"order_id": "#R1", "reason": "ordered by mistake"},
        ),
        (
            "modify_pending_order_address",
            {
                "order_id": "#R1",
                "address": {"city": "Easton", "zip": "10001"},
            },
            {
                "order_id": "#R1",
                "address": {"city": "Weston", "zip": "10001"},
            },
        ),
        (
            "modify_pending_order_items",
            {"order_id": "#R1", "item_ids": ["item-a", "item-b"]},
            {"order_id": "#R1", "item_ids": ["item-b", "item-a"]},
        ),
    ],
)
def test_any_argument_drift_restages_instead_of_consuming(
    tool_name: str,
    original: dict[str, object],
    changed: dict[str, object],
) -> None:
    controller = MutationConfirmationController(conversation_id="conversation-2")
    first_intent, _ = _stage_and_approve(
        controller,
        tool_name=tool_name,
        arguments=original,
    )

    result = controller.guard_write(
        tool_name=tool_name,
        arguments=changed,
        tool_call_id="call-changed",
        message_index=5,
    )

    assert result.authorized is False
    assert result.decision == "restaged"
    assert result.intent_id != first_intent
    receipts = controller.private_receipts()
    assert [receipt.state for receipt in receipts] == ["superseded", "pending"]
    assert receipts[0].invalidation_reason == "argument_change"
    assert receipts[0].replaced_by_intent_id == receipts[1].intent_id
    assert controller.mutation_confirmation_ok is None


def test_changing_the_write_tool_restages_even_when_arguments_match() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-3")
    arguments = {"order_id": "#R1", "item_ids": ["item-a"]}
    _stage_and_approve(
        controller,
        tool_name="return_delivered_order_items",
        arguments=arguments,
    )

    result = controller.guard_write(
        tool_name="exchange_delivered_order_items",
        arguments=arguments,
        tool_call_id="call-other-tool",
        message_index=5,
    )

    assert result.authorized is False
    assert result.decision == "restaged"


def test_mapping_key_order_does_not_change_the_exact_action() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-4")
    original = {
        "order_id": "#R1",
        "address": {"city": "Easton", "zip": "10001"},
    }
    _stage_and_approve(
        controller,
        tool_name="modify_pending_order_address",
        arguments=original,
    )
    reordered = {
        "address": {"zip": "10001", "city": "Easton"},
        "order_id": "#R1",
    }

    result = controller.guard_write(
        tool_name="modify_pending_order_address",
        arguments=reordered,
        tool_call_id="call-reordered",
        message_index=5,
    )

    assert result.authorized is True
    assert result.decision == "consumed"


@pytest.mark.parametrize(
    "reply",
    [
        "No.",
        "Yes, but don't cancel it.",
        "Cancel #R2 instead.",
        "Yes if you can refund it today.",
    ],
)
def test_non_assent_invalidates_instead_of_authorizing(reply: str) -> None:
    controller = MutationConfirmationController(conversation_id="conversation-5")
    arguments = {"order_id": "#R1", "reason": "no longer needed"}
    staged = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-stage",
        message_index=1,
    )
    controller.observe_assistant_message(
        text="I can cancel #R1 because it is no longer needed. Should I proceed?",
        message_index=2,
    )

    rejected = controller.observe_user_message(text=reply, message_index=3)

    assert rejected.decision == "invalidated"
    assert rejected.intent_id == staged.intent_id
    assert controller.has_pending_intent is False
    assert controller.private_receipts()[0].invalidation_reason == "non_assent"
    retry = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-retry",
        message_index=4,
    )
    assert retry.authorized is False
    assert retry.decision == "staged"
    assert retry.intent_id != staged.intent_id


def test_assent_without_a_later_visible_proposal_invalidates() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-6")
    controller.guard_write(
        tool_name="cancel_pending_order",
        arguments={"order_id": "#R1"},
        tool_call_id="call-stage",
        message_index=2,
    )

    result = controller.observe_user_message(text="Yes.", message_index=3)

    assert result.decision == "invalidated"
    receipt = controller.private_receipts()[0]
    assert receipt.state == "invalidated"
    assert receipt.invalidation_reason == "missing_proposal"


def test_assent_before_any_write_cannot_authorize_a_future_write() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-7")

    assert (
        controller.observe_user_message(text="Yes.", message_index=1).decision
        == "no_pending"
    )
    result = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments={"order_id": "#R1"},
        tool_call_id="call-stage",
        message_index=2,
    )
    assert result.authorized is False
    assert result.decision == "staged"


@pytest.mark.parametrize("later_role", ["assistant", "user"])
def test_any_later_message_after_approval_invalidates_the_approval(
    later_role: str,
) -> None:
    controller = MutationConfirmationController(conversation_id="conversation-8")
    _, arguments = _stage_and_approve(controller)

    if later_role == "assistant":
        controller.observe_assistant_message(
            text="Before I do that, here is another detail.",
            message_index=5,
        )
        expected_reason = "post_confirmation_assistant_message"
    else:
        result = controller.observe_user_message(
            text="One more thing.",
            message_index=5,
        )
        assert result.decision == "invalidated"
        expected_reason = "post_confirmation_user_message"

    receipt = controller.private_receipts()[0]
    assert receipt.state == "invalidated"
    assert receipt.invalidation_reason == expected_reason
    retry = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-after-message",
        message_index=6,
    )
    assert retry.authorized is False
    assert retry.decision == "staged"


def test_non_write_tools_do_not_create_confirmation_state() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-9")

    result = controller.guard_write(
        tool_name="get_order_details",
        arguments={"order_id": "#R1"},
        tool_call_id="call-read",
        message_index=1,
    )

    assert result.authorized is True
    assert result.decision == "not_required"
    assert controller.private_receipts() == []
    assert controller.mutation_confirmation_ok is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"value": math.nan},
        {"value": math.inf},
        {"value": b"not-json"},
        {1: "non-string-key"},
    ],
)
def test_non_json_arguments_fail_closed(arguments: dict[object, object]) -> None:
    controller = MutationConfirmationController(conversation_id="conversation-10")

    with pytest.raises(ValueError):
        controller.guard_write(
            tool_name="cancel_pending_order",
            arguments=arguments,  # type: ignore[arg-type]
            tool_call_id="call-invalid",
            message_index=1,
        )

    assert controller.private_receipts() == []


def test_private_receipts_use_digests_not_message_or_argument_content() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-11")
    arguments = {
        "order_id": "private-order-reference",
        "address": {"address1": "private street value"},
    }
    staged = controller.guard_write(
        tool_name="modify_pending_order_address",
        arguments=arguments,
        tool_call_id="call-stage",
        message_index=2,
    )
    controller.observe_assistant_message(
        text=(
            "I can modify the address for private-order-reference to "
            "private street value. Proceed?"
        ),
        message_index=3,
    )
    controller.observe_user_message(
        text="Yes, please.",
        message_index=4,
    )

    public_json = json.dumps(staged.model_dump(), sort_keys=True)
    assert staged.intent_id not in public_json
    assert staged.action_sha256 not in public_json
    assert "private-order-reference" not in public_json

    receipts = controller.private_receipts()
    private_json = json.dumps(receipts[0].model_dump(), sort_keys=True)
    assert "private-order-reference" not in private_json
    assert "private street value" not in private_json
    assert "Yes, please" not in private_json
    assert receipts[0].proposal_message_sha256 is not None
    assert receipts[0].decision_message_sha256 is not None

    approved_snapshot = receipts[0]
    controller.guard_write(
        tool_name="modify_pending_order_address",
        arguments=arguments,
        tool_call_id="call-consume",
        message_index=5,
    )
    assert approved_snapshot.state == "approved"
    assert controller.private_receipts()[0].state == "consumed"


def test_message_indexes_must_preserve_causal_order() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-12")
    controller.guard_write(
        tool_name="cancel_pending_order",
        arguments={"order_id": "#R1"},
        tool_call_id="call-stage",
        message_index=3,
    )

    with pytest.raises(ValueError, match="must follow the staged write"):
        controller.observe_assistant_message(
            text="I can cancel the order.",
            message_index=3,
        )
    with pytest.raises(ValueError, match="must follow the staged write"):
        controller.observe_user_message(text="Yes.", message_index=2)


def test_exact_retry_must_be_a_new_call_after_the_customer_decision() -> None:
    controller = MutationConfirmationController(conversation_id="conversation-13")
    _, arguments = _stage_and_approve(controller)

    with pytest.raises(ValueError, match="must follow the customer confirmation"):
        controller.guard_write(
            tool_name="cancel_pending_order",
            arguments=arguments,
            tool_call_id="call-replayed-out-of-order",
            message_index=4,
        )
    with pytest.raises(ValueError, match="must use a new tool call"):
        controller.guard_write(
            tool_name="cancel_pending_order",
            arguments=arguments,
            tool_call_id="call-stage",
            message_index=5,
        )

    assert controller.private_receipts()[0].state == "approved"
    consumed = controller.guard_write(
        tool_name="cancel_pending_order",
        arguments=arguments,
        tool_call_id="call-new-retry",
        message_index=5,
    )
    assert consumed.authorized is True
    assert consumed.decision == "consumed"


_PAYPAL_RETURN_ARGUMENTS = {
    "order_id": "#R0000042",
    "item_ids": ["item_00420"],
    "payment_method_id": "paypal_000042",
}


def test_natural_paypal_return_proposal_does_not_require_internal_ids() -> None:
    proposal = (
        "I can start the return for the jacket from order #R0000042 and send "
        "the refund to your PayPal account. You will receive an email with "
        "the return instructions. Would you like me to proceed?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="return_delivered_order_items",
        arguments=_PAYPAL_RETURN_ARGUMENTS,
    )
    assert "item_00420" not in proposal
    assert "paypal_000042" not in proposal


def test_item_modification_noun_is_a_valid_human_proposal() -> None:
    proposal = (
        "I've prepared the item modification for order #R0000042. The running "
        "shoes and keyboard will be changed to the requested variants, and the "
        "$41.92 difference will be charged to your gift card. Should I proceed?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_items",
        arguments={
            "order_id": "#R0000042",
            "item_ids": ["item_00001", "item_00002"],
            "new_item_ids": ["item_00003", "item_00004"],
            "payment_method_id": "gift_card_000042",
        },
    )


@pytest.mark.parametrize(
    "proposal",
    [
        (
            "I've prepared the payment change for order #R0000042. I can "
            "switch it to Visa ending 6411. Should I proceed?"
        ),
        (
            "The payment change for order #R0000042 is ready. I can use the "
            "card ending in 6411. Would you like me to continue?"
        ),
    ],
)
def test_credit_card_proposal_accepts_customer_facing_card_description(
    proposal: str,
) -> None:
    assert proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_payment",
        arguments={
            "order_id": "#R0000042",
            "payment_method_id": "credit_card_000042",
        },
    )


def test_credit_card_proposal_rejects_conflicting_payment_source() -> None:
    proposal = (
        "I've prepared the payment change for order #R0000042. I can switch "
        "it from Visa ending 6411 to PayPal. Should I proceed?"
    )

    assert not proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_payment",
        arguments={
            "order_id": "#R0000042",
            "payment_method_id": "credit_card_000042",
        },
    )


def test_domestic_address_proposal_does_not_require_country_name() -> None:
    proposal = (
        "I've prepared the address change for order #R0000042 to "
        "8328 Lantern Way, Unit 4959, Washington, DC 20001. "
        "Would you like me to proceed?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": "USA",
            "zip": "20001",
        },
    )


def test_domestic_address_proposal_still_requires_every_other_visible_field() -> None:
    proposal = (
        "I've prepared the address change for order #R0000042 to "
        "8328 Lantern Way, Unit 4959, Washington, DC. "
        "Would you like me to proceed?"
    )

    assert not proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": "USA",
            "zip": "20001",
        },
    )


def test_domestic_address_state_must_not_match_inside_order() -> None:
    proposal = (
        "I've prepared the address change for order #R0000042 to "
        "12 Main Street, Portland 97201. Would you like me to proceed?"
    )

    assert not proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "12 Main Street",
            "address2": "",
            "city": "Portland",
            "state": "OR",
            "country": "USA",
            "zip": "97201",
        },
    )


def test_domestic_address_rejects_explicit_conflicting_country() -> None:
    proposal = (
        "I've prepared the address change for order #R0000042 to "
        "8328 Lantern Way, Unit 4959, Washington, DC 20001, Canada. "
        "Would you like me to proceed?"
    )

    assert not proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": "USA",
            "zip": "20001",
        },
    )


@pytest.mark.parametrize(
    "proposal",
    [
        (
            "The requested ZIP is 20001. I've prepared the address change for "
            "order #R0000042 to 8328 Lantern Way, Unit 4959, Washington, "
            "DC 20001, Canada. Would you like me to proceed?"
        ),
        (
            "I've prepared the address change for order #R0000042 to "
            "8328 Lantern Way, Unit 4959, Washington, DC 20001 and set the "
            "country to Canada. Would you like me to proceed?"
        ),
    ],
)
def test_domestic_address_rejects_country_suffix_bypasses(proposal: str) -> None:
    assert not proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": "USA",
            "zip": "20001",
        },
    )


@pytest.mark.parametrize(
    "country",
    ["US", "U.S.", "USA", "U.S.A.", "United States", "United States of America"],
)
def test_domestic_address_accepts_explicit_domestic_country(country: str) -> None:
    proposal = (
        "I've prepared the address change for order #R0000042 to "
        f"8328 Lantern Way, Unit 4959, Washington, DC 20001, {country}. "
        "Would you like me to proceed?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="modify_pending_order_address",
        arguments={
            "order_id": "#R0000042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": "USA",
            "zip": "20001",
        },
    )


@pytest.mark.parametrize("country", ["US", "U.S.", "USA", "United States"])
def test_domestic_country_aliases_may_be_implicit_for_user_address(
    country: str,
) -> None:
    proposal = (
        "I've prepared the default address change to "
        "8328 Lantern Way, Unit 4959, Washington, DC 20001. "
        "Would you like me to proceed?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="modify_user_address",
        arguments={
            "user_id": "retail_user_0042",
            "address1": "8328 Lantern Way",
            "address2": "Unit 4959",
            "city": "Washington",
            "state": "DC",
            "country": country,
            "zip": "20001",
        },
    )


def test_international_address_proposal_requires_and_accepts_country_name() -> None:
    arguments = {
        "user_id": "retail_user_0042",
        "address1": "10 King Street",
        "address2": "",
        "city": "Toronto",
        "state": "ON",
        "country": "Canada",
        "zip": "M5H 1A1",
    }
    missing_country = (
        "I've prepared the default address change to "
        "10 King Street, Toronto, ON M5H 1A1. Would you like me to proceed?"
    )
    complete = (
        "I've prepared the default address change to "
        "10 King Street, Toronto, ON M5H 1A1, Canada. "
        "Would you like me to proceed?"
    )

    assert not proposal_matches_action(
        missing_country,
        tool_name="modify_user_address",
        arguments=arguments,
    )
    assert proposal_matches_action(
        complete,
        tool_name="modify_user_address",
        arguments=arguments,
    )


@pytest.mark.parametrize(
    "proposal",
    [
        (
            "I can return the jacket from order #R0000099 and send the refund "
            "to your PayPal account. Should I proceed?"
        ),
        (
            "I can return item_00999 from order #R0000042 and send the refund "
            "to your PayPal account. Should I proceed?"
        ),
        (
            "I can return the jacket from order #R0000042 and send the refund "
            "to your gift card. Should I proceed?"
        ),
    ],
)
def test_observable_wrong_return_target_is_rejected(proposal: str) -> None:
    assert not proposal_matches_action(
        proposal,
        tool_name="return_delivered_order_items",
        arguments=_PAYPAL_RETURN_ARGUMENTS,
    )


def test_human_proposal_matching_and_action_digest_ignore_mapping_key_order() -> None:
    reordered = {
        "payment_method_id": "paypal_000042",
        "item_ids": ["item_00420"],
        "order_id": "#R0000042",
    }
    proposal = (
        "I can return the jacket from order #R0000042 and refund it to PayPal. "
        "Would you like me to go ahead?"
    )

    assert proposal_matches_action(
        proposal,
        tool_name="return_delivered_order_items",
        arguments=reordered,
    )
    assert action_sha256(
        "return_delivered_order_items", _PAYPAL_RETURN_ARGUMENTS
    ) == action_sha256("return_delivered_order_items", reordered)


def test_natural_proposal_still_requires_a_later_standalone_assent() -> None:
    controller = MutationConfirmationController(conversation_id="return-conversation")
    staged = controller.guard_write(
        tool_name="return_delivered_order_items",
        arguments=_PAYPAL_RETURN_ARGUMENTS,
        tool_call_id="prepare-return",
        message_index=2,
    )
    controller.observe_assistant_message(
        text=(
            "I can return the jacket from order #R0000042 and refund it to "
            "PayPal. Would you like me to proceed?"
        ),
        message_index=3,
    )

    too_early = controller.guard_write(
        tool_name="return_delivered_order_items",
        arguments=_PAYPAL_RETURN_ARGUMENTS,
        tool_call_id="retry-before-answer",
        message_index=4,
    )

    assert staged.authorized is False
    assert too_early.authorized is False
    assert too_early.decision == "awaiting_confirmation"
    assert controller.mutation_confirmation_ok is None
    approved = controller.observe_user_message(text="Yes, please.", message_index=5)
    assert approved.decision == "approved"
    consumed = controller.guard_write(
        tool_name="return_delivered_order_items",
        arguments=_PAYPAL_RETURN_ARGUMENTS,
        tool_call_id="execute-after-answer",
        message_index=6,
    )
    assert consumed.authorized is True
    assert consumed.decision == "consumed"
