from __future__ import annotations

import pytest

from retail_support_study_v1.canonical import load_json
from retail_support_study_v1.confirmation import MutationConfirmationReceipt
from retail_support_study_v1.constants import PACKAGE_ROOT
from retail_support_study_v1.simulator_contracts import (
    SimulatorChoreography,
    confirmation_contract_inventory,
)


def _task(task_id: str) -> dict[str, object]:
    return next(
        task
        for task in load_json(PACKAGE_ROOT / "prepared" / "tasks.runtime.json")
        if str(task["id"]) == task_id
    )


def _pending(sequence: int) -> MutationConfirmationReceipt:
    return MutationConfirmationReceipt(
        conversation_id="conversation",
        sequence=sequence,
        intent_id=f"intent-{sequence}",
        tool_name="modify_pending_order_items",
        action_sha256=str(sequence) * 64,
        staged_tool_call_id=f"tool-{sequence}",
        staged_message_index=sequence,
        state="pending",
        proposal_message_index=sequence + 1,
        proposal_message_sha256="a" * 64,
    )


def _pending_with_action(sequence: int, action_sha256: str) -> MutationConfirmationReceipt:
    receipt = _pending(sequence)
    receipt.action_sha256 = action_sha256
    return receipt


def _proposal_messages(index: int, text: str) -> list[dict[str, str]]:
    messages = [
        {"role": "assistant", "content": "filler"}
        for _ in range(index + 1)
    ]
    messages[index] = {"role": "assistant", "content": text}
    return messages


def test_contract_inventory_covers_every_reviewed_confirmation_event() -> None:
    assert confirmation_contract_inventory() == {
        "task_count": 14,
        "event_count": 15,
    }
    for task_id in ("5", "6", "7", "8", "9", "18", "21", "24", "71", "72", "84", "91", "92", "98"):
        SimulatorChoreography(_task(task_id))


def test_task21_requires_the_scenario_correction_once_then_allows_assent() -> None:
    choreography = SimulatorChoreography(_task("21"))
    first = _pending(1)
    assert choreography.evaluate_confirmation_reply(
        text="Yes, please.", receipts=[first]
    ) is not None
    assert choreography.evaluate_confirmation_reply(
        text=(
            "Before I confirm, add the no-backlight 80% keyboard with linear "
            "switches too."
        ),
        receipts=[first],
    ) is None

    first.state = "invalidated"
    second = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes, please.",
        receipts=[first, second],
        messages=_proposal_messages(
            second.proposal_message_index or 0,
            (
                "I can also change the no-backlight 80% linear keyboard in "
                "the order."
            ),
        ),
    ) is None


@pytest.mark.parametrize(
    "proposal",
    [
        (
            "Here is the full modification for the order: Running Shoes, black, "
            "synthetic, size 9, rubber sole; Mechanical Keyboard, no backlight, "
            "80% size, linear switch. Shall I proceed?"
        ),
        (
            "The proposed order modification lists the Mechanical Keyboard as "
            "no-backlight, 80 percent size, with linear switches. Please confirm."
        ),
    ],
)
def test_task21_proposal_acknowledges_exact_variant_without_magic_action_words(
    proposal: str,
) -> None:
    choreography = SimulatorChoreography(_task("21"))
    first = _pending_with_action(1, "a" * 64)
    correction = (
        "Also change the keyboard to the no-backlight 80% model with linear "
        "switches."
    )
    assert choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[first],
    ) is None

    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            proposal,
        ),
    ) is None


@pytest.mark.parametrize(
    "correction",
    [
        (
            "Yes, I'd also like to change the mechanical keyboard to the "
            "no-backlight, size 80%, linear-switch version."
        ),
        (
            "Before I confirm, I'd also like to switch the mechanical keyboard "
            "to the no-backlight, 80 percent, linear-switch model."
        ),
        (
            "Before you go ahead with the shoe change, I also need to modify "
            "the mechanical keyboard. I want to switch it to the no-backlight, "
            "size 80%, linear-switch model."
        ),
    ],
)
def test_task21_correction_before_tool_preparation_is_not_requested_twice(
    correction: str,
) -> None:
    choreography = SimulatorChoreography(_task("21"))
    assert choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[],
        messages=[
            {
                "role": "assistant",
                "content": (
                    "Before I proceed, are there any other items in the order "
                    "that you want to change?"
                ),
            }
        ],
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "The proposed order modification includes the Mechanical "
                "Keyboard with no backlight, 80% size, and linear switches."
            ),
        ),
    ) is None


@pytest.mark.parametrize(
    "non_request",
    [
        (
            "I am not asking you to change the no-backlight 80% linear "
            "keyboard."
        ),
        (
            "Can you explain whether I could switch to the no-backlight 80% "
            "linear keyboard later?"
        ),
    ],
)
def test_task21_does_not_consume_negated_or_hypothetical_keyboard_mentions(
    non_request: str,
) -> None:
    choreography = SimulatorChoreography(_task("21"))
    assert choreography.evaluate_confirmation_reply(
        text=non_request,
        receipts=[],
        messages=[
            {
                "role": "assistant",
                "content": "Before I proceed, are there any other changes?",
            }
        ],
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "The proposed order modification lists the Mechanical Keyboard "
                "as no-backlight, 80% size, with linear switches."
            ),
        ),
    ) is not None


def test_task21_anything_else_prompt_records_the_early_correction() -> None:
    choreography = SimulatorChoreography(_task("21"))
    correction = (
        "Yes, I also want to change the mechanical keyboard to the no-backlight, "
        "size 80%, linear-switch mechanical keyboard."
    )
    assert choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[],
        messages=[
            {
                "role": "assistant",
                "content": (
                    "You mentioned a couple of changes. Is there anything else "
                    "in this order you'd like to modify?"
                ),
            }
        ],
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "Mechanical Keyboard: no backlight, 80%, linear. "
                "Running Shoes: black, synthetic, size 9, rubber."
            ),
        ),
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "The proposed order modification lists the Mechanical Keyboard "
                "as no-backlight, 80% size, with linear switches. Shall I proceed?"
            ),
        ),
    ) is None


def test_task21_initial_conditional_change_is_rejected_before_decision_prompt() -> None:
    choreography = SimulatorChoreography(_task("21"))
    correction = (
        "I also want to change the mechanical keyboard to the no-backlight, "
        "80% size, linear-switch model."
    )
    premature = choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[],
        messages=[{"role": "assistant", "content": "Hi! How can I help?"}],
    )
    assert premature is not None
    assert premature.boundary == "premature_confirmation_condition"
    assert premature.trigger_message_index == 0

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "The proposed order modification lists the Mechanical Keyboard "
                "as no-backlight, 80% size, with linear switches. Shall I proceed?"
            ),
        ),
    ) is not None


def test_task21_early_correction_does_not_excuse_an_omitted_proposal() -> None:
    choreography = SimulatorChoreography(_task("21"))
    correction = (
        "I also want to change the mechanical keyboard to the no-backlight, "
        "80% size, linear-switch model."
    )
    assert choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[],
        messages=[
            {
                "role": "assistant",
                "content": "Before I proceed, are there any other changes?",
            }
        ],
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            "I am ready to change only the running shoes. Shall I proceed?",
        ),
    ) is not None


@pytest.mark.parametrize(
    ("correction", "proposal"),
    [
        (
            (
                "Actually, return the more expensive tablet and put the refund "
                "on a gift card instead."
            ),
            (
                "The updated return is the gold 8-inch 128GB tablet from order "
                "#R0000950, with the refund going to your gift card."
            ),
        ),
        (
            (
                "I'd rather send back the pricier one and receive store credit."
            ),
            (
                "I have prepared the return of the gold 128GB tablet, refunded "
                "as store credit."
            ),
        ),
    ],
)
def test_task84_proposal_can_name_the_selected_tablet_without_repeating_comparison(
    correction: str,
    proposal: str,
) -> None:
    choreography = SimulatorChoreography(_task("84"))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text=correction,
        receipts=[first],
    ) is None

    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            proposal,
        ),
    ) is None


def test_task84_updated_proposal_must_still_name_return_and_gift_card() -> None:
    choreography = SimulatorChoreography(_task("84"))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Return the more expensive tablet to a gift card instead.",
        receipts=[first],
    ) is None

    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            "I have selected the gold 128GB tablet. Shall I proceed?",
        ),
    ) is not None


def test_task84_correction_before_preparation_is_honored_at_final_confirmation() -> None:
    choreography = SimulatorChoreography(_task("84"))
    assert choreography.evaluate_confirmation_reply(
        text=(
            "Actually, return the more expensive one, the gold 128GB tablet, "
            "and refund it to a gift card instead."
        ),
        receipts=[],
        messages=[
            {
                "role": "assistant",
                "content": "Shall I proceed with the cheaper tablet return?",
            }
        ],
    ) is None

    pending = _pending_with_action(1, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[pending],
        messages=_proposal_messages(
            pending.proposal_message_index or 0,
            (
                "The updated return is the gold 128GB tablet from order "
                "#R0000950, refunded to your gift card. Shall I proceed?"
            ),
        ),
    ) is None


@pytest.mark.parametrize(
    "proposal",
    [
        "Payment method: Mastercard ending 5834.",
        "The refund will go to the Visa card ending 2147.",
    ],
)
def test_task98_proposal_can_name_the_selected_card_without_magic_action_words(
    proposal: str,
) -> None:
    choreography = SimulatorChoreography(_task("98"))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Use the other card for the payment instead.",
        receipts=[first],
    ) is None

    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            proposal,
        ),
    ) is None


@pytest.mark.parametrize("task_id", ["5", "6", "7", "8", "9"])
def test_exchange_correction_can_use_swap(task_id: str) -> None:
    choreography = SimulatorChoreography(_task(task_id))
    pending = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Just swap the desk lamp.",
        receipts=[pending],
    ) is None


@pytest.mark.parametrize("task_id", ["71", "72"])
def test_paypal_backpack_proposal_does_not_require_a_magic_action_verb(
    task_id: str,
) -> None:
    choreography = SimulatorChoreography(_task(task_id))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Use PayPal and only update the backpack.",
        receipts=[first],
    ) is None

    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            "Backpack: blue canvas. Payment method: PayPal.",
        ),
    ) is None


def test_task21_repeats_the_correction_when_the_agent_did_not_change_action() -> None:
    choreography = SimulatorChoreography(_task("21"))
    first = _pending_with_action(1, "a" * 64)
    correction = (
        "Also change the keyboard to the no-backlight 80% model with linear "
        "switches."
    )
    assert choreography.evaluate_confirmation_reply(
        text=correction, receipts=[first]
    ) is None

    first.state = "invalidated"
    unchanged = _pending_with_action(2, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.", receipts=[first, unchanged]
    ) is not None
    assert choreography.evaluate_confirmation_reply(
        text=correction, receipts=[first, unchanged]
    ) is None


def test_stop_token_is_rejected_while_a_confirmation_is_pending() -> None:
    choreography = SimulatorChoreography(_task("22"))
    violation = choreography.evaluate_confirmation_reply(
        text="Yes.", receipts=[_pending(1)], stopped=True
    )
    assert violation is not None
    assert violation.event.event_id == "pending-confirmation-stop"


def test_task22_is_not_treated_as_a_confirmation_time_correction() -> None:
    choreography = SimulatorChoreography(_task("22"))
    assert choreography.evaluate_confirmation_reply(
        text="Yes, please.", receipts=[_pending(1)]
    ) is None


def test_ambiguous_pronouns_do_not_satisfy_task24_or_task84() -> None:
    task24 = SimulatorChoreography(_task("24"))
    assert task24.evaluate_confirmation_reply(
        text="Actually, don't cancel it; keep it.", receipts=[_pending(1)]
    ) is None

    task84 = SimulatorChoreography(_task("84"))
    assert task84.evaluate_confirmation_reply(
        text="Return the pricier one to store credit.", receipts=[_pending(1)]
    ) is None


@pytest.mark.parametrize(
    ("task_id", "positive", "negative"),
    [
        ("5", "Just exchange the desk lamp.", "Do not exchange the desk lamp."),
        ("6", "Just exchange the desk lamp.", "Do not exchange the desk lamp."),
        ("7", "Only exchange the lamp.", "Do not exchange the desk lamp."),
        ("8", "Just exchange the lamp.", "Do not exchange the desk lamp."),
        ("9", "Only exchange the desk lamp.", "Do not exchange the desk lamp."),
        ("18", "Exchange it for the same one.", "Do not exchange the same chair."),
        (
            "21",
            "Also change the no-backlight 80% linear keyboard.",
            "Do not change the no-backlight 80% linear keyboard.",
        ),
        ("24", "Actually, keep it.", "Do not keep the grill; cancel it."),
        (
            "71",
            "Use PayPal and just modify the backpack.",
            "Do not use PayPal; only modify the backpack.",
        ),
        (
            "72",
            "Use PayPal and only modify the backpack.",
            "Not PayPal; only modify the backpack.",
        ),
        (
            "84",
            "Return the more expensive one to a gift card.",
            "Do not return the more expensive tablet to a gift card.",
        ),
        (
            "91",
            "Also return the smart watch and the e-reader.",
            "Do not return the smart watch or e-reader.",
        ),
        (
            "92",
            "Return the smartwatch and e-reader too.",
            "Don't return the smartwatch or e-reader.",
        ),
        (
            "98",
            "Use a different card for payment instead.",
            "Do not use the other card for payment.",
        ),
    ],
)
def test_reviewed_confirmation_contracts_accept_intent_and_reject_opposite(
    task_id: str,
    positive: str,
    negative: str,
) -> None:
    positive_choreography = SimulatorChoreography(_task(task_id))
    assert positive_choreography.evaluate_confirmation_reply(
        text=positive, receipts=[_pending(1)]
    ) is None

    negative_choreography = SimulatorChoreography(_task(task_id))
    assert negative_choreography.evaluate_confirmation_reply(
        text=negative, receipts=[_pending(1)]
    ) is not None


def test_task5_second_confirmation_switches_to_returning_the_bottle() -> None:
    choreography = SimulatorChoreography(_task("5"))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Just exchange the desk lamp.", receipts=[first]
    ) is None
    first.state = "invalidated"
    second = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Instead, return the water bottle.",
        receipts=[first, second],
        messages=_proposal_messages(
            second.proposal_message_index or 0,
            "I can exchange just the desk lamp.",
        ),
    ) is None

    negative = SimulatorChoreography(_task("5"))
    negative_first = _pending_with_action(1, "a" * 64)
    assert negative.evaluate_confirmation_reply(
        text="Only exchange the lamp.", receipts=[negative_first]
    ) is None
    negative_first.state = "invalidated"
    negative_second = _pending_with_action(2, "b" * 64)
    assert negative.evaluate_confirmation_reply(
        text="Do not return the bottle.",
        receipts=[negative_first, negative_second],
        messages=_proposal_messages(
            negative_second.proposal_message_index or 0,
            "I can exchange only the desk lamp.",
        ),
    ) is not None


def test_task5_prompt_exposes_each_confirmation_event_once_in_order() -> None:
    choreography = SimulatorChoreography(_task("5"))
    first_clause, second_clause = choreography.source_clauses

    assert choreography.source_clause_for_prompt(
        receipts=[],
        messages=[{"role": "assistant", "content": "Which item is this about?"}],
    ) is None

    first = _pending_with_action(1, "a" * 64)
    first_messages = _proposal_messages(
        first.proposal_message_index or 0,
        "I can exchange the water bottle and desk lamp. Shall I proceed?",
    )
    assert choreography.source_clause_for_prompt(
        receipts=[first],
        messages=first_messages,
    ) == first_clause
    assert choreography.evaluate_confirmation_reply(
        text="Only exchange the desk lamp.",
        receipts=[first],
        messages=first_messages,
    ) is None

    first.state = "invalidated"
    second = _pending_with_action(2, "b" * 64)
    second_messages = _proposal_messages(
        second.proposal_message_index or 0,
        "I can exchange only the desk lamp. Shall I proceed?",
    )
    assert choreography.source_clause_for_prompt(
        receipts=[first, second],
        messages=second_messages,
    ) == second_clause
    assert choreography.evaluate_confirmation_reply(
        text="Return the water bottle instead.",
        receipts=[first, second],
        messages=second_messages,
    ) is None

    second.state = "invalidated"
    third = _pending_with_action(3, "c" * 64)
    third_messages = _proposal_messages(
        third.proposal_message_index or 0,
        "I can return the water bottle instead. Shall I proceed?",
    )
    assert choreography.source_clause_for_prompt(
        receipts=[first, second, third],
        messages=third_messages,
    ) is None
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, second, third],
        messages=third_messages,
    ) is None


@pytest.mark.parametrize(
    "assistant_prompt",
    (
        "Can you confirm which tablet?",
        "Can you confirm which tablet you would like to return?",
        "Can you confirm your return address?",
        "Can you confirm the cancellation reason?",
        "Before I proceed, which tablet?",
        "Are there any other details?",
    ),
)
def test_task84_choice_questions_do_not_trigger_confirmation_event(
    assistant_prompt: str,
) -> None:
    choreography = SimulatorChoreography(_task("84"))
    messages = [{"role": "assistant", "content": assistant_prompt}]

    assert choreography.source_clause_for_prompt(
        receipts=[],
        messages=messages,
    ) is None
    violation = choreography.evaluate_confirmation_reply(
        text=(
            "Actually, return the more expensive tablet and put the refund "
            "on my gift card."
        ),
        receipts=[],
        messages=messages,
    )

    assert violation is not None
    assert violation.boundary == "premature_confirmation_condition"


def test_task84_explicit_action_approval_triggers_confirmation_event() -> None:
    choreography = SimulatorChoreography(_task("84"))
    messages = [
        {
            "role": "assistant",
            "content": "Can you confirm you want me to return the tablet?",
        }
    ]

    assert choreography.source_clause_for_prompt(
        receipts=[],
        messages=messages,
    ) == choreography.source_clauses[0]
    assert choreography.evaluate_confirmation_reply(
        text=(
            "Actually, return the more expensive tablet and put the refund "
            "on my gift card."
        ),
        receipts=[],
        messages=messages,
    ) is None


def test_task5_ignored_correction_does_not_skip_the_second_event() -> None:
    choreography = SimulatorChoreography(_task("5"))
    first = _pending_with_action(1, "a" * 64)
    lamp_only = "Just exchange the desk lamp."
    assert choreography.evaluate_confirmation_reply(
        text=lamp_only, receipts=[first]
    ) is None

    first.state = "invalidated"
    repeated = _pending_with_action(2, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.", receipts=[first, repeated]
    ) is not None
    assert choreography.evaluate_confirmation_reply(
        text=lamp_only, receipts=[first, repeated]
    ) is None

    repeated.state = "invalidated"
    changed = _pending_with_action(3, "b" * 64)
    violation = choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, repeated, changed],
        messages=_proposal_messages(
            changed.proposal_message_index or 0,
            "I can exchange only the desk lamp.",
        ),
    )
    assert violation is not None
    assert violation.event.event_id == "task-5-confirmation-2"


@pytest.mark.parametrize(
    ("task_id", "correction", "proposal"),
    [
        (
            "18",
            "Exchange it for the same chair instead.",
            "I can exchange it for the gray leather office chair.",
        ),
        (
            "91",
            "Also return the smart watch and the e-reader.",
            "I can return the smart watch in this order first.",
        ),
        (
            "92",
            "Return the smartwatch and e-reader too.",
            "I can return the e-reader in this order first.",
        ),
    ],
)
def test_correct_fallback_or_multi_order_proposal_can_acknowledge_correction(
    task_id: str,
    correction: str,
    proposal: str,
) -> None:
    choreography = SimulatorChoreography(_task(task_id))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text=correction, receipts=[first]
    ) is None
    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(changed.proposal_message_index or 0, proposal),
    ) is None


@pytest.mark.parametrize(
    ("task_id", "correction", "proposal"),
    [
        (
            "21",
            "Also change the no-backlight 80% linear keyboard.",
            (
                "I will leave the keyboard unchanged until you confirm. If you "
                "approve, I will also change it to the no-backlight 80% linear "
                "keyboard."
            ),
        ),
        (
            "84",
            "Return the more expensive one to a gift card.",
            (
                "I will not return the more expensive tablet until you confirm. "
                "Should I return it for a gift-card refund?"
            ),
        ),
    ],
)
def test_confirmation_language_does_not_negate_a_correct_proposal(
    task_id: str,
    correction: str,
    proposal: str,
) -> None:
    choreography = SimulatorChoreography(_task(task_id))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text=correction, receipts=[first]
    ) is None
    first.state = "invalidated"
    changed = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Yes.",
        receipts=[first, changed],
        messages=_proposal_messages(changed.proposal_message_index or 0, proposal),
    ) is None


@pytest.mark.parametrize(
    ("task_id", "opposite"),
    [
        ("5", "Only exchange the bottle, not the desk lamp."),
        ("18", "Exchange a different chair, not the same one."),
        (
            "91",
            "Return the skateboards, not the smart watch or the e-reader.",
        ),
        ("92", "Return the skateboards, not the smartwatch or e-reader."),
        ("98", "Use PayPal instead, not the other card."),
    ],
)
def test_contrastive_opposite_does_not_satisfy_customer_correction(
    task_id: str,
    opposite: str,
) -> None:
    choreography = SimulatorChoreography(_task(task_id))
    assert choreography.evaluate_confirmation_reply(
        text=opposite, receipts=[_pending(1)]
    ) is not None


def test_task5_second_event_allows_contrast_with_the_lamp() -> None:
    choreography = SimulatorChoreography(_task("5"))
    first = _pending_with_action(1, "a" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Only exchange the desk lamp.", receipts=[first]
    ) is None
    first.state = "invalidated"
    second = _pending_with_action(2, "b" * 64)
    assert choreography.evaluate_confirmation_reply(
        text="Do not return the lamp; return the water bottle instead.",
        receipts=[first, second],
        messages=_proposal_messages(
            second.proposal_message_index or 0,
            "I can exchange only the desk lamp.",
        ),
    ) is None
