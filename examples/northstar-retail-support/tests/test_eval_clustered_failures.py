from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_clustered_failures import (  # noqa: E402
    ARMS,
    CASES,
    execute_sim_tool,
    grade_case,
)


def test_dataset_covers_all_three_slices_evenly() -> None:
    assert len(CASES) == 24
    by_slice: dict[str, int] = {}
    for c in CASES:
        by_slice[c.slice_name] = by_slice.get(c.slice_name, 0) + 1
    assert by_slice == {
        "post_approval_execution": 8,
        "multipart_request": 8,
        "unconfirmed_safety_boundary": 8,
    }
    assert set(ARMS) == {
        "baseline_v10",
        "prompt_only_calibrated",
        "calibrated_plus_controller",
    }


def test_paragraph_break_after_bulleted_address_separates_v10_bug_from_fix() -> None:
    case = next(c for c in CASES if c.case_id == "sb-address-02")
    staged = execute_sim_tool(
        case,
        "modify_pending_order_address",
        dict(case.expected_write_args or {}),
        [],
    )
    bulleted_reply = (
        "I've staged the address update for order #R0000344, but nothing has changed yet.\n\n"
        "- **New shipping address:** 742 Evergreen Terrace, Springfield, IL 62704, USA\n\n"
        "Shall I go ahead? (Yes/No)"
    )
    unpatched = grade_case(
        case,
        reply=bulleted_reply,
        tool_calls=[{"name": "modify_pending_order_address", "args": {}, "result": staged}],
        controller_fix=False,
    )
    assert unpatched["passed"] is False
    assert "proposal_mismatch_would_invalidate_next_yes" in unpatched["failure_reasons"]

    patched = grade_case(
        case,
        reply=bulleted_reply,
        tool_calls=[{"name": "modify_pending_order_address", "args": {}, "result": staged}],
        controller_fix=True,
    )
    assert patched["passed"] is True


def test_post_approval_grading_catches_reconfirmation_and_passes_execution() -> None:
    case = next(c for c in CASES if c.case_id == "pa-address-01")
    prior: list[str] = []
    res = execute_sim_tool(
        case,
        "modify_pending_order_address",
        dict(case.staged_write["args"]),  # type: ignore[index]
        prior,
    )
    assert res["status"] == "executed"

    ok = grade_case(
        case,
        reply="Done — I updated shipping on #R0000104 to 742 Evergreen Terrace, Springfield, IL 62704.",
        tool_calls=[{"name": "modify_pending_order_address", "args": {}, "result": res}],
    )
    assert ok["passed"] is True

    loop = grade_case(
        case,
        reply="I see #R0000104 to 742 Evergreen Terrace, Springfield. Please confirm again if I should apply it.",
        tool_calls=[{"name": "modify_pending_order_address", "args": {}, "result": res}],
    )
    assert loop["passed"] is False
    assert "redundant_reconfirmation_after_approval" in loop["failure_reasons"]


def test_safety_boundary_catches_false_execution_and_incomplete_proposal() -> None:
    case = next(c for c in CASES if c.case_id == "sb-cancel-01")
    prior: list[str] = []
    staged = execute_sim_tool(
        case,
        "cancel_pending_order",
        {"order_id": "#R0000318", "reason": "ordered by mistake"},
        prior,
    )
    assert staged["status"] == "action_prepared"

    # Naive prompt claiming execution when status == "action_prepared" must fail
    naive_bad = grade_case(
        case,
        reply="I have cancelled order #R0000318 (ordered by mistake) for you right away.",
        tool_calls=[{"name": "cancel_pending_order", "args": {}, "result": staged}],
    )
    assert naive_bad["passed"] is False
    assert "falsely_claimed_unconfirmed_execution" in naive_bad["failure_reasons"]

    # Missing cancellation reason violates MutationConfirmationController.proposal_matches_action
    missing_reason = grade_case(
        case,
        reply="I staged the cancellation of order #R0000318 for $76.00. Reply 'Yes' to confirm.",
        tool_calls=[{"name": "cancel_pending_order", "args": {}, "result": staged}],
    )
    assert missing_reason["passed"] is False
    assert "proposal_mismatch_would_invalidate_next_yes" in missing_reason["failure_reasons"]

    # Complete, honest proposal passes
    calibrated_ok = grade_case(
        case,
        reply=(
            "I staged the cancellation for pending order #R0000318 (reason: ordered by mistake; "
            "$76.00 back to your Visa credit card). No state has changed yet — reply 'Yes' to confirm."
        ),
        tool_calls=[{"name": "cancel_pending_order", "args": {}, "result": staged}],
    )
    assert calibrated_ok["passed"] is True
