"""Offline tests for the NeMo Relay SQL agent and OTLP GenAI ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from relay_agent_lens.agent import parse_action
from relay_agent_lens.database import TASKS, QueryRejected, execute_readonly, fixture
from relay_agent_lens.forge import _span_rows

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def test_parse_action_accepts_plain_and_fenced_json():
    assert parse_action('{"action":"schema"}') == {"action": "schema"}
    assert parse_action('```json\n{"action":"finish","answer":[[1,"a"]]}\n```') == {
        "action": "finish",
        "answer": [[1, "a"]],
    }
    with pytest.raises(ValueError):
        parse_action('{"action":"drop_database"}')


def test_readonly_authorizer_blocks_writes_and_reference_sql_handles_traps():
    db = fixture(11)
    try:
        with pytest.raises(QueryRejected):
            execute_readonly(db, "DELETE FROM orders")
        by_id = {t.id: execute_readonly(db, t.reference_sql) for t in TASKS}
        # Seed 11 trap check: cancelled giant order (id 1105) is excluded; tie between 1106 and 1107 picks 1106.
        assert by_id["largest_paid_order"] == [[1106, 1244]]
        assert by_id["no_paid_orders"] == [[3, "customer_3"], [4, "customer_4"], [6, "customer_6"]]
    finally:
        db.close()


def test_recorded_relay_export_parses_20_spans_and_timeout_event():
    ev_dir = EXAMPLE_ROOT / "evidence" / "live-2026-11-18"
    rows = _span_rows(ev_dir)
    assert len(rows) == 20
    assert [r["operation"] for r in rows].count("invoke_agent") == 3
    assert [r["operation"] for r in rows].count("chat") == 10
    assert [r["operation"] for r in rows].count("execute_tool") == 7

    errors = [r for r in rows if r["status_code"] == 2]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "internal_error"
    assert errors[0]["exception_type"] == "TimeoutError"
    assert "TimeoutError" in errors[0]["status_message"]
