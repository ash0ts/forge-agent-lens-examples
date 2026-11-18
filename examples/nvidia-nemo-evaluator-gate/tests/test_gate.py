"""Offline tests for the NeMo Evaluator release gate."""
from __future__ import annotations

import json
from pathlib import Path

from evaluator_gate.cli import build_result
from evaluator_gate.data import DATA_DIR, VARIANTS, control_cases, load_policy, load_prompts, load_tickets
from evaluator_gate.gate import decide, evaluate_control, reconcile, score

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def test_replay_october_passes_and_rejects_control():
    cases = load_tickets(DATA_DIR / "tickets.json")["cases"]
    prompts = load_prompts(DATA_DIR / "prompts.json")
    policy = load_policy(DATA_DIR / "policy.json")
    ev = EXAMPLE_ROOT / "evidence" / "replay-2026-10"
    records = [json.loads(line) for line in (ev / "responses.jsonl").read_text().splitlines() if line.strip()]
    c_records = [json.loads(line) for line in (ev / "control-responses.jsonl").read_text().splitlines() if line.strip()]
    control = evaluate_control(cases, c_records, prompts, policy)
    result = build_result(cases, prompts, policy, records, model="test", control=control, mode="replay")

    d = result["decision"]
    assert d["decision"] == "pass"
    assert result["ci_exit_code"] == 0
    assert result["enable_thinking"] is False
    assert d["variants"]["baseline"]["heldout"]["correct"] == 42
    assert d["variants"]["candidate"]["heldout"]["correct"] == 48
    assert control["metrics"]["correct"] == 2
    assert control["rejected"] is True
    assert d["checks"]["negative_control_rejected"] is True


def test_paired_regression_holds_release():
    cases = load_tickets(DATA_DIR / "tickets.json")["cases"]
    policy = load_policy(DATA_DIR / "policy.json")
    records = []
    flipped = False
    for c in cases:
        records.append({"case_id": c["case_id"], "variant": "baseline", "output": c["expected"], "state": "scored"})
        out = c["expected"]
        if c["split"] == "heldout" and not flipped:
            out = "billing" if c["expected"] != "billing" else "inference"
            flipped = True
        records.append({"case_id": c["case_id"], "variant": "candidate", "output": out, "state": "scored"})

    rows, issues = reconcile(cases, records, VARIANTS)
    score(rows)
    d = decide(rows, issues, policy)
    assert d["checks"]["paired_regression_budget"] is False
    assert d["decision"] == "hold"


def test_missing_or_off_vocab_stays_in_denominator():
    cases = load_tickets(DATA_DIR / "tickets.json")["cases"]
    policy = load_policy(DATA_DIR / "policy.json")
    records = []
    for i, c in enumerate(cases):
        records.append({"case_id": c["case_id"], "variant": "baseline", "output": c["expected"], "state": "scored"})
        if i == 0:
            continue  # missing candidate row
        if i == 1:
            records.append({"case_id": c["case_id"], "variant": "candidate",
                            "output": f"The answer is {c['expected']}.", "state": "scored"})
        else:
            records.append({"case_id": c["case_id"], "variant": "candidate", "output": c["expected"], "state": "scored"})

    rows, issues = reconcile(cases, records, VARIANTS)
    score(rows)
    d = decide(rows, issues, policy)
    assert d["checks"]["valid_membership"] is False
    assert d["checks"]["complete_responses"] is False
    assert d["checks"]["strict_label_contract"] is False
    assert d["decision"] == "hold"


def test_passing_control_breaks_gate():
    cases = load_tickets(DATA_DIR / "tickets.json")["cases"]
    prompts = load_prompts(DATA_DIR / "prompts.json")
    policy = load_policy(DATA_DIR / "policy.json")
    records = [{"case_id": c["case_id"], "variant": v, "output": c["expected"], "state": "scored"}
               for c in cases for v in VARIANTS]
    bad_control = [{"case_id": c["case_id"], "variant": "control", "output": c["expected"], "state": "scored"}
                   for c in control_cases(cases)]
    control = evaluate_control(cases, bad_control, prompts, policy)
    assert control["rejected"] is False
    rows, issues = reconcile(cases, records, VARIANTS)
    score(rows)
    d = decide(rows, issues, policy, control=control)
    assert d["checks"]["negative_control_rejected"] is False
    assert d["decision"] == "hold"
