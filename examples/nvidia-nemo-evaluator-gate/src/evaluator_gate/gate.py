"""Collect one answer per (ticket, prompt), score with NeMo Evaluator, apply the policy.

Every request stays in the denominator. A timeout, an empty answer or an off-vocabulary
label is a failed case, not a dropped row.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from nemo_evaluator import ScorerInput, exact_match

from .data import LABELS, VARIANTS, control_cases

Ask = Callable[..., dict]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect(cases: list[dict], prompts: dict[str, str], ask: Ask, *, journal: Path,
            max_tokens: int = 1024, max_workers: int = 4) -> list[dict]:
    """One attempt per (case, variant) in a fixed shuffled order. Rows are journaled as they land."""
    schedule = [(case, variant) for case in cases for variant in prompts]
    random.Random(491).shuffle(schedule)

    def execute(item):
        case, variant = item
        row = {"case_id": case["case_id"], "variant": variant, "started_at": stamp()}
        response = ask(prompts[variant], case["prompt"], max_tokens=max_tokens)
        output = response.get("text") or ""
        row.update(output=output, model=response.get("model"), finish_reason=response.get("finish_reason"),
                   usage=response.get("usage") or {}, latency_ms=response.get("latency_ms"),
                   http_status=response.get("http_status"), error=response.get("error"))
        if response.get("error"):
            row["state"] = "timeout" if "timeout" in str(response["error"]).lower() else "request_error"
        elif response.get("finish_reason") == "length":
            row["state"] = "truncated"
        elif not output.strip():
            row["state"] = "empty_response"
        else:
            row["state"] = "scored"
        return row

    rows = []
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w") as stream, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(execute, item): index for index, item in enumerate(schedule)}
        for future in as_completed(futures):
            row = future.result()
            row["schedule_index"] = futures[future]
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
    return sorted(rows, key=lambda r: r["schedule_index"])


def reconcile(cases: list[dict], records: list[dict], variants: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    """Join records back onto the frozen cases. Missing or duplicated rows become issues."""
    grouped = defaultdict(list)
    issues = []
    for record in records:
        key = (record.get("case_id"), record.get("variant"))
        if record.get("variant") not in variants:
            issues.append("unexpected_variant")
            continue
        grouped[key].append(record)
    rows = []
    for case in cases:
        for variant in variants:
            found = grouped[(case["case_id"], variant)]
            row = {**case, "variant": variant, "output": "", "submitted_count": len(found)}
            if not found:
                row["state"] = "missing"
            elif len(found) > 1:
                row["state"] = "duplicate"
            else:
                row.update({k: v for k, v in found[0].items() if k not in ("case_id", "variant")})
            row["strict_label"] = row["state"] == "scored" and row["output"].strip() in LABELS
            rows.append(row)
    return rows, sorted(set(issues))


def score(rows: list[dict]) -> dict:
    """NeMo Evaluator's exact_match decides correctness; the label contract decides validity."""
    import importlib.metadata
    disagreements = 0
    for row in rows:
        verdict = exact_match(ScorerInput(response=row.get("output", ""), target=row["expected"]))
        row["nemo_correct"] = bool(verdict["correct"])
        row["correct"] = bool(row["state"] == "scored" and row["strict_label"] and row["nemo_correct"])
        disagreements += row["nemo_correct"] != row["correct"]
    return {"scorer": "nemo_evaluator.exact_match", "version": importlib.metadata.version("nemo-evaluator"),
            "normalization_disagreements": disagreements}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r.get("correct") is True for r in rows)
    p = correct / n if n else None
    z = 1.959963984540054
    if n:
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        wilson = [round(max(0, center - margin), 4), round(min(1, center + margin), 4)]
    else:
        wilson = None
    latencies = sorted(r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float)))
    return {"cases": n, "correct": correct, "accuracy": p, "wilson_95": wilson,
            "complete": sum(r["state"] == "scored" for r in rows),
            "invalid_labels": sum(r["state"] == "scored" and not r["strict_label"] for r in rows),
            "states": dict(Counter(r["state"] for r in rows)),
            "latency_ms_p50": latencies[(len(latencies) - 1) // 2] if latencies else None,
            "latency_ms_p95": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None}


def decide(rows: list[dict], issues: list[str], policy: dict) -> dict:
    metrics = {v: {s: summarize([r for r in rows if r["variant"] == v and (s == "all" or r["split"] == s)])
                   for s in ("all", "development", "heldout")} for v in VARIANTS}
    groups = {label: {v: summarize([r for r in rows if r["variant"] == v and r["split"] == "heldout"
                                    and r["group"] == label]) for v in VARIANTS} for label in LABELS}
    by_id = {(r["case_id"], r["variant"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["variant"] != "baseline" or row["split"] != "heldout":
            continue
        cand = by_id[(row["case_id"], "candidate")]
        b, c = row.get("correct") is True, cand.get("correct") is True
        pairs.append({"case_id": row["case_id"], "group": row["group"], "baseline_correct": b,
                      "candidate_correct": c, "regression": b and not c, "improvement": c and not b})
    b, c = metrics["baseline"]["heldout"], metrics["candidate"]["heldout"]
    delta = (c["accuracy"] - b["accuracy"]) if c["accuracy"] is not None and b["accuracy"] is not None else None
    checks = {
        "valid_membership": not issues and all(r["submitted_count"] == 1 for r in rows),
        "required_heldout_count": b["cases"] >= policy["required_heldout_cases"] and b["cases"] == c["cases"],
        "complete_responses": not policy["require_complete"] or all(r["state"] == "scored" for r in rows),
        "strict_label_contract": not policy["require_strict_label"] or all(r["strict_label"] for r in rows),
        "candidate_absolute_floor": c["accuracy"] is not None and c["accuracy"] >= policy["candidate_heldout_accuracy_min"],
        "every_category_floor": all(g["candidate"]["accuracy"] is not None
                                    and g["candidate"]["accuracy"] >= policy["group_accuracy_min"] for g in groups.values()),
        "candidate_delta_floor": delta is not None and delta >= policy["min_heldout_delta"],
        "paired_regression_budget": sum(p["regression"] for p in pairs) <= policy["max_paired_regressions"],
    }
    return {"decision": "pass" if all(checks.values()) else "hold", "checks": checks, "variants": metrics,
            "heldout_groups": groups, "heldout_pairs": pairs, "heldout_delta": delta, "issues": issues}


def run_control(cases: list[dict], prompts: dict[str, str], ask: Ask, *, journal: Path,
                policy: dict, max_tokens: int = 1024, max_workers: int = 4) -> dict:
    """Ask the deliberately bad prompt the same questions. The gate must reject it."""
    chosen = control_cases(cases)
    records = collect(chosen, {"control": prompts["control"]}, ask, journal=journal,
                      max_tokens=max_tokens, max_workers=max_workers)
    rows, issues = reconcile(chosen, records, ("control",))
    scorer = score(rows)
    metrics = summarize(rows)
    rejected = bool(not issues and metrics["accuracy"] is not None
                    and metrics["accuracy"] < policy["candidate_heldout_accuracy_min"])
    return {"prompt": prompts["control"], "rows": rows, "metrics": metrics, "scorer": scorer,
            "rejected": rejected, "case_ids": [c["case_id"] for c in chosen]}


def report_markdown(result: dict) -> str:
    d = result["decision"]
    lines = ["# Release gate report", "",
             f"Decision: **{d['decision']}**. CI exit code: **{result['ci_exit_code']}**. Model: `{result['model']}`.",
             "", "Synthetic hand-authored tickets. This gate decides whether a prompt change may proceed; it deploys nothing.", "",
             "| Variant | Split | Correct / cases | Accuracy | Complete |", "| --- | --- | --- | --- | --- |"]
    for variant, splits in d["variants"].items():
        for split, m in splits.items():
            acc = "" if m["accuracy"] is None else f"{m['accuracy']:.3f}"
            lines.append(f"| {variant} | {split} | {m['correct']} / {m['cases']} | {acc} | {m['complete']} |")
    if result.get("control"):
        c = result["control"]["metrics"]
        lines += ["", f"Negative control (deliberately bad prompt): {c['correct']} / {c['cases']} correct, "
                      f"{'rejected' if result['control']['rejected'] else 'NOT rejected'} by the gate."]
    lines += ["", "Policy checks:", ""] + [f"- {k}: {'pass' if v else 'FAIL'}" for k, v in d["checks"].items()]
    return "\n".join(lines) + "\n"
