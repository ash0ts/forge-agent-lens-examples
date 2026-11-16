#!/usr/bin/env python3
"""Reproduce every failure-detection, taxonomy-accuracy, latency, and cost number in
'We Benchmarked Jev Against Gemma Inside Agent Lens' from the scrubbed public CSV/JSONL records,
and optionally log an Evaluation Table (wandb.EvalTable) to a CoreWeave Forge Workspace.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

GEMMA_INPUT_PRICE = 0.10 / 1e6
GEMMA_OUTPUT_PRICE = 0.34 / 1e6
JEV_INPUT_PRICE = 42.0 / 1e9
JEV_OUTPUT_PRICE = 0.0
GEMMA_SINGLE_CALL_ESTIMATE_S = 8.0


def load_turns() -> list[dict[str, str]]:
    with (DATA_DIR / "turns.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_audit_calls() -> list[dict[str, str]]:
    with (DATA_DIR / "audit_calls.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def confusion(pred_flags: list[int], truth_flags: list[int]) -> dict[str, Any]:
    tp = sum(1 for p, t in zip(pred_flags, truth_flags) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_flags, truth_flags) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred_flags, truth_flags) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(pred_flags, truth_flags) if p == 0 and t == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_summary() -> dict[str, Any]:
    turns = load_turns()
    audit_calls = load_audit_calls()
    jev_raw = load_jsonl(DATA_DIR / "jev_raw.jsonl")

    truth = [int(r["failure_truth_audited"]) for r in turns]
    gemma_pred = [int(r["gemma_failed"]) for r in turns]
    jev_05 = [
        1 if max(float(r["jev_failed_p"]), float(r["jev_tool_error_p"])) >= 0.5 else 0
        for r in turns
    ]
    jev_08 = [
        1 if max(float(r["jev_failed_p"]), float(r["jev_tool_error_p"])) >= 0.8 else 0
        for r in turns
    ]

    # Single-choice agreement rates
    n = len(turns)
    agreement = {
        "status": sum(int(r["agree_status"]) for r in turns) / n,
        "primary_in_gemma_set": sum(int(r["agree_primary_in_gemma_set"]) for r in turns) / n,
        "sentiment": sum(int(r["agree_sentiment"]) for r in turns) / n,
        "target": sum(int(r["agree_target"]) for r in turns) / n,
        "language": sum(int(r["agree_language"]) for r in turns) / n,
        "failed": sum(int(r["agree_failed"]) for r in turns) / n,
    }

    # Adjudicated taxonomy field scores (agreements + ties credited to both)
    audit_by_field: dict[str, list[str]] = {
        "status": [],
        "intents": [],
        "sentiment": [],
        "language": [],
        "failure category": [],
    }
    for row in audit_calls:
        raw_field = row["field"].strip().lower()
        winner = row["winner"].strip()
        if "status" in raw_field:
            audit_by_field["status"].append(winner)
        if "intents" in raw_field:
            audit_by_field["intents"].append(winner)
        if "sentiment" in raw_field or "target" in raw_field:
            audit_by_field["sentiment"].append(winner)
        if "language" in raw_field:
            audit_by_field["language"].append(winner)
        if raw_field == "failure category":
            audit_by_field["failure category"].append(winner)

    def field_score(total: int, winners: list[str]) -> dict[str, int]:
        gemma_losses = sum(1 for w in winners if w == "Jev")
        jev_losses = sum(1 for w in winners if w == "Gemma")
        return {"gemma": total - gemma_losses, "jev": total - jev_losses, "total": total}

    both_fired_count = sum(1 for g, j in zip(gemma_pred, jev_05) if g == 1 and j == 1)
    taxonomy_scores = {
        "turn_status": field_score(100, audit_by_field["status"]),
        "intent_category": field_score(100, audit_by_field["intents"]),
        "sentiment_and_target": field_score(100, audit_by_field["sentiment"]),
        "language": field_score(100, audit_by_field["language"]),
        "failure_category_when_both_fired": field_score(
            both_fired_count, audit_by_field["failure category"]
        ),
    }

    # Latency & cost across all 200 calls per judge
    jev_latencies = sorted(float(item["latency_s"]) for item in jev_raw)
    jev_p50 = statistics.median(jev_latencies)
    jev_p95 = jev_latencies[int(round(0.95 * (len(jev_latencies) - 1)))]
    jev_max = max(jev_latencies)

    jev_in_tokens = sum(
        int(r["jev_intent_input_tokens"]) + int(r["jev_failure_input_tokens"]) for r in turns
    )
    jev_out_tokens = sum(
        int(r["jev_intent_output_tokens"]) + int(r["jev_failure_output_tokens"]) for r in turns
    )
    gemma_in_tokens = sum(
        int(r["gemma_intent_input_tokens"]) + int(r["gemma_failure_input_tokens"]) for r in turns
    )
    gemma_out_tokens = sum(
        int(r["gemma_intent_output_tokens"]) + int(r["gemma_failure_output_tokens"]) for r in turns
    )

    jev_cost = jev_in_tokens * JEV_INPUT_PRICE + jev_out_tokens * JEV_OUTPUT_PRICE
    gemma_cost = gemma_in_tokens * GEMMA_INPUT_PRICE + gemma_out_tokens * GEMMA_OUTPUT_PRICE

    return {
        "turns": n,
        "calls_per_judge": len(jev_raw),
        "reviewed_failures": sum(truth),
        "agreement_rates": {k: round(v, 4) for k, v in agreement.items()},
        "failure_detection": {
            "gemma": confusion(gemma_pred, truth),
            "jev_threshold_0_5": confusion(jev_05, truth),
            "jev_threshold_0_8": confusion(jev_08, truth),
        },
        "taxonomy_scores": taxonomy_scores,
        "latency_seconds": {
            "jev_p50": round(jev_p50, 4),
            "jev_p95": round(jev_p95, 4),
            "jev_max": round(jev_max, 4),
            "gemma_concurrent_estimate_s": GEMMA_SINGLE_CALL_ESTIMATE_S,
            "estimated_speedup_vs_jev_p50": round(GEMMA_SINGLE_CALL_ESTIMATE_S / jev_p50, 1),
        },
        "cost_usd": {
            "jev_200_calls": round(jev_cost, 6),
            "gemma_200_calls": round(gemma_cost, 6),
            "cost_ratio_gemma_over_jev": round(gemma_cost / jev_cost, 2),
        },
    }


def log_to_forge_workspace(project_path: str, summary: dict[str, Any]) -> str:
    import wandb

    entity, project = project_path.split("/", 1)
    turns = load_turns()
    run = wandb.init(
        entity=entity,
        project=project,
        name="jev-vs-gemma-100-turn-audit",
        job_type="judge-benchmark",
        config={
            "gemma_model": "google/gemma-4-31B-it",
            "jev_model": "jev-1.13.0",
            "turns": summary["turns"],
            "jev_threshold": 0.5,
        },
    )
    columns = [
        "turn",
        "failure_truth_audited",
        "gemma_failed",
        "jev_failed_at_0_5",
        "jev_max_failure_prob",
        "gemma_intent_categories",
        "jev_primary_category",
        "gemma_sentiment",
        "jev_sentiment",
        "jev_total_latency_s",
    ]
    table_cls = getattr(wandb, "EvalTable", wandb.Table)
    table = table_cls(columns=columns)
    for r in turns:
        max_p = round(max(float(r["jev_failed_p"]), float(r["jev_tool_error_p"])), 4)
        tot_lat = round(float(r["jev_intent_latency_s"]) + float(r["jev_failure_latency_s"]), 4)
        table.add_data(
            int(r["turn"]),
            int(r["failure_truth_audited"]),
            int(r["gemma_failed"]),
            int(r["jev_failed"]),
            max_p,
            r["gemma_intent_categories"],
            r["jev_primary_category"],
            r["gemma_sentiment"],
            r["jev_sentiment"],
            tot_lat,
        )
    run.log({"judge_comparison_turns": table})
    run.summary.update(
        {
            "gemma_precision": summary["failure_detection"]["gemma"]["precision"],
            "gemma_recall": summary["failure_detection"]["gemma"]["recall"],
            "jev_0_5_precision": summary["failure_detection"]["jev_threshold_0_5"]["precision"],
            "jev_0_5_recall": summary["failure_detection"]["jev_threshold_0_5"]["recall"],
            "jev_p50_latency_s": summary["latency_seconds"]["jev_p50"],
            "jev_cost_usd": summary["cost_usd"]["jev_200_calls"],
            "gemma_cost_usd": summary["cost_usd"]["gemma_200_calls"],
        }
    )
    url = f"https://forge.coreweave.com/wandb/{entity}/{project}/runs/{run.id}"
    run.finish()
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-forge-project",
        help="Optional entity/project on CoreWeave Forge to log the comparison EvalTable.",
    )
    args = parser.parse_args()
    summary = compute_summary()
    if args.log_forge_project:
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("Set WANDB_API_KEY before logging to CoreWeave Forge.")
        summary["forge_run_url"] = log_to_forge_workspace(args.log_forge_project, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
