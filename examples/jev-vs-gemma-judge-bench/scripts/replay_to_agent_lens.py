#!/usr/bin/env python3
"""Emit scrubbed Jev-vs-Gemma judge-comparison traces to CoreWeave Forge Agent Lens
using the shipped `coreweave` Forge SDK (`coreweave.forge.agentlens.tracing`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_records() -> list[dict[str, Any]]:
    with (DATA_DIR / "turns.csv").open(encoding="utf-8") as f:
        turns = {int(r["turn"]): r for r in csv.DictReader(f)}
    gemma_rows = {
        (int(item["turn"]), str(item["judge"])): item
        for item in (
            json.loads(line)
            for line in (DATA_DIR / "gemma_labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    jev_rows = {
        (int(item["turn"]), str(item["judge"])): item
        for item in (
            json.loads(line)
            for line in (DATA_DIR / "jev_raw.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    records: list[dict[str, Any]] = []
    for idx in sorted(turns):
        t = turns[idx]
        records.append(
            {
                "turn": idx,
                "summary": t,
                "gemma_intent": gemma_rows[(idx, "intent")],
                "gemma_failure": gemma_rows[(idx, "failure")],
                "jev_intent": jev_rows[(idx, "intent")],
                "jev_failure": jev_rows[(idx, "failure")],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="wandb/agent-lens-jev-gemma-replay-v1",
        help="Destination entity/project in CoreWeave Forge Agent Lens",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Transmit spans via coreweave.forge.agentlens.tracing (default is dry run).",
    )
    parser.add_argument(
        "--log-eval-table",
        action="store_true",
        help="Also log a side-by-side wandb.EvalTable comparing Gemma and Jev across the 100 turns.",
    )
    args = parser.parse_args()

    records = load_records()[: args.limit]
    if not args.emit and not args.log_eval_table:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "project": args.project,
                    "forge_url": f"https://forge.coreweave.com/agent-lens/{args.project}",
                    "turns_planned": len(records),
                    "judge_spans_planned": len(records) * 4,
                },
                indent=2,
            )
        )
        return 0

    if args.emit:
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("Set WANDB_API_KEY before running with --emit.")

        from coreweave.forge.agentlens import tracing
        from coreweave.forge.agentlens.tracing import Message, TextPart, Usage

        tracing.init(args.project, autopatch_integrations=False)
        try:
            for rec in records:
                idx = rec["turn"]
                s = rec["summary"]
                conv_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jev-gemma-scrubbed-v2:{idx}"))
                conv = tracing.start_conversation(
                    conversation_id=conv_id,
                    conversation_name=f"scrubbed-judge-turn-{idx:03d}",
                    agent_name="insights-judge-comparison-replay",
                    model="google/gemma-4-31B-it",
                    attributes={
                        "benchmark.turn_index": idx,
                        "benchmark.failure_truth_audited": int(s["failure_truth_audited"]),
                        "benchmark.gemma_failed": int(s["gemma_failed"]),
                        "benchmark.jev_failed": int(s["jev_failed"]),
                    },
                )
                user_prompt = (
                    f"[Scrubbed turn {idx:03d}] Evaluate intent ({s['gemma_intent_categories']}) "
                    f"and failure status (audited_failed={s['failure_truth_audited']})."
                )
                turn = conv.start_turn(
                    user_message=user_prompt,
                    agent_name="insights-judge-comparison-replay",
                    agent_version="scrubbed-public-replay-v2",
                    model="google/gemma-4-31B-it",
                )
                for label, provider, model_id, payload, in_tok, out_tok in [
                    (
                        "gemma-intent",
                        "forge-serverless-inference",
                        "google/gemma-4-31B-it",
                        rec["gemma_intent"],
                        int(s["gemma_intent_input_tokens"]),
                        int(s["gemma_intent_output_tokens"]),
                    ),
                    (
                        "gemma-failure",
                        "forge-serverless-inference",
                        "google/gemma-4-31B-it",
                        rec["gemma_failure"],
                        int(s["gemma_failure_input_tokens"]),
                        int(s["gemma_failure_output_tokens"]),
                    ),
                    (
                        "jev-intent",
                        "typesafe",
                        str(rec["jev_intent"]["model"]),
                        rec["jev_intent"]["outcome"],
                        int(s["jev_intent_input_tokens"]),
                        int(s["jev_intent_output_tokens"]),
                    ),
                    (
                        "jev-failure",
                        "typesafe",
                        str(rec["jev_failure"]["model"]),
                        rec["jev_failure"]["outcome"],
                        int(s["jev_failure_input_tokens"]),
                        int(s["jev_failure_output_tokens"]),
                    ),
                ]:
                    llm = turn.start_llm(model=model_id, provider_name=provider)
                    llm.set_attributes({"benchmark.judge_stage": label})
                    llm.record(
                        input_messages=[Message(role="user", parts=[TextPart(content=user_prompt)])],
                        output_messages=[
                            Message(role="assistant", parts=[TextPart(content=json.dumps(payload, sort_keys=True))])
                        ],
                        usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
                        response_model=model_id,
                    )
                    llm.end()

                turn.record(
                    output_messages=[
                        Message(
                            role="assistant",
                            parts=[
                                TextPart(
                                    content=(
                                        f"Gemma failed={s['gemma_failed']}, "
                                        f"Jev failed={s['jev_failed']} (max_p={max(float(s['jev_failed_p']), float(s['jev_tool_error_p'])):.3f}), "
                                        f"audited={s['failure_truth_audited']}."
                                    )
                                )
                            ],
                        )
                    ]
                )
                turn.end()
                conv.end()
            tracing.force_flush()
        finally:
            tracing.shutdown()

    eval_run_urls: dict[str, str] = {}
    if args.log_eval_table:
        import tempfile
        import wandb

        ent, proj_name = args.project.split("/", 1)
        columns = [
            "turn",
            "audited_failed",
            "agent_lens_url",
            "predicted_failed",
            "primary_intent",
            "max_fail_prob",
            "correct_failure_decision",
            "latency_s",
            "cost_usd",
        ]
        variant_rows: dict[str, list[list[object]]] = {
            "gemma-4-31B-it": [],
            "jev-1.13.0": [],
        }
        for rec in records:
            idx = rec["turn"]
            s = rec["summary"]
            conv_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jev-gemma-scrubbed-v2:{idx}"))
            lens_url = f"https://forge.coreweave.com/agent-lens/{args.project}/conversations/{conv_id}"
            truth = int(s["failure_truth_audited"])
            g_pred = int(s["gemma_failed"])
            j_pred = int(s["jev_failed"])
            g_cost = (
                (int(s["gemma_intent_input_tokens"]) + int(s["gemma_failure_input_tokens"])) * 0.10 / 1_000_000
                + (int(s["gemma_intent_output_tokens"]) + int(s["gemma_failure_output_tokens"])) * 0.34 / 1_000_000
            )
            j_cost = (int(s["jev_intent_input_tokens"]) + int(s["jev_failure_input_tokens"])) * 42.0 / 1_000_000_000
            j_lat = float(s["jev_intent_latency_s"]) + float(s["jev_failure_latency_s"])
            variant_rows["gemma-4-31B-it"].append(
                [idx, bool(truth), lens_url, bool(g_pred), s["gemma_intent_categories"], None, g_pred == truth, None, round(g_cost, 6)]
            )
            variant_rows["jev-1.13.0"].append(
                [
                    idx,
                    bool(truth),
                    lens_url,
                    bool(j_pred),
                    s["jev_primary_category"],
                    round(max(float(s["jev_failed_p"]), float(s["jev_tool_error_p"])), 4),
                    j_pred == truth,
                    round(j_lat, 4),
                    round(j_cost, 6),
                ]
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            for variant_name, rows in variant_rows.items():
                with wandb.init(
                    entity=ent,
                    project=proj_name,
                    name=f"{variant_name}-insights-judge",
                    job_type="judge-benchmark",
                    dir=tmpdir,
                    config={"judge_variant": variant_name, "turns": len(rows)},
                ) as run:
                    et = wandb.EvalTable(
                        columns=columns,
                        input_columns=["turn", "audited_failed", "agent_lens_url"],
                        score_columns=["correct_failure_decision", "latency_s", "cost_usd"],
                        data=rows,
                    )
                    run.log({"insights_judge_eval": et})
                    eval_run_urls[variant_name] = run.url

    print(
        json.dumps(
            {
                "mode": "emitted" if args.emit else "eval_table_logged",
                "project": args.project,
                "forge_url": f"https://forge.coreweave.com/agent-lens/{args.project}",
                "eval_run_urls": eval_run_urls,
                "turns_processed": len(records),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
