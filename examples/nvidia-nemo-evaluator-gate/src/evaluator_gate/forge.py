"""Put the gate in Forge: one EvalTable to read, a policy table, lineage artifacts, a summary CI can trust."""
from __future__ import annotations

import json
from pathlib import Path

import wandb

from .data import LABELS, VARIANTS

INPUT_COLUMNS = ["case_id", "group", "ticket", "expected"]
OUTPUT_COLUMNS = ["baseline_output", "candidate_output", "control_output"]
SCORE_COLUMNS = ["baseline_correct", "candidate_correct", "control_correct", "regression", "improvement"]


def _heldout_rows(result: dict) -> list[list]:
    by_id = {}
    for row in result["rows"]:
        if row["split"] != "heldout":
            continue
        entry = by_id.setdefault(row["case_id"], {"case_id": row["case_id"], "group": row["group"],
                                                  "ticket": row["prompt"], "expected": row["expected"]})
        entry[f"{row['variant']}_output"] = row.get("output", "")
        entry[f"{row['variant']}_correct"] = row.get("correct")
        entry[f"{row['variant']}_state"] = row.get("state")
    if result.get("control"):
        for row in result["control"]["rows"]:
            entry = by_id.get(row["case_id"])
            if entry is not None:
                entry["control_output"] = row.get("output", "")
                entry["control_correct"] = row.get("correct")
    pairs = {p["case_id"]: p for p in result["decision"]["heldout_pairs"]}
    rows = []
    for case_id in sorted(by_id):
        e = by_id[case_id]
        p = pairs.get(case_id, {})
        rows.append([e["case_id"], e["group"], e["ticket"], e["expected"],
                     e.get("baseline_output", ""), e.get("candidate_output", ""), e.get("control_output"),
                     e.get("baseline_correct"), e.get("candidate_correct"), e.get("control_correct"),
                     p.get("regression", False), p.get("improvement", False)])
    return rows


def publish(result: dict, *, project: str, run_name: str, output_dir: Path, data_dir: Path) -> dict:
    entity, project_name = project.split("/", 1)
    decision = result["decision"]
    run = wandb.init(entity=entity, project=project_name, name=run_name, job_type="release-gate",
                     config={"model": result["model"], "policy": result["policy"],
                             "tickets_sha256": result["tickets_sha256"], "prompts_sha256": result["prompts_sha256"],
                             "scorer": result["scorer"]},
                     settings=wandb.Settings(init_timeout=60))
    try:
        inputs = wandb.Artifact("release-gate-inputs", type="gate-inputs",
                                description="Frozen tickets, prompts and policy the gate was run against.",
                                metadata={"tickets_sha256": result["tickets_sha256"], "prompts_sha256": result["prompts_sha256"]})
        for name in ("tickets.json", "prompts.json", "policy.json"):
            inputs.add_file(str(data_dir / name))
        run.log_artifact(inputs)

        heldout = wandb.EvalTable(columns=INPUT_COLUMNS + OUTPUT_COLUMNS + SCORE_COLUMNS,
                                  data=_heldout_rows(result), input_columns=INPUT_COLUMNS,
                                  output_columns=OUTPUT_COLUMNS, score_columns=SCORE_COLUMNS)
        run.log({"release_gate/heldout": heldout})

        every = wandb.Table(columns=["case_id", "split", "group", "variant", "ticket", "expected", "output",
                                     "state", "correct", "latency_ms", "finish_reason"],
                            data=[[r["case_id"], r["split"], r["group"], r["variant"], r["prompt"], r["expected"],
                                   r.get("output", ""), r["state"], r.get("correct"), r.get("latency_ms"),
                                   r.get("finish_reason")] for r in result["rows"]])
        run.log({"release_gate/all_responses": every})

        checks = wandb.Table(columns=["check", "result"],
                             data=[[k, "pass" if v else "FAIL"] for k, v in decision["checks"].items()])
        run.log({"release_gate/policy_checks": checks})

        categories = wandb.Table(columns=["group", "variant", "correct", "cases", "accuracy", "floor"],
                                 data=[[g, v, m["correct"], m["cases"], m["accuracy"], result["policy"]["group_accuracy_min"]]
                                       for g, variants in decision["heldout_groups"].items() for v, m in variants.items()])
        run.log({"release_gate/category_accuracy": categories})
        for v in VARIANTS:
            run.log({f"release_gate/category_bar/{v}": wandb.plot.bar(
                wandb.Table(columns=["group", "accuracy"],
                            data=[[g, decision["heldout_groups"][g][v]["accuracy"]] for g in LABELS]),
                "group", "accuracy", title=f"Held-out accuracy by category: {v}")})

        regressions = wandb.Table(columns=["case_id", "group", "baseline_correct", "candidate_correct", "kind"],
                                  data=[[p["case_id"], p["group"], p["baseline_correct"], p["candidate_correct"],
                                         "regression" if p["regression"] else "improvement"]
                                        for p in decision["heldout_pairs"] if p["regression"] or p["improvement"]])
        run.log({"release_gate/paired_changes": regressions})

        summary = {"gate/decision": decision["decision"], "gate/ci_exit_code": result["ci_exit_code"],
                   "gate/heldout_delta": decision["heldout_delta"],
                   "gate/paired_regressions": sum(p["regression"] for p in decision["heldout_pairs"]),
                   "gate/paired_improvements": sum(p["improvement"] for p in decision["heldout_pairs"])}
        for k, v in decision["checks"].items():
            summary[f"gate/check/{k}"] = bool(v)
        for v in VARIANTS:
            for split, m in decision["variants"][v].items():
                summary[f"{v}/{split}/accuracy"] = m["accuracy"]
                summary[f"{v}/{split}/correct"] = m["correct"]
                summary[f"{v}/{split}/cases"] = m["cases"]
            for g in LABELS:
                summary[f"{v}/heldout/{g}/accuracy"] = decision["heldout_groups"][g][v]["accuracy"]
        if result.get("control"):
            summary["control/accuracy"] = result["control"]["metrics"]["accuracy"]
            summary["control/correct"] = result["control"]["metrics"]["correct"]
            summary["control/cases"] = result["control"]["metrics"]["cases"]
            summary["gate/check/negative_control_rejected"] = result["control"]["rejected"]
        run.summary.update(summary)

        evidence = wandb.Artifact("release-gate-evidence", type="gate-evidence",
                                  description="Every response, the scorer verdicts and the decision.")
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
                evidence.add_file(str(path))
        run.log_artifact(evidence)
        url, run_id = run.url, run.id
    finally:
        run.finish()
    receipt = {"run_url": url, "run_path": f"{project}/{run_id}", "decision": decision["decision"]}
    (output_dir / "forge-run.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def verify(run_path: str) -> dict:
    """What CI reads: the decision as Forge stored it, not a local file."""
    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    summary = dict(run.summary)
    return {"run_path": run_path, "decision": summary.get("gate/decision"),
            "ci_exit_code": summary.get("gate/ci_exit_code"),
            "checks": {k.removeprefix("gate/check/"): v for k, v in summary.items() if k.startswith("gate/check/")}}
