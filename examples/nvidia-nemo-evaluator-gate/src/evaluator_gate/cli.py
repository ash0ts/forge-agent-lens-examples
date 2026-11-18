"""evaluator-gate run | replay | verify"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import DATA_DIR, VARIANTS, load_policy, load_prompts, load_tickets, sha256
from .gate import collect, decide, reconcile, report_markdown, run_control, score, stamp


def build_result(cases, prompts, policy, records, *, model, control=None, mode) -> dict:
    rows, issues = reconcile(cases, records, VARIANTS)
    scorer = score(rows)
    decision = decide(rows, issues, policy)
    if control is not None:
        decision["checks"]["negative_control_rejected"] = control["rejected"]
        decision["decision"] = "pass" if all(decision["checks"].values()) else "hold"
    exit_code = 0 if decision["decision"] == "pass" else 1
    return {"mode": mode, "created_at": stamp(), "model": model, "policy": policy, "scorer": scorer,
            "tickets_sha256": sha256({"cases": cases}), "prompts_sha256": sha256(prompts),
            "decision": decision, "rows": rows, "control": control, "ci_exit_code": exit_code,
            "limitations": [
                "Synthetic hand-authored tickets test this routing task, not general model quality.",
                "Held-out means excluded from prompt development, not secret from repository readers.",
                "The gate decides whether a prompt change may proceed. It deploys nothing.",
                "Each request is attempted once; failures stay in the denominator.",
            ]}


def write_outputs(result: dict, output_dir: Path) -> None:
    slim = {k: v for k, v in result.items()}
    (output_dir / "result.json").write_text(json.dumps(slim, indent=2, sort_keys=True, default=str))
    (output_dir / "report.md").write_text(report_markdown(result))


def cmd_run(args) -> int:
    from .forge import publish
    from .inference import ForgeInference

    data_dir = Path(args.data_dir)
    tickets = load_tickets(data_dir / "tickets.json")
    cases = tickets["cases"]
    prompts = load_prompts(data_dir / "prompts.json")
    policy = load_policy(data_dir / "policy.json")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ForgeInference(args.project, model=args.model)
    try:
        records = collect(cases, {v: prompts[v] for v in VARIANTS}, client.ask,
                          journal=output_dir / "responses.jsonl", max_tokens=args.max_tokens,
                          max_workers=args.workers)
        control = None
        if not args.skip_control:
            control = run_control(cases, prompts, client.ask, journal=output_dir / "control-responses.jsonl",
                                  policy=policy, max_tokens=args.max_tokens, max_workers=args.workers)
    finally:
        client.close()

    result = build_result(cases, prompts, policy, records, model=args.model, control=control, mode="live")
    write_outputs(result, output_dir)
    receipt = publish(result, project=args.project, run_name=args.run_name, output_dir=output_dir, data_dir=data_dir)
    print(json.dumps({"decision": result["decision"]["decision"], "ci_exit_code": result["ci_exit_code"],
                      "run_url": receipt["run_url"], "report": str(output_dir / "report.md")}, indent=2))
    return result["ci_exit_code"]


def cmd_replay(args) -> int:
    """Re-score saved responses. Makes no model calls; useful offline and in CI."""
    source = Path(args.responses)
    data_dir = Path(args.data_dir)
    cases = load_tickets(data_dir / "tickets.json")["cases"]
    prompts = load_prompts(data_dir / "prompts.json")
    policy = load_policy(data_dir / "policy.json")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    control = None
    control_path = source.with_name("control-responses.jsonl")
    if control_path.exists():
        from .data import control_cases
        from .gate import summarize
        chosen = control_cases(cases)
        c_records = [json.loads(line) for line in control_path.read_text().splitlines() if line.strip()]
        rows, issues = reconcile(chosen, c_records, ("control",))
        c_scorer = score(rows)
        metrics = summarize(rows)
        control = {"prompt": prompts["control"], "rows": rows, "metrics": metrics, "scorer": c_scorer,
                   "rejected": bool(not issues and metrics["accuracy"] is not None
                                    and metrics["accuracy"] < policy["candidate_heldout_accuracy_min"]),
                   "case_ids": [c["case_id"] for c in chosen]}
    model = next((r.get("model") for r in records if r.get("model")), "unknown")
    result = build_result(cases, prompts, policy, records, model=model, control=control, mode="replay")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(result, output_dir)
    print(report_markdown(result))
    if args.project:
        from .forge import publish
        receipt = publish(result, project=args.project, run_name=args.run_name, output_dir=output_dir, data_dir=data_dir)
        print(json.dumps(receipt, indent=2))
    return result["ci_exit_code"]


def cmd_verify(args) -> int:
    from .forge import verify
    receipt = verify(args.run)
    print(json.dumps(receipt, indent=2))
    if receipt["decision"] != "pass":
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluator-gate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Ask the model, score with NeMo Evaluator, log the gate to Forge.")
    run.add_argument("--project", required=True, help="entity/project in Forge")
    run.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B")
    run.add_argument("--output", required=True)
    run.add_argument("--run-name", default="release-gate")
    run.add_argument("--data-dir", default=str(DATA_DIR))
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--skip-control", action="store_true")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="Re-score a saved responses.jsonl offline.")
    replay.add_argument("--responses", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--data-dir", default=str(DATA_DIR))
    replay.add_argument("--project", help="Also publish the replayed gate to this Forge entity/project.")
    replay.add_argument("--run-name", default="release-gate-replay")
    replay.set_defaults(func=cmd_replay)

    verify = sub.add_parser("verify", help="Read the decision back from a Forge run (what CI checks).")
    verify.add_argument("--run", required=True, help="entity/project/run_id")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
