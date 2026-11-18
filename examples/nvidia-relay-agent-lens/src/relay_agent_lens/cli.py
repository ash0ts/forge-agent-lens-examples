"""relay-agent-lens: run the Relay-instrumented SQL agent into Agent Lens, publish the companion run, verify."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .agent import run_agent
from .inference import DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relay-agent-lens")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run three SQL tasks (one with an injected tool timeout) and export spans.")
    run.add_argument("--project", default=None, help="entity/project for Agent Lens and the companion run. "
                                                      "Omit to export only to the local gen_ai JSONL file.")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--output", type=Path, default=Path("out/run"))
    run.add_argument("--run-name", default=None)
    run.add_argument("--fault-task", default="largest_paid_order")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--skip-publish", action="store_true", help="Export to Agent Lens but skip the W&B run.")

    ver = sub.add_parser("verify", help="Read a conversation back from Agent Lens.")
    ver.add_argument("--project", required=True)
    ver.add_argument("--conversation", required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = asyncio.run(run_agent(args.output, project=args.project, model=args.model,
                                       fault_task=args.fault_task, seed=args.seed))
        print(json.dumps({"metrics": result["metrics"], "spans_exported": result["span_records"],
                          "agent_lens_url": result["agent_lens_url"]}, indent=2))
        if args.project and not args.skip_publish:
            from .forge import publish
            info = publish(result, args.output, project=args.project, run_name=args.run_name)
            print(json.dumps(info, indent=2))
        return 0 if result["metrics"]["correct"] == result["metrics"]["tasks"] else 1
    if args.command == "verify":
        from .forge import verify
        report = verify(args.project, args.conversation)
        print(json.dumps(report, indent=2))
        return 0 if report["spans_in_agent_lens"] > 0 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
