from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import require_hex_digest
from .emitter import emit_frozen_to_agent_lens, plan_frozen_emission


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retail-support-study")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--tau-root", type=Path, required=True)
    sub.add_parser("probe-models")
    sub.add_parser("preview")
    sub.add_parser("verify")

    run = sub.add_parser("run")
    run.add_argument("--cohort", required=True)
    run.add_argument("--approve-preview-sha", required=True)
    run.add_argument("--max-new-coordinates", type=int)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument(
        "--live-agent-lens",
        action="store_true",
        help=(
            "emit real-time support-agent and MCP spans to the fixed Agent Lens "
            "project bound by the approved preview"
        ),
    )

    review = sub.add_parser("review-canary")
    review.add_argument("--verify", action="store_true")

    verify_run = sub.add_parser("verify-cohort")
    verify_run.add_argument("--cohort", choices=("canary", "full"), required=True)
    verify_run.add_argument("--preview-sha")

    score = sub.add_parser("score-cohort")
    score.add_argument("--cohort", choices=("canary", "full"), required=True)
    score.add_argument("--preview-sha")

    emission_plan = sub.add_parser("plan-emission")
    emission_plan.add_argument("--ledger-path", type=Path, required=True)

    emit = sub.add_parser("emit")
    emit.add_argument("--ledger-path", type=Path, required=True)
    emit.add_argument("--approve-ledger-sha", required=True)
    emit.add_argument("--approve-payload-sha", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .remap import prepare_source

        _print(prepare_source(args.tau_root))
    elif args.command == "probe-models":
        from .model_probe import probe_model_routes

        result = probe_model_routes()
        _print(result)
        if result["status"] != "ready_for_preview":
            return 2
    elif args.command == "preview":
        from .preview import write_preview

        _print(write_preview())
    elif args.command == "verify":
        from .verify import verify_static

        _print(verify_static())
    elif args.command == "run":
        from .runner import run_cohort

        preview_sha = require_hex_digest(
            args.approve_preview_sha, label="approve-preview-sha"
        )
        run_kwargs = {
            "approve_preview_sha": preview_sha,
            "max_new_coordinates": args.max_new_coordinates,
            "concurrency": args.concurrency,
        }
        if args.live_agent_lens:
            run_kwargs["live_agent_lens"] = True
        _print(run_cohort(args.cohort, **run_kwargs).model_dump())
    elif args.command == "review-canary":
        from .review import build_canary_review_template, verify_canary_review

        _print(
            verify_canary_review() if args.verify else build_canary_review_template()
        )
    elif args.command == "verify-cohort":
        from .verify import verify_cohort

        if args.preview_sha is None:
            _print(verify_cohort(args.cohort))
        else:
            preview_sha = require_hex_digest(args.preview_sha, label="preview-sha")
            _print(verify_cohort(args.cohort, preview_sha256=preview_sha))
    elif args.command == "score-cohort":
        from .scoring import score_cohort

        if args.preview_sha is None:
            _print(score_cohort(args.cohort))
        else:
            preview_sha = require_hex_digest(args.preview_sha, label="preview-sha")
            _print(score_cohort(args.cohort, preview_sha256=preview_sha))
    elif args.command == "plan-emission":
        _print(plan_frozen_emission(ledger_path=args.ledger_path))
    elif args.command == "emit":
        _print(
            emit_frozen_to_agent_lens(
                ledger_path=args.ledger_path,
                approve_ledger_sha=args.approve_ledger_sha,
                approve_payload_sha=args.approve_payload_sha,
            )
        )
    else:  # pragma: no cover
        parser.error("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
