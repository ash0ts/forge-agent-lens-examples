from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import TextIO

from .behavior import RuntimeBehaviorProfile
from .constants import PREPARED_DIR
from .mcp_protocol import RetailMCPProtocol, parse_error_response

AUTHENTICATED_USER_ENV = "RETAIL_SUPPORT_AUTHENTICATED_USER_ID"


def serve_stdio(
    protocol: RetailMCPProtocol,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    """Serve newline-delimited MCP JSON-RPC without writing logs to stdout."""

    for line in input_stream:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = parse_error_response()
        else:
            response = protocol.handle_message(message)
        if response is None:
            continue
        output_stream.write(
            json.dumps(
                response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        output_stream.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="τ³ retail MCP stdio server")
    parser.add_argument(
        "--db",
        type=Path,
        default=PREPARED_DIR / "db.json",
        help="prepared, remapped RetailDB JSON",
    )
    parser.add_argument(
        "--authenticated-user-id",
        default=os.environ.get(AUTHENTICATED_USER_ENV),
        help=(
            "optional pre-authenticated session user "
            f"(or ${AUTHENTICATED_USER_ENV}); otherwise identity lookup binds it"
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "identifier for this stdio MCP process/session; generated when "
            "the server is launched directly"
        ),
    )
    parser.add_argument(
        "--allowed-tool",
        action="append",
        default=None,
        help="tool exposed by this specialist; repeat for each allowed tool",
    )
    parser.add_argument(
        "--behavior-profile-json",
        help="reviewed latency and fault profile as canonical JSON",
    )
    parser.add_argument(
        "--conversation-id",
        help="public conversation identifier used for deterministic behavior",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database does not exist: {args.db}")

    behavior_profile = None
    if args.behavior_profile_json is not None:
        try:
            raw_behavior = json.loads(args.behavior_profile_json)
        except json.JSONDecodeError as exc:
            parser.error(f"behavior profile is not valid JSON: {exc}")
        if not isinstance(raw_behavior, dict):
            parser.error("behavior profile must be a JSON object")
        try:
            behavior_profile = RuntimeBehaviorProfile.from_mapping(raw_behavior)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"behavior profile is invalid: {exc}")
        if not args.conversation_id:
            parser.error("--conversation-id is required with a behavior profile")

    protocol = RetailMCPProtocol.from_db_path(
        args.db,
        authenticated_user_id=args.authenticated_user_id,
        session_id=args.session_id or str(uuid.uuid4()),
        allowed_tool_names=args.allowed_tool,
        behavior_profile=behavior_profile,
        conversation_id=args.conversation_id,
    )
    serve_stdio(
        protocol,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
