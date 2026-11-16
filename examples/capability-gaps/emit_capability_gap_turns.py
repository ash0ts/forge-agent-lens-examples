#!/usr/bin/env python3
"""Plan or emit a frozen Agent Lens capability-boundary corpus using the
CoreWeave Forge SDK (`coreweave.forge.agentlens.tracing`).

Supports both the historical v1 180-turn manifest (`corpus_manifest.json`) and
the expanded v2 240-turn manifest (`corpus_manifest_v2.json`) whose distinct
operational failure phrasings satisfy Agent Lens Insights HDBSCAN clustering
thresholds (`>=100` signatures in window, `>=15` distinct wordings per category).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "corpus_manifest.json"
V2_MANIFEST = SCRIPT_DIR / "corpus_manifest_v2.json"
EXPECTED_PROJECT = "wandb/agent-lens-capability-gaps-demo"
AGENT_NAME = "training-operations-review-agent"
AGENT_VERSION = "staged-capability-boundary-v2"
MODEL_NAME = "deterministic-script-no-model-call"
_BATCH_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]


@dataclass(frozen=True)
class Scenario:
    user_message: str
    tool_calls: tuple[ToolCall, ...]
    final_answer: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _messages(role: str, content: str) -> str:
    return json.dumps(
        [{"role": role, "parts": [{"type": "text", "content": content}]}],
        separators=(",", ":"),
    )


def _digest_path_for(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".sha256")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    digest_file = _digest_path_for(path)
    if not digest_file.is_file():
        raise ValueError(f"missing digest file for manifest: {digest_file}")
    expected = digest_file.read_text().strip()
    if digest != expected:
        raise ValueError(
            f"manifest digest drift: expected {expected}, observed {digest}"
        )
    manifest = json.loads(raw)
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported manifest schema")
    if manifest.get("destination_project") != EXPECTED_PROJECT:
        raise ValueError("manifest destination does not match the fixed demo project")
    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("manifest must contain scenario families")
    family_ids = [item.get("id") for item in families]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("manifest family IDs must be unique")
    total = sum(int(item.get("count", 0)) for item in families)
    if total != manifest.get("planned_turns") or total <= 0:
        raise ValueError("manifest family counts do not sum to planned_turns")
    return manifest, digest


def _job(index: int) -> str:
    return f"demo-train-{100 + index:03d}"


def _worker(index: int) -> str:
    return f"worker-{2 + index % 7:02d}"


def _diagnostic_calls(index: int) -> tuple[ToolCall, ToolCall]:
    job = _job(index)
    worker = _worker(index)
    elapsed = 14 + index % 11
    return (
        ToolCall(
            "get_training_job",
            {"environment": "production", "job": job},
            {
                "status": "ok",
                "job": job,
                "state": "running",
                "minutes_without_progress": elapsed,
            },
        ),
        ToolCall(
            "get_worker_health",
            {"environment": "production", "job": job, "window": "15m"},
            {
                "status": "ok",
                "job": job,
                "unhealthy_worker": worker,
                "last_heartbeat_minutes_ago": elapsed,
                "healthy_workers": 7,
            },
        ),
    )


_V2_LIMITATION_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Production job {job} has stalled on NCCL collective sync. Check which worker dropped out and restart that worker pod.",
        "{worker} missed heartbeats during NCCL collective sync while 7 peers stayed healthy. This read-only diagnostics agent has no pod-restart tool; pass {worker} on {job} to the on-call operator.",
    ),
    (
        "Training run {job} is hung after checkpoint upload. Identify the stuck replica and drain its GPU node.",
        "{worker} stopped reporting after checkpoint upload. Draining GPU nodes is outside this agent's read-only capability; open a node-drain request for {worker} on {job}.",
    ),
    (
        "Job {job} lost step throughput in production. Find the wedged worker rank and cordoning-evict it.",
        "Rank on {worker} has made zero step progress for 18 minutes. This agent cannot cordon or evict Kubernetes nodes; escalate {worker} ({job}) with the health output above.",
    ),
    (
        "Our distributed SFT job {job} is frozen. Locate the unresponsive worker and trigger a container recycle.",
        "{worker} is unresponsive while the rest of {job} is healthy. Container recycling is not exposed to this read-only assistant; hand the {worker} diagnostic summary to cluster ops.",
    ),
    (
        "Production run {job} hit an IB fabric timeout on one rank. Isolate the bad worker and reschedule it onto a fresh host.",
        "{worker} stopped heartbeating after the IB fabric timeout. Rescheduling workers onto new hosts requires write permissions this agent lacks; forward {job}/{worker} to on-call.",
    ),
    (
        "Check why {job} stopped advancing global_step and reboot the stalled worker instance.",
        "{worker} is the stalled instance blocking global_step on {job}. Instance reboots are not supported in this read-only agent; submit a scoped reboot ticket for {worker}.",
    ),
    (
        "Find the dead worker holding up barrier sync on {job} and replace that pod from the warm pool.",
        "{worker} is holding up barrier sync on {job}. Replacing pods from the warm pool exceeds this agent's read-only toolset; route {worker} to the training operator.",
    ),
    (
        "Inspect {job} for Xid GPU errors and terminate the affected worker container.",
        "{worker} stopped reporting health while 7 peers remain active on {job}. Terminating containers is unsupported by this read-only agent; share {worker} with infrastructure on-call.",
    ),
    (
        "Data-loader prefetch froze on {job}. Identify the hung worker and send SIGTERM to restart its process group.",
        "{worker} has not sent a heartbeat in over 15 minutes on {job}. Sending signals or restarting worker processes is not a capability this agent has.",
    ),
    (
        "Gradient all-reduce timed out in {job}. Pinpoint the straggler worker and swap it out.",
        "{worker} is the straggler blocking all-reduce on {job}. Swapping active production workers is unsupported here; use the attached {worker} check to file an operator action.",
    ),
)

_V2_BARE_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Roll {job} back to its previous checkpoint and resume training immediately.",
        "I only have read-only inspection tools and cannot roll back checkpoints or resume {job}.",
    ),
    (
        "Scale production job {job} from 8 workers down to 6 healthy workers so it stops hanging.",
        "I cannot resize worker replicas or mutate the spec of running job {job}.",
    ),
    (
        "Patch the learning-rate schedule environment variable on live job {job} and restart rank {worker}.",
        "Modifying live job environment variables or restarting {worker} is unsupported by this agent.",
    ),
    (
        "Quarantine the faulty GPU host behind {worker} on {job} so the scheduler stops placing pods there.",
        "I cannot quarantine hosts or update cluster scheduler taints for {job}.",
    ),
    (
        "Flush the Redis dataloader lease for {job} and bounce stalled worker {worker}.",
        "Flushing external Redis leases and bouncing {worker} are write operations this agent cannot perform.",
    ),
    (
        "Force-checkpoint {job} right now and kill stuck worker {worker}.",
        "Triggering on-demand checkpoints or killing {worker} is outside what this read-only agent can execute.",
    ),
    (
        "Migrate {job} to the backup GPU namespace because {worker} stopped heartbeating.",
        "Namespace migration is not supported; I can only read current job and worker status for {job}.",
    ),
    (
        "Clear the stuck NCCL communicator lock on {job} and relaunch {worker}.",
        "I cannot clear runtime locks or relaunch {worker} on production job {job}.",
    ),
    (
        "Promote {job}'s penultimate checkpoint to Registry and stop the stalled run.",
        "Promoting checkpoints and stopping production runs are unsupported mutations in this agent.",
    ),
    (
        "Rebind {worker} to a spare NIC on {job} and restart its training loop.",
        "Network interface rebinding and training-loop restarts are capabilities this agent does not possess.",
    ),
)


def _helpful_read_only(index: int, schema_version: int = 1) -> Scenario:
    job, worker = _job(index), _worker(index)
    if schema_version >= 2:
        q_tmpl, a_tmpl = _V2_LIMITATION_TEMPLATES[index % len(_V2_LIMITATION_TEMPLATES)]
        return Scenario(
            user_message=q_tmpl.format(job=job, worker=worker),
            tool_calls=_diagnostic_calls(index),
            final_answer=a_tmpl.format(job=job, worker=worker),
        )
    action = "restart" if index % 2 == 0 else "reschedule"
    return Scenario(
        user_message=(
            f"Production job {job} has stopped making progress. Find the unhealthy "
            f"worker and {action} only that worker if the evidence supports it."
        ),
        tool_calls=_diagnostic_calls(index),
        final_answer=(
            f"{worker} has stopped reporting while the other workers remain healthy. "
            f"This agent can inspect production state but cannot {action} workers. "
            f"Send a scoped {action} request for {worker} on {job} to the on-call "
            "operator, with the evidence above attached."
        ),
    )


def _bare_limit(index: int, schema_version: int = 1) -> Scenario:
    job, worker = _job(index), _worker(index)
    if schema_version >= 2:
        q_tmpl, a_tmpl = _V2_BARE_TEMPLATES[index % len(_V2_BARE_TEMPLATES)]
        return Scenario(
            user_message=q_tmpl.format(job=job, worker=worker),
            tool_calls=_diagnostic_calls(index),
            final_answer=a_tmpl.format(job=job, worker=worker),
        )
    action = "restart" if index % 3 else "reschedule"
    return Scenario(
        user_message=(
            f"Inspect stalled production job {job}, then {action} its unhealthy worker."
        ),
        tool_calls=_diagnostic_calls(index),
        final_answer=f"I cannot {action} {worker} or change the state of {job}.",
    )


def _supported_diagnostic(index: int, schema_version: int = 1) -> Scenario:
    del schema_version
    job, worker = _job(index), _worker(index)
    return Scenario(
        user_message=(
            f"Why has production job {job} stopped making progress? Inspect only the "
            "job and worker evidence; do not change anything."
        ),
        tool_calls=_diagnostic_calls(index),
        final_answer=(
            f"{worker} has missed heartbeats while seven peers remain healthy, which "
            f"explains why {job} is still running without progress. No state was changed."
        ),
    )


def _false_gap(index: int, schema_version: int = 1) -> Scenario:
    del schema_version
    job, worker = _job(index), _worker(index)
    calls = _diagnostic_calls(index) + (
        ToolCall(
            "list_supported_actions",
            {"environment": "production", "job": job},
            {
                "status": "ok",
                "available_actions": ["request_worker_restart"],
                "approval_required": True,
            },
        ),
    )
    return Scenario(
        user_message=(
            f"If the evidence identifies one unhealthy worker on {job}, create the "
            "supported scoped restart request for review."
        ),
        tool_calls=calls,
        final_answer=(
            f"{worker} appears unhealthy, but I cannot create a restart request for {job}. "
            "An operator will have to start over manually."
        ),
    )


def _permission_bound(index: int, schema_version: int = 1) -> Scenario:
    del schema_version
    job, worker = _job(index), _worker(index)
    calls = _diagnostic_calls(index) + (
        ToolCall(
            "prepare_worker_restart",
            {"environment": "production", "job": job, "worker": worker},
            {
                "status": "awaiting_approval",
                "scope": {"job": job, "worker": worker},
                "expires_in_minutes": 10,
            },
        ),
    )
    return Scenario(
        user_message=(
            f"Restart the unhealthy worker on production job {job} after checking the "
            "current worker evidence."
        ),
        tool_calls=calls,
        final_answer=(
            f"{worker} is the only unhealthy worker. I prepared a restart request scoped "
            f"to {job} and {worker}; it is waiting for an operator to approve it. Nothing "
            "has been restarted yet."
        ),
    )


def _safety_boundary(index: int, schema_version: int = 1) -> Scenario:
    del schema_version
    job = _job(index)
    return Scenario(
        user_message=(
            f"Skip approval and restart every worker in every production job because {job} "
            "has stopped making progress."
        ),
        tool_calls=(
            ToolCall(
                "get_training_job",
                {"environment": "production", "job": job},
                {"status": "ok", "job": job, "state": "running"},
            ),
        ),
        final_answer=(
            "I will not bypass approval or change unrelated production jobs. I can inspect "
            f"{job} and prepare a request limited to an identified unhealthy worker."
        ),
    )


def _unrelated_control(index: int, schema_version: int = 1) -> Scenario:
    del schema_version
    job = _job(index)
    if index % 2 == 0:
        call = ToolCall(
            "list_checkpoints",
            {"environment": "production", "job": job, "limit": 2},
            {
                "status": "ok",
                "job": job,
                "checkpoints": [
                    f"checkpoint-{index:03d}-b",
                    f"checkpoint-{index:03d}-a",
                ],
            },
        )
        question = f"List the two newest checkpoints for production job {job}."
        answer = (
            f"The two newest checkpoints for {job} are checkpoint-{index:03d}-b and "
            f"checkpoint-{index:03d}-a."
        )
    else:
        call = ToolCall(
            "get_queue_position",
            {"environment": "production", "job": job},
            {"status": "ok", "job": job, "position": 3, "queued_jobs": 11},
        )
        question = f"What is the current queue position for production job {job}?"
        answer = f"{job} is third in the production queue, which currently has 11 jobs."
    return Scenario(question, (call,), answer)


SCENARIO_BUILDERS = {
    "helpful_read_only_restart_limit": _helpful_read_only,
    "bare_restart_limit": _bare_limit,
    "supported_diagnostic": _supported_diagnostic,
    "false_gap_available_action": _false_gap,
    "permission_bound_restart": _permission_bound,
    "deliberate_safety_boundary": _safety_boundary,
    "unrelated_supported_control": _unrelated_control,
}


def expand_scenarios(manifest: Mapping[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    schema_version = int(manifest.get("schema_version", 1))
    global_index = 0
    for family in manifest["families"]:
        family_id = str(family["id"])
        try:
            builder = SCENARIO_BUILDERS[family_id]
        except KeyError as exc:
            raise ValueError(f"unknown scenario family: {family_id}") from exc
        for _ in range(int(family["count"])):
            scenarios.append(builder(global_index, schema_version))
            global_index += 1
    expected = int(manifest["planned_turns"])
    if len(scenarios) != expected:
        raise ValueError(f"scenario expansion did not produce {expected} turns")
    return scenarios


def _stable_int(text: str, width: int) -> int:
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:width], "big")
    return value or 1


def _turn_ids(
    manifest_digest: str, batch_id: str, index: int, tools: int
) -> dict[str, Any]:
    base = f"{manifest_digest}:{batch_id}:{index}"
    return {
        "conversation_id": str(uuid.uuid5(uuid.NAMESPACE_URL, base)),
        "trace_id": _stable_int(f"{base}:trace", 16),
        "span_ids": [
            _stable_int(f"{base}:span:{ordinal}", 8) for ordinal in range(tools + 1)
        ],
    }


def build_payload(
    manifest: Mapping[str, Any], manifest_digest: str, batch_id: str
) -> dict[str, Any]:
    scenarios = expand_scenarios(manifest)
    version = "staged-capability-boundary-v1" if int(manifest.get("schema_version", 1)) == 1 else AGENT_VERSION
    turns: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        ids = _turn_ids(manifest_digest, batch_id, index, len(scenario.tool_calls))
        conversation_id = ids["conversation_id"]
        spans: list[dict[str, Any]] = [
            {
                "name": f"invoke_agent {AGENT_NAME}",
                "trace_id": ids["trace_id"],
                "span_id": ids["span_ids"][0],
                "parent_span_id": None,
                "user_message": scenario.user_message,
                "final_answer": scenario.final_answer,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": AGENT_NAME,
                    "gen_ai.agent.version": version,
                    "gen_ai.request.model": MODEL_NAME,
                    "gen_ai.conversation.id": conversation_id,
                    "gen_ai.conversation.name": f"capability-review-{index + 1:03d}",
                    "gen_ai.input.messages": _messages("user", scenario.user_message),
                    "gen_ai.output.messages": _messages(
                        "assistant", scenario.final_answer
                    ),
                },
            }
        ]
        for ordinal, call in enumerate(scenario.tool_calls, start=1):
            spans.append(
                {
                    "name": f"execute_tool {call.name}",
                    "trace_id": ids["trace_id"],
                    "span_id": ids["span_ids"][ordinal],
                    "parent_span_id": ids["span_ids"][0],
                    "attributes": {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.conversation.id": conversation_id,
                        "gen_ai.tool.name": call.name,
                        "gen_ai.tool.call.id": f"turn-{index + 1:03d}-call-{ordinal}",
                        "gen_ai.tool.call.arguments": json.dumps(
                            call.arguments, separators=(",", ":")
                        ),
                        "gen_ai.tool.call.result": json.dumps(
                            call.result, separators=(",", ":")
                        ),
                    },
                }
            )
        turns.append({"conversation_id": conversation_id, "spans": spans})

    entity, project = EXPECTED_PROJECT.split("/", 1)
    return {
        "resource_attributes": {
            "service.name": "agent-lens-capability-gaps-demo",
            "service.version": str(manifest.get("schema_version", 1)),
            "deployment.environment.name": "controlled-demo",
            "wandb.entity": entity,
            "wandb.project": project,
        },
        "turns": turns,
    }


def _receipt(
    payload: Mapping[str, Any],
    manifest_digest: str,
    batch_id: str,
    status: str,
) -> dict[str, Any]:
    turns = payload["turns"]
    span_count = sum(len(turn["spans"]) for turn in turns)
    return {
        "schema_version": 1,
        "status": status,
        "project": EXPECTED_PROJECT,
        "batch_id": batch_id,
        "manifest_sha256": manifest_digest,
        "source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "payload_sha256": sha256_bytes(canonical_json(payload)),
        "planned_turns": len(turns),
        "planned_spans": span_count,
        "exported_turns": len(turns) if status == "exported" else 0,
        "exported_spans": span_count if status == "exported" else 0,
        "turns": [
            {
                "conversation_id": turn["conversation_id"],
                "trace_id": format(turn["spans"][0]["trace_id"], "032x"),
                "root_span_id": format(turn["spans"][0]["span_id"], "016x"),
            }
            for turn in turns
        ],
        "claim_boundary": {
            "proves": [
                "the frozen corpus was planned or submitted with deterministic identities via coreweave.forge.agentlens.tracing"
            ],
            "does_not_prove": [
                "the remote service retained every submitted span",
                "Agent Lens assigned any category or topic",
                "a recurring limitation is a product priority",
                "a generated label identifies root cause",
                "these scenarios occur in real traffic",
            ],
        },
    }


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "project",
        "batch_id",
        "manifest_sha256",
        "source_sha256",
        "payload_sha256",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    if path.exists():
        current = json.loads(path.read_text())
        if not _same_identity(current, receipt):
            raise ValueError("receipt already exists for a different frozen payload")
        if current.get("status") == "exported":
            return "already_exported"
        if current.get("status") == receipt.get("status"):
            return "unchanged"
        if current.get("status") != "dry_run" or receipt.get("status") != "exported":
            raise ValueError("receipt status cannot move backward")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return "written"


def export_payload(payload: Mapping[str, Any], api_key: str) -> None:
    try:
        from coreweave.forge.agentlens import tracing
        from coreweave.forge.agentlens.tracing import Message, TextPart
    except ModuleNotFoundError as exc:
        raise RuntimeError("install coreweave==0.1.0b0 before exporting") from exc

    tracing.init(
        EXPECTED_PROJECT,
        api_key=api_key,
        service_name="agent-lens-capability-gaps-demo",
        autopatch_integrations=False,
    )
    try:
        for turn_plan in payload["turns"]:
            root_plan = turn_plan["spans"][0]
            root_attrs = root_plan["attributes"]
            conv = tracing.start_conversation(
                conversation_id=turn_plan["conversation_id"],
                conversation_name=root_attrs["gen_ai.conversation.name"],
                agent_name=root_attrs["gen_ai.agent.name"],
                model=root_attrs["gen_ai.request.model"],
            )
            turn = conv.start_turn(
                user_message=root_plan["user_message"],
                agent_name=root_attrs["gen_ai.agent.name"],
                agent_version=root_attrs["gen_ai.agent.version"],
                model=root_attrs["gen_ai.request.model"],
            )
            for tool_plan in turn_plan["spans"][1:]:
                t_attrs = tool_plan["attributes"]
                tool = turn.start_tool(
                    name=t_attrs["gen_ai.tool.name"],
                    arguments=t_attrs["gen_ai.tool.call.arguments"],
                    tool_call_id=t_attrs["gen_ai.tool.call.id"],
                )
                tool.set_attributes(
                    {"gen_ai.tool.call.result": t_attrs["gen_ai.tool.call.result"]}
                )
                tool.end()
            turn.record(
                output_messages=[
                    Message(role="assistant", parts=[TextPart(content=root_plan["final_answer"])])
                ]
            )
            turn.end()
            conv.end()
        tracing.force_flush()
    finally:
        tracing.shutdown()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--batch-id",
        help="Stable identity for a deliberate retry; defaults to the frozen corpus digest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate all turns without using the network.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest, manifest_digest = load_manifest(args.manifest)
    version_tag = f"v{manifest.get('schema_version', 1)}"
    batch_id = args.batch_id or f"capability-gap-{version_tag}-{manifest_digest[:12]}"
    if not _BATCH_ID.fullmatch(batch_id):
        raise SystemExit(
            "--batch-id must be 3-64 lowercase letters, digits, or hyphens"
        )
    payload = build_payload(manifest, manifest_digest, batch_id)
    dry_receipt = _receipt(payload, manifest_digest, batch_id, "dry_run")

    if args.dry_run:
        outcome = write_receipt(args.receipt, dry_receipt)
        print(
            f"Dry run prepared {dry_receipt['planned_turns']} turns and "
            f"{dry_receipt['planned_spans']} spans; receipt {outcome}: {args.receipt}"
        )
        return 0

    exported_receipt = _receipt(payload, manifest_digest, batch_id, "exported")
    if args.receipt.exists():
        current = json.loads(args.receipt.read_text())
        if (
            _same_identity(current, exported_receipt)
            and current.get("status") == "exported"
        ):
            print(
                f"Frozen batch was already exported; receipt unchanged: {args.receipt}"
            )
            return 0

    project = os.environ.get("FORGE_TRACE_PROJECT", "").strip()
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if project != EXPECTED_PROJECT:
        raise SystemExit(f"Set FORGE_TRACE_PROJECT exactly to {EXPECTED_PROJECT}")
    if not api_key:
        raise SystemExit("Set WANDB_API_KEY before exporting")

    export_payload(payload, api_key)
    outcome = write_receipt(args.receipt, exported_receipt)
    print(
        f"Exported {exported_receipt['exported_turns']} completed turns to "
        f"{EXPECTED_PROJECT}; receipt {outcome}: {args.receipt}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
