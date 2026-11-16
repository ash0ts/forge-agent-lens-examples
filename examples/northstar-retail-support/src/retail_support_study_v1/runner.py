from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import digest_value, load_json, write_json_atomic
from .confirmation import MutationConfirmationController
from .constants import (
    LIVE_AGENT_LENS_PROJECTS,
    MAX_AGENT_TOOL_CALLS,
    MAX_CONVERSATION_STEPS,
    RUNS_DIR,
    SCHEMA_VERSION,
    STUDY_ID,
    SUPPORT_MAX_OUTPUT_TOKENS,
    SUPPORT_MODEL,
    SUPPORT_TEMPERATURE,
    USER_CONFORMANCE_RETRIES,
    USER_ENABLE_THINKING,
    USER_MAX_OUTPUT_TOKENS,
    USER_MODEL,
    USER_REPAIR_SEED_OFFSET,
    USER_TEMPERATURE,
)
from .cost_policy import (
    bundle_has_disallowed_model_evidence,
    continues_frozen_attempt_failures,
    is_allowed_retry_cost_ambiguity,
)
from .inference import (
    AnthropicInferenceClient,
    InferenceCallError,
    SupportInferenceClient,
    WandbInferenceClient,
)
from .live_telemetry import LiveAgentTurn, LiveTelemetrySession
from .mcp_protocol import MCPProcessClient, MCPTimeoutError, MCPToolError
from .pricing import CostBudget
from .runtime_snapshot import RuntimeSnapshot, verify_runtime_snapshot
from .schemas import (
    AgentProfile,
    BehaviorProfile,
    ConversationBundle,
    Coordinate,
    ModelCall,
    ModelRoute,
    RunLedger,
    SimulatorConformanceReceipt,
    ToolCallRecord,
    Usage,
)
from .simulator_contracts import (
    CONFIRMATION_CONDITION_INSTRUCTION,
    SimulatorChoreography,
    confirmation_prompt_redactions,
    confirmation_source_clauses,
    simulator_repair_instruction,
)
from .simulator_output import project_simulator_output
from .traffic_config import ALL_TOOL_NAMES, route_agent_profile_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _receipt_identity(coordinate: Coordinate, preview_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "preview_sha256": preview_sha,
        "coordinate_id": coordinate.coordinate_id,
        "coordinate": coordinate.model_dump(mode="json"),
    }


def _seal_attempt(value: dict[str, Any]) -> dict[str, Any]:
    sealed = {**value, "attempt_sha256": ""}
    sealed["attempt_sha256"] = digest_value(sealed)
    return sealed


def _read_attempt_receipt(
    path: Path, *, coordinate: Coordinate, preview_sha: str
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt receipt must be a regular file")
    value = load_json(path)
    claimed_digest = str(value.get("attempt_sha256") or "")
    unsigned = {**value, "attempt_sha256": ""}
    if claimed_digest != digest_value(unsigned):
        raise ValueError(f"attempt receipt content drift: {coordinate.coordinate_id}")
    identity = _receipt_identity(coordinate, preview_sha)
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise ValueError(f"attempt receipt identity drift: {coordinate.coordinate_id}")
    if value.get("status") not in {"started", "complete", "unavailable", "uncertain"}:
        raise ValueError(
            f"attempt receipt has invalid status: {coordinate.coordinate_id}"
        )
    return value


def _task_instructions(
    task: dict[str, Any],
    *,
    confirmation_source_clause: str | None = None,
) -> str:
    scenario = task["user_scenario"]
    instructions = scenario.get("instructions") if isinstance(scenario, dict) else None
    if not isinstance(instructions, (str, dict)):
        raise ValueError(f"task {task['id']} has malformed user instructions")
    prompt_redactions = confirmation_prompt_redactions(task)
    deferred_clauses = tuple(source for source, _ in prompt_redactions)
    if confirmation_source_clause is not None and (
        confirmation_source_clause not in deferred_clauses
    ):
        raise ValueError("requested simulator clause is not registered for task")
    if isinstance(instructions, str):
        visible = instructions
        for clause, replacement in prompt_redactions:
            if clause != confirmation_source_clause:
                visible = visible.replace(clause, replacement)
        return visible.strip()
    blocks = []
    for key in ("task_instructions", "reason_for_call", "known_info", "unknown_info"):
        value = instructions.get(key)
        if value:
            visible_value = str(value)
            for clause, replacement in prompt_redactions:
                if clause != confirmation_source_clause:
                    visible_value = visible_value.replace(clause, replacement)
            visible_value = visible_value.strip()
            if visible_value:
                blocks.append(
                    f"{key.replace('_', ' ').title()}: {visible_value}"
                )
    return "\n".join(blocks)


def _user_system_prompt(
    task: dict[str, Any],
    profile: str,
    runtime: RuntimeSnapshot,
    *,
    confirmation_source_clause: str | None = None,
    repair_instruction: str | None = None,
) -> str:
    profile_file = {
        "ordinary": "ordinary-user.md",
        "concise": "concise-user.md",
        "skeptical": "skeptical-user.md",
    }[profile]
    prompt = (
        f"{runtime.user_guidelines}\n\n{runtime.prompts[profile_file]}\n\n"
        f"{CONFIRMATION_CONDITION_INSTRUCTION.strip()}\n\n"
        "<scenario>\n"
        f"{_task_instructions(task, confirmation_source_clause=confirmation_source_clause)}"
        "\n</scenario>"
    )
    if repair_instruction is not None:
        prompt = f"{prompt}\n\n{repair_instruction}"
    return prompt


def _simulator_seed(*, coordinate: Coordinate, step: int, repair: int) -> int:
    return coordinate.seed + step + repair * USER_REPAIR_SEED_OFFSET


def _agent_system_prompt(
    runtime: RuntimeSnapshot,
    *,
    agent_profile: AgentProfile | None = None,
) -> str:
    role = ""
    if agent_profile is not None:
        role = (
            "\n\n<assigned-support-role>\n"
            f"You are the {agent_profile.agent_name} agent. "
            f"{agent_profile.purpose}\n"
            "Use only the tools made available for this role.\n"
            "</assigned-support-role>"
        )
    return (
        f"{runtime.prompts['support-agent.md']}\n\n"
        f"<retail-policy>\n{runtime.policy}\n</retail-policy>{role}"
    )


def _user_view(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            result.append({"role": "assistant", "content": message.get("content", "")})
        elif role == "assistant" and message.get("content"):
            result.append({"role": "user", "content": message["content"]})
        # Tool messages are evidence for the support agent, not the simulated customer.
    return result


def _usage_total(records: list[ModelCall]) -> Usage:
    observed = [
        record.usage.observed_cost_usd
        for record in records
        if record.usage.observed_cost_usd is not None
    ]
    provider_reported = [
        record.usage.provider_reported_cost_usd
        for record in records
        if record.usage.provider_reported_cost_usd is not None
    ]
    return Usage(
        input_tokens=sum(record.usage.input_tokens for record in records),
        cached_input_tokens=sum(record.usage.cached_input_tokens for record in records),
        output_tokens=sum(record.usage.output_tokens for record in records),
        total_tokens=sum(record.usage.total_tokens for record in records),
        provider_reported_cost_usd=(
            round(sum(provider_reported), 12) if provider_reported else None
        ),
        observed_cost_usd=round(sum(observed), 12) if observed else None,
    )


def _cost_evidence(records: list[ModelCall]) -> tuple[float, float, list[str]]:
    observed = round(
        sum(
            float(record.usage.observed_cost_usd)
            for record in records
            if record.usage.observed_cost_usd is not None
        ),
        12,
    )
    ambiguous = round(
        sum(record.ambiguous_cost_reserve_usd for record in records),
        12,
    )
    reasons = [
        reason for record in records for reason in record.cost_uncertainty_reasons
    ]
    return observed, ambiguous, reasons


def _record_bundle_cost(budget: CostBudget, bundle: ConversationBundle) -> None:
    observed, ambiguous, reasons = _cost_evidence(bundle.model_calls)
    budget.record(observed)
    if ambiguous:
        for index, reason in enumerate(reasons):
            budget.record_ambiguous(
                ambiguous if index == 0 else 0.0,
                reason=reason,
            )


@dataclass(frozen=True)
class _CoordinateInferenceClients:
    """One coordinate's provider clients sharing one exact cost boundary."""

    user: WandbInferenceClient
    support: SupportInferenceClient
    budget: CostBudget


def _traffic_rows(runtime: RuntimeSnapshot) -> dict[str, Any] | None:
    traffic = runtime.preview.get("traffic")
    if traffic is None:
        return None
    if not isinstance(traffic, dict):
        raise ValueError("preview traffic contract must be an object")
    return traffic


def _model_route(
    runtime: RuntimeSnapshot,
    coordinate: Coordinate,
) -> ModelRoute | None:
    traffic = _traffic_rows(runtime)
    if traffic is None:
        if coordinate.model_route_id is not None:
            raise ValueError("coordinate model route has no preview traffic contract")
        return None
    route_id = coordinate.model_route_id
    if not route_id:
        raise ValueError("v10 coordinate has no preview-sealed model route")
    raw_routes = traffic.get("model_routes")
    if not isinstance(raw_routes, list):
        raise ValueError("preview traffic model routes must be a list")
    routes = {
        route.route_id: route
        for route in (ModelRoute.model_validate(raw) for raw in raw_routes)
    }
    try:
        return routes[route_id]
    except KeyError as exc:
        raise ValueError(f"unknown coordinate model route: {route_id}") from exc


def _agent_profile(
    runtime: RuntimeSnapshot,
    *,
    first_customer_text: str,
) -> AgentProfile | None:
    traffic = _traffic_rows(runtime)
    if traffic is None:
        return None
    raw_profiles = traffic.get("agent_profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("preview traffic agent profiles must be a list")
    profiles = {
        profile.profile_id: profile
        for profile in (AgentProfile.model_validate(raw) for raw in raw_profiles)
    }
    selected_id = route_agent_profile_id(first_customer_text)
    try:
        return profiles[selected_id]
    except KeyError as exc:
        raise ValueError(f"routed agent profile is not preview-sealed: {selected_id}") from exc


def _behavior_profile(runtime: RuntimeSnapshot) -> dict[str, Any] | None:
    traffic = _traffic_rows(runtime)
    if traffic is None:
        return None
    raw = traffic.get("behavior_profile")
    if not isinstance(raw, dict):
        raise ValueError("preview traffic behavior profile must be an object")
    return BehaviorProfile.model_validate(raw).model_dump(mode="json")


def _coordinate_inference_clients(
    coordinate: Coordinate,
    *,
    runtime: RuntimeSnapshot,
    cost_cap_usd: float,
) -> _CoordinateInferenceClients:
    """Bind simulator and support clients to one coordinate-level budget.

    The simulator always uses W&B Inference. The support provider is selected
    only from the coordinate's preview-sealed route; there is no runtime model
    substitution or provider fallback.
    """

    budget = CostBudget(
        cost_cap_usd,
        price_lock=dict(runtime.pricing_lock),
    )
    user = WandbInferenceClient(budget=budget)
    route = _model_route(runtime, coordinate)
    if route is None or route.provider_name == "wandb":
        support: SupportInferenceClient = user
    elif route.provider_name == "anthropic":
        support = AnthropicInferenceClient(budget=budget)
    else:  # pragma: no cover - ModelRoute validation already narrows providers
        raise ValueError(f"unsupported support provider: {route.provider_name}")
    return _CoordinateInferenceClients(user=user, support=support, budget=budget)


def _run_coordinate_with_client(
    coordinate: Coordinate,
    *,
    client: WandbInferenceClient | None = None,
    user_client: WandbInferenceClient | None = None,
    support_client: SupportInferenceClient | None = None,
    budget: CostBudget | None = None,
    tasks_by_id: dict[str, dict[str, Any]],
    runtime: RuntimeSnapshot,
    live_telemetry: LiveTelemetrySession | None = None,
) -> ConversationBundle:
    """Run one isolated coordinate and reconcile its private client budget."""

    legacy_client = client
    if client is not None:
        if user_client is not None or support_client is not None or budget is not None:
            raise ValueError("legacy client cannot be combined with split clients")
        user_client = client
        support_client = client
        budget = client.budget
    if user_client is None or support_client is None or budget is None:
        raise ValueError("coordinate requires user, support, and budget bindings")
    run_kwargs: dict[str, Any] = {
        "tasks_by_id": tasks_by_id,
        "runtime": runtime,
    }
    if legacy_client is not None:
        run_kwargs["client"] = legacy_client
    else:
        run_kwargs["user_client"] = user_client
        run_kwargs["support_client"] = support_client
    if live_telemetry is not None:
        run_kwargs["live_telemetry"] = live_telemetry
    bundle = run_coordinate(coordinate, **run_kwargs)
    observed, ambiguous, reasons = _cost_evidence(bundle.model_calls)
    if abs(budget.observed_usd - observed) > 1e-12:
        raise ValueError(
            f"coordinate client observed cost drift: {coordinate.coordinate_id}"
        )
    if abs(budget.ambiguous_cost_reserve_usd - ambiguous) > 1e-12:
        raise ValueError(
            f"coordinate client ambiguous cost drift: {coordinate.coordinate_id}"
        )
    if list(budget.uncertainty_reasons) != reasons:
        raise ValueError(
            f"coordinate client uncertainty reason drift: {coordinate.coordinate_id}"
        )
    return bundle


def _complete_and_record(
    client: SupportInferenceClient,
    records: list[ModelCall],
    *,
    allow_bounded_cost_ambiguity: bool = False,
    **request: Any,
) -> Any:
    try:
        completion = client.complete(**request)
    except InferenceCallError as exc:
        records.append(exc.record)
        raise
    records.append(completion.record)
    if (
        completion.record.ambiguous_cost_reserve_usd
        and not allow_bounded_cost_ambiguity
    ):
        raise RuntimeError(
            "model-call cost became uncertain; the frozen response cannot be "
            "used for further paid work"
        )
    if (
        completion.record.ambiguous_cost_reserve_usd
        and not is_allowed_retry_cost_ambiguity(completion.record)
    ):
        raise RuntimeError(
            "model-call cost ambiguity is outside the approved retry policy"
        )
    return completion


def _has_terminal_agent_response(messages: list[dict[str, Any]]) -> bool:
    """Exclude the fixed greeting when deciding whether the agent answered."""

    return any(
        message.get("role") == "assistant"
        and message.get("content")
        and not message.get("tool_calls")
        for message in messages[1:]
    )


def _parse_tool_arguments(raw: dict[str, Any]) -> dict[str, Any]:
    arguments = json.loads(raw["function"].get("arguments") or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must decode to an object")
    return arguments


def _record_simulator_conformance(
    receipts: list[SimulatorConformanceReceipt],
    *,
    violation: Any,
    rejected_call: ModelCall,
    rejected_text: str,
    message_index: int,
    repair_call: ModelCall | None,
    accepted_text: str | None,
    outcome: str,
) -> None:
    receipts.append(
        SimulatorConformanceReceipt(
            schema_version=3,
            sequence=len(receipts) + 1,
            event_id=violation.event.event_id,
            boundary=violation.boundary,
            staged_tool_call_id=violation.staged_tool_call_id,
            staged_action_sha256=violation.staged_action_sha256,
            trigger_message_index=violation.trigger_message_index,
            trigger_message_sha256=violation.trigger_message_sha256,
            rejected_call_id=rejected_call.call_id,
            rejected_output_sha256=digest_value(rejected_call.output_message),
            rejected_message_index=message_index,
            rejected_message_sha256=digest_value(
                {"role": "user", "content": rejected_text}
            ),
            repair_call_id=repair_call.call_id if repair_call is not None else None,
            repair_output_sha256=(
                digest_value(repair_call.output_message)
                if repair_call is not None
                else None
            ),
            accepted_message_index=(
                message_index if accepted_text is not None else None
            ),
            accepted_message_sha256=(
                digest_value({"role": "user", "content": accepted_text})
                if accepted_text is not None
                else None
            ),
            outcome=outcome,
        )
    )


def _next_user_reply(
    *,
    client: WandbInferenceClient,
    coordinate: Coordinate,
    step: int,
    task: dict[str, Any],
    runtime: RuntimeSnapshot,
    messages: list[dict[str, Any]],
    confirmation: MutationConfirmationController,
    choreography: SimulatorChoreography,
    model_calls: list[ModelCall],
    conformance_receipts: list[SimulatorConformanceReceipt],
    allow_bounded_cost_ambiguity: bool,
) -> tuple[str, bool]:
    violation: Any | None = None
    rejected_call: ModelCall | None = None
    rejected_text: str | None = None
    for repair in range(USER_CONFORMANCE_RETRIES + 1):
        repair_instruction = (
            simulator_repair_instruction(violation)
            if violation is not None
            else None
        )
        if violation is None:
            confirmation_source_clause = choreography.source_clause_for_prompt(
                receipts=confirmation.private_receipts(),
                messages=messages,
            )
        elif violation.boundary == "prepared_action":
            registered_clauses = confirmation_source_clauses(task)
            confirmation_source_clause = (
                violation.event.source_clause
                if violation.event.source_clause in registered_clauses
                else None
            )
        else:
            confirmation_source_clause = None
        try:
            user_messages = [
                {
                    "role": "system",
                    "content": _user_system_prompt(
                        task,
                        coordinate.profile,
                        runtime,
                        confirmation_source_clause=confirmation_source_clause,
                        repair_instruction=repair_instruction,
                    ),
                },
                *_user_view(messages),
            ]
            completion = _complete_and_record(
                client,
                model_calls,
                allow_bounded_cost_ambiguity=allow_bounded_cost_ambiguity,
                participant="user_simulator",
                model=USER_MODEL,
                temperature=USER_TEMPERATURE,
                seed=_simulator_seed(
                    coordinate=coordinate,
                    step=step,
                    repair=repair,
                ),
                messages=user_messages,
                max_tokens=USER_MAX_OUTPUT_TOKENS,
                enable_thinking=USER_ENABLE_THINKING,
            )
        except Exception:
            if violation is not None and rejected_call is not None:
                repair_call = (
                    model_calls[-1]
                    if model_calls
                    and model_calls[-1].participant == "user_simulator"
                    and model_calls[-1].call_id != rejected_call.call_id
                    else None
                )
                if repair_call is not None:
                    repair_call.transcript_disposition = (
                        "rejected_simulator_conformance"
                    )
                _record_simulator_conformance(
                    conformance_receipts,
                    violation=violation,
                    rejected_call=rejected_call,
                    rejected_text=str(rejected_text or ""),
                    message_index=len(messages),
                    repair_call=repair_call,
                    accepted_text=None,
                    outcome="unavailable",
                )
            raise

        text, stopped = project_simulator_output(completion.message)
        current_violation = choreography.evaluate_confirmation_reply(
            text=text,
            receipts=confirmation.private_receipts(),
            messages=messages,
            stopped=stopped,
        )
        if current_violation is None:
            if violation is not None and rejected_call is not None:
                if stopped or not text:
                    completion.record.transcript_disposition = (
                        "rejected_simulator_conformance"
                    )
                    _record_simulator_conformance(
                        conformance_receipts,
                        violation=violation,
                        rejected_call=rejected_call,
                        rejected_text=str(rejected_text or ""),
                        message_index=len(messages),
                        repair_call=completion.record,
                        accepted_text=None,
                        outcome="unavailable",
                    )
                    raise RuntimeError(
                        "simulator conformance repair returned no customer correction"
                    )
                _record_simulator_conformance(
                    conformance_receipts,
                    violation=violation,
                    rejected_call=rejected_call,
                    rejected_text=str(rejected_text or ""),
                    message_index=len(messages),
                    repair_call=completion.record,
                    accepted_text=text,
                    outcome="repaired",
                )
            return text, stopped

        completion.record.transcript_disposition = (
            "rejected_simulator_conformance"
        )
        if violation is None:
            violation = current_violation
            rejected_call = completion.record
            rejected_text = text
            continue
        _record_simulator_conformance(
            conformance_receipts,
            violation=violation,
            rejected_call=rejected_call,
            rejected_text=str(rejected_text or ""),
            message_index=len(messages),
            repair_call=completion.record,
            accepted_text=None,
            outcome="unavailable",
        )
        raise RuntimeError(
            "simulator failed the bounded scenario-conformance repair"
        )
    raise RuntimeError("simulator conformance retry policy exhausted")


def run_coordinate(
    coordinate: Coordinate,
    *,
    client: WandbInferenceClient | None = None,
    user_client: WandbInferenceClient | None = None,
    support_client: SupportInferenceClient | None = None,
    tasks_by_id: dict[str, dict[str, Any]],
    runtime: RuntimeSnapshot,
    live_telemetry: LiveTelemetrySession | None = None,
) -> ConversationBundle:
    if client is not None:
        if user_client is not None or support_client is not None:
            raise ValueError("legacy client cannot be combined with split clients")
        user_client = client
        support_client = client
    if user_client is None or support_client is None:
        raise ValueError("run_coordinate requires user and support inference clients")

    conversation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, coordinate.coordinate_id))
    model_calls: list[ModelCall] = []
    tool_calls: list[ToolCallRecord] = []
    simulator_conformance_receipts: list[SimulatorConformanceReceipt] = []
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    confirmation = MutationConfirmationController(conversation_id=conversation_id)
    task = tasks_by_id[coordinate.logical_task_id]
    choreography = SimulatorChoreography(task)
    allow_bounded_cost_ambiguity = continues_frozen_attempt_failures(
        dict(runtime.preview)
    )
    active_turn: LiveAgentTurn | None = None
    route = _model_route(runtime, coordinate)
    if route is not None:
        if getattr(support_client, "provider_name", None) != route.provider_name:
            raise ValueError("support client provider differs from coordinate route")
        if getattr(support_client, "server_address", None) != route.server_address:
            raise ValueError("support client server differs from coordinate route")
    support_model = route.model if route is not None else SUPPORT_MODEL
    provider_name = (
        route.provider_name
        if route is not None
        else str(getattr(support_client, "provider_name", "wandb"))
    )
    server_address = (
        route.server_address
        if route is not None
        else str(
            getattr(
                support_client,
                "server_address",
                "api.inference.wandb.ai",
            )
        )
    )
    selected_agent: AgentProfile | None = None
    support_temperature = (
        route.temperature if route is not None else SUPPORT_TEMPERATURE
    )

    def start_live_turn() -> LiveAgentTurn | None:
        if live_telemetry is None:
            return None
        args = (
            conversation_id,
            [messages[-1]],
            {record.call_id: record for record in tool_calls},
        )
        return live_telemetry.start_turn(
            *args,
            agent_name=(
                selected_agent.agent_name
                if selected_agent is not None
                else "retail-support-agent"
            ),
            requested_model=support_model,
            requested_temperature=support_temperature,
            provider_name=provider_name,
            server_address=server_address,
            server_port=443,
        )

    try:
        # The first customer turn is generated before selecting an agent or
        # starting its MCP server. Routing therefore sees only the visible
        # customer request, never the benchmark task ID or expected action.
        first_text, first_stopped = _next_user_reply(
            client=user_client,
            coordinate=coordinate,
            step=0,
            task=task,
            runtime=runtime,
            messages=messages,
            confirmation=confirmation,
            choreography=choreography,
            model_calls=model_calls,
            conformance_receipts=simulator_conformance_receipts,
            allow_bounded_cost_ambiguity=allow_bounded_cost_ambiguity,
        )
        if first_stopped or not first_text:
            raise RuntimeError(
                "user simulator stopped before the support agent received a request"
            )
        messages.append({"role": "user", "content": first_text})
        confirmation.observe_user_message(
            text=first_text,
            message_index=len(messages) - 1,
        )
        selected_agent = _agent_profile(
            runtime,
            first_customer_text=first_text,
        )
        allowed_tool_names = (
            selected_agent.tool_names if selected_agent is not None else None
        )
        with MCPProcessClient(
            db_path=runtime.db_path,
            policy_path=runtime.policy_path,
            authenticated_user_id=str(task["authenticated_user_id"]),
            package_source_dir=runtime.package_source_dir,
            allowed_tool_names=allowed_tool_names,
            behavior_profile=_behavior_profile(runtime),
            conversation_id=(
                conversation_id if _traffic_rows(runtime) is not None else None
            ),
        ) as mcp:
            tool_defs = mcp.list_openai_tools()
            expected_tool_count = (
                len(selected_agent.tool_names)
                if selected_agent is not None
                else len(ALL_TOOL_NAMES)
            )
            if len(tool_defs) != expected_tool_count:
                raise RuntimeError(
                    f"expected {expected_tool_count} MCP tools, "
                    f"discovered {len(tool_defs)}"
                )

            active_turn = start_live_turn()
            for step in range(MAX_CONVERSATION_STEPS):
                if not messages or messages[-1]["role"] == "assistant":
                    text, stopped = _next_user_reply(
                        client=user_client,
                        coordinate=coordinate,
                        step=step,
                        task=task,
                        runtime=runtime,
                        messages=messages,
                        confirmation=confirmation,
                        choreography=choreography,
                        model_calls=model_calls,
                        conformance_receipts=simulator_conformance_receipts,
                        allow_bounded_cost_ambiguity=(
                            allow_bounded_cost_ambiguity
                        ),
                    )
                    if stopped:
                        terminal = _has_terminal_agent_response(messages)
                        if not terminal:
                            raise RuntimeError(
                                "user simulator stopped before the support agent "
                                "returned a terminal response"
                            )
                        bundle = ConversationBundle(
                            schema_version=SCHEMA_VERSION,
                            study_id=STUDY_ID,
                            conversation_id=conversation_id,
                            coordinate=coordinate,
                            requested_models={
                                "support_agent": support_model,
                                "user_simulator": USER_MODEL,
                            },
                            returned_models={
                                "support_agent": sorted(
                                    {
                                        call.returned_model
                                        for call in model_calls
                                        if call.participant == "support_agent"
                                    }
                                ),
                                "user_simulator": sorted(
                                    {
                                        call.returned_model
                                        for call in model_calls
                                        if call.participant == "user_simulator"
                                    }
                                ),
                            },
                            status="complete",
                            messages=messages,
                            model_calls=model_calls,
                            tool_calls=tool_calls,
                            usage=_usage_total(model_calls),
                            mutation_confirmation_receipts=(
                                confirmation.private_receipts()
                            ),
                            mutation_confirmation_ok=(
                                confirmation.mutation_confirmation_ok
                            ),
                            simulator_conformance_receipts=(
                                simulator_conformance_receipts
                            ),
                            terminal_response=terminal,
                        )
                        bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
                        return bundle
                    if text:
                        messages.append({"role": "user", "content": text})
                        confirmation.observe_user_message(
                            text=text,
                            message_index=len(messages) - 1,
                        )
                        if live_telemetry is not None:
                            if active_turn is not None:  # pragma: no cover - invariant
                                raise RuntimeError(
                                    "cannot start a new live turn before ending the prior turn"
                                )
                            active_turn = start_live_turn()
                    if not text:
                        raise RuntimeError(
                            "user simulator returned neither dialogue nor a stop token"
                        )

                agent_messages = [
                    {
                        "role": "system",
                        "content": _agent_system_prompt(
                            runtime,
                            agent_profile=selected_agent,
                        ),
                    },
                    *messages,
                ]
                support_request: dict[str, Any] = {
                    "participant": "support_agent",
                    "model": support_model,
                    "temperature": support_temperature,
                    "enable_thinking": (
                        route.enable_thinking if route is not None else None
                    ),
                    "seed": None,
                    "messages": agent_messages,
                    "tools": tool_defs,
                    "max_tokens": SUPPORT_MAX_OUTPUT_TOKENS,
                }

                def complete_support_call() -> Any:
                    return _complete_and_record(
                        support_client,
                        model_calls,
                        allow_bounded_cost_ambiguity=(
                            allow_bounded_cost_ambiguity
                        ),
                        **support_request,
                    )

                completion = (
                    active_turn.complete_support_call(
                        complete_support_call,
                        model_calls,
                        support_request,
                    )
                    if active_turn is not None
                    else complete_support_call()
                )
                output = completion.message
                raw_calls = output.get("tool_calls") or []
                if raw_calls:
                    messages.append(output)
                    if len(raw_calls) > 1:
                        for raw in raw_calls:
                            try:
                                arguments = _parse_tool_arguments(raw)
                            except (
                                KeyError,
                                TypeError,
                                ValueError,
                                json.JSONDecodeError,
                            ):
                                arguments = {}
                            record = ToolCallRecord(
                                call_id=str(raw.get("id") or ""),
                                name=str(raw.get("function", {}).get("name") or ""),
                                arguments=arguments,
                                error_type="ParallelToolCallsRejected",
                                error_message=(
                                    "the retail policy permits one tool call at a time"
                                ),
                                execution_disposition="rejected",
                                duration_ms=0.0,
                            )
                            tool_calls.append(record)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": raw["id"],
                                    "content": json.dumps(
                                        {
                                            "error": (
                                                "The retail policy allows one tool call "
                                                "at a time. Retry with one call."
                                            )
                                        }
                                    ),
                                }
                            )
                        if len(tool_calls) >= MAX_AGENT_TOOL_CALLS:
                            raise RuntimeError("agent tool-call cap reached")
                        continue
                    if len(tool_calls) >= MAX_AGENT_TOOL_CALLS:
                        raise RuntimeError("agent tool-call cap reached")
                    raw = raw_calls[0]
                    name = raw["function"]["name"]
                    try:
                        arguments = _parse_tool_arguments(raw)
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        record = ToolCallRecord(
                            call_id=str(raw.get("id") or ""),
                            name=str(raw.get("function", {}).get("name") or ""),
                            arguments={},
                            error_type="InvalidToolArguments",
                            error_message=str(exc),
                            execution_disposition="rejected",
                            duration_ms=0.0,
                        )
                        tool_calls.append(record)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": raw["id"],
                                "content": json.dumps(
                                    {
                                        "error": "invalid_tool_arguments",
                                        "message": "tool arguments must be a JSON object",
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                        continue
                    try:
                        arguments = mcp.validate_tool_arguments(name, arguments)
                    except Exception as exc:
                        record = ToolCallRecord(
                            call_id=str(raw.get("id") or ""),
                            name=name,
                            arguments=arguments,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            execution_disposition="rejected",
                            duration_ms=0.0,
                        )
                        tool_calls.append(record)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": raw["id"],
                                "content": json.dumps(
                                    {
                                        "error": "invalid_tool_arguments",
                                        "message": str(exc),
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                        continue
                    started = time.perf_counter()
                    gate = confirmation.guard_write(
                        tool_name=name,
                        arguments=arguments,
                        tool_call_id=raw["id"],
                        message_index=len(messages) - 1,
                    )
                    if not gate.authorized and gate.decision == (
                        "awaiting_confirmation"
                    ):
                        record = ToolCallRecord(
                            call_id=raw["id"],
                            name=name,
                            arguments=arguments,
                            error_type="ConfirmationStillPending",
                            error_message=(
                                "explain the prepared action and obtain a new "
                                "customer confirmation before retrying"
                            ),
                            execution_disposition="rejected",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                        content = json.dumps(
                            {
                                "error": "confirmation_still_pending",
                                "message": record.error_message,
                            },
                            sort_keys=True,
                        )
                    elif not gate.authorized:
                        if gate.public_result is None:  # pragma: no cover - invariant
                            raise RuntimeError("prepared write has no public result")
                        record = ToolCallRecord(
                            call_id=raw["id"],
                            name=name,
                            arguments=arguments,
                            result=gate.public_result,
                            execution_disposition="prepared",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                        content = json.dumps(
                            gate.public_result,
                            sort_keys=True,
                        )
                    else:
                        try:
                            result = (
                                active_turn.execute_mcp(
                                    lambda record_request_id: mcp.call_tool(
                                        name,
                                        arguments,
                                        on_request_id=record_request_id,
                                    ),
                                    name=name,
                                    arguments=arguments,
                                    call_id=raw["id"],
                                    session_id=mcp.session_id,
                                )
                                if active_turn is not None
                                else mcp.call_tool(name, arguments)
                            )
                            record = ToolCallRecord(
                                call_id=raw["id"],
                                name=name,
                                arguments=arguments,
                                result=result,
                                execution_disposition="executed",
                                duration_ms=(time.perf_counter() - started) * 1000,
                            )
                            content = json.dumps(
                                result, ensure_ascii=False, sort_keys=True
                            )
                        except Exception as exc:
                            if isinstance(exc, MCPToolError):
                                public_error_type = exc.code
                            elif isinstance(exc, MCPTimeoutError):
                                public_error_type = "timeout"
                            else:
                                public_error_type = type(exc).__name__
                            record = ToolCallRecord(
                                call_id=raw["id"],
                                name=name,
                                arguments=arguments,
                                error_type=public_error_type,
                                error_message=str(exc),
                                execution_disposition="executed",
                                duration_ms=(time.perf_counter() - started) * 1000,
                            )
                            content = json.dumps(
                                {"error": public_error_type, "message": str(exc)},
                                sort_keys=True,
                            )
                    tool_calls.append(record)
                    messages.append(
                        {"role": "tool", "tool_call_id": raw["id"], "content": content}
                    )
                    continue

                text = str(output.get("content") or "").strip()
                if not text:
                    raise RuntimeError(
                        "support agent returned neither content nor a tool call"
                    )
                messages.append({"role": "assistant", "content": text})
                confirmation.observe_assistant_message(
                    text=text,
                    message_index=len(messages) - 1,
                )
                if active_turn is not None:
                    active_turn.finish(
                        messages[-1],
                        {record.call_id: record for record in tool_calls},
                    )
                    active_turn = None

            raise RuntimeError("conversation step cap reached")
    except Exception as exc:
        if active_turn is not None:
            active_turn.fail(exc)
            active_turn = None
        bundle = ConversationBundle(
            schema_version=SCHEMA_VERSION,
            study_id=STUDY_ID,
            conversation_id=conversation_id,
            coordinate=coordinate,
            requested_models={
                "support_agent": support_model,
                "user_simulator": USER_MODEL,
            },
            returned_models={
                participant: sorted(
                    {
                        call.returned_model
                        for call in model_calls
                        if call.participant == participant
                    }
                )
                for participant in ("support_agent", "user_simulator")
            },
            status="unavailable",
            unavailable_reason=f"{type(exc).__name__}: {exc}",
            messages=messages,
            model_calls=model_calls,
            tool_calls=tool_calls,
            usage=_usage_total(model_calls),
            mutation_confirmation_receipts=confirmation.private_receipts(),
            mutation_confirmation_ok=confirmation.mutation_confirmation_ok,
            simulator_conformance_receipts=simulator_conformance_receipts,
            terminal_response=False,
        )
        bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
        return bundle


def _attempt_from_bundle(
    *,
    coordinate: Coordinate,
    preview_sha: str,
    bundle: ConversationBundle,
    started_at: str | None,
    completed_at: str | None = None,
    reconciled_at: str | None = None,
    continue_frozen_attempt_failures: bool = False,
) -> dict[str, Any]:
    observed, ambiguous, reasons = _cost_evidence(bundle.model_calls)
    has_disallowed_model_evidence = bundle_has_disallowed_model_evidence(bundle)
    status = (
        bundle.status
        if (
            continue_frozen_attempt_failures
            and not has_disallowed_model_evidence
        )
        else (
            "uncertain"
            if ambiguous or has_disallowed_model_evidence
            else bundle.status
        )
    )
    value: dict[str, Any] = {
        **_receipt_identity(coordinate, preview_sha),
        "status": status,
        "started_at": started_at,
        "bundle_sha256": bundle.bundle_sha256,
        "observed_cost_usd": observed,
        "ambiguous_cost_reserve_usd": ambiguous,
        "cost_exposure_upper_bound_usd": round(observed + ambiguous, 12),
        "cost_uncertainty_reasons": reasons,
    }
    if completed_at is not None:
        value["completed_at"] = completed_at
    if reconciled_at is not None:
        value["reconciled_at"] = reconciled_at
    return _seal_attempt(value)


def _write_ledger(
    *,
    path: Path,
    cohort: str,
    preview_sha: str,
    coordinates: list[Coordinate],
    completed: list[str],
    unavailable: dict[str, str],
    bundle_digests: dict[str, str],
    budget: CostBudget,
    status: str,
) -> RunLedger:
    ledger = RunLedger(
        schema_version=SCHEMA_VERSION,
        study_id=STUDY_ID,
        cohort=cohort,
        preview_sha256=preview_sha,
        planned_coordinate_ids=[item.coordinate_id for item in coordinates],
        completed_coordinate_ids=completed,
        unavailable=unavailable,
        observed_cost_usd=budget.observed_usd,
        ambiguous_cost_reserve_usd=budget.ambiguous_cost_reserve_usd,
        cost_exposure_upper_bound_usd=budget.cost_exposure_upper_bound_usd,
        cost_uncertainty_reasons=(
            list(budget.uncertainty_reasons)
            if budget.ambiguous_cost_reserve_usd
            else []
        ),
        bundle_digests=bundle_digests,
        status=status,
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    write_json_atomic(path, ledger.model_dump(mode="json"))
    return ledger


def _load_resume_ledger(
    path: Path,
    *,
    cohort: str,
    preview_sha: str,
    planned_coordinate_ids: list[str],
) -> RunLedger | None:
    """Validate the identity of an existing run before any paid call resumes."""

    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("run ledger must be a regular file")
    ledger = RunLedger.model_validate(load_json(path))
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("run ledger content digest mismatch")
    if ledger.study_id != STUDY_ID or ledger.cohort != cohort:
        raise ValueError("run ledger identity mismatch")
    if ledger.preview_sha256 != preview_sha:
        raise ValueError("run ledger preview mismatch")
    if ledger.planned_coordinate_ids != planned_coordinate_ids:
        raise ValueError("run ledger planned coordinates mismatch")
    return ledger


def _require_fresh_replacement_directory(path: Path) -> None:
    """Forbid imported artifacts in a replacement canary's new run directory."""

    if path.is_symlink():
        raise ValueError("replacement canary run directory cannot be a symlink")
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("replacement canary run path must be a directory")
    try:
        first_entry = next(path.iterdir())
    except StopIteration:
        return
    raise ValueError(
        "replacement canary requires a fresh run directory; found pre-existing "
        f"entry: {first_entry.name}"
    )


def _resume_json_files(path: Path, *, label: str) -> dict[str, Path]:
    """Index a retained artifact directory without following imported paths."""

    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"resume {label} directory must be a regular directory")
    files: dict[str, Path] = {}
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise ValueError(f"resume {label} contains an invalid entry: {entry.name}")
        if entry.stem in files:
            raise ValueError(f"resume {label} repeats an artifact: {entry.stem}")
        files[entry.stem] = entry
    return files


def _validate_replacement_resume_artifacts(
    *,
    cohort_dir: Path,
    ledger: RunLedger,
    coordinates: list[Coordinate],
    preview_sha: str,
    continue_frozen_attempt_failures: bool,
) -> None:
    """Fail closed on any missing or mismatched replacement-run paid artifact."""

    if cohort_dir.is_symlink() or not cohort_dir.is_dir():
        raise ValueError("replacement canary resume directory must be regular")
    coordinates_by_id = {coordinate.coordinate_id: coordinate for coordinate in coordinates}
    expected_order = [coordinate.coordinate_id for coordinate in coordinates]
    expected = set(expected_order)
    completed = ledger.completed_coordinate_ids
    unavailable = ledger.unavailable
    if len(completed) != len(set(completed)) or not set(completed) <= expected:
        raise ValueError("replacement resume ledger completed coordinates mismatch")
    if not set(unavailable) <= expected or set(completed) & set(unavailable):
        raise ValueError("replacement resume ledger unavailable coordinates mismatch")

    bundle_files = _resume_json_files(cohort_dir / "bundles", label="bundle")
    attempt_files = _resume_json_files(cohort_dir / "attempts", label="attempt")
    if set(bundle_files) != set(ledger.bundle_digests):
        raise ValueError(
            "replacement resume ledger bundle digests do not match retained files"
        )
    if set(attempt_files) != set(bundle_files):
        raise ValueError(
            "replacement resume attempt files do not match retained bundles"
        )

    completed_from_bundles: list[str] = []
    unavailable_from_bundles: dict[str, str] = {}
    observed_cost = 0.0
    ambiguous_cost = 0.0
    uncertainty_reasons: list[str] = []
    for coordinate_id in expected_order:
        bundle_path = bundle_files.get(coordinate_id)
        if bundle_path is None:
            continue
        coordinate = coordinates_by_id[coordinate_id]
        bundle = ConversationBundle.model_validate(load_json(bundle_path))
        if bundle.coordinate != coordinate:
            raise ValueError(
                f"replacement resume bundle coordinate drift: {coordinate_id}"
            )
        if bundle.bundle_sha256 != ledger.bundle_digests[coordinate_id]:
            raise ValueError(
                f"replacement resume bundle ledger digest drift: {coordinate_id}"
            )
        if bundle.bundle_sha256 != digest_value(bundle.unsigned_payload()):
            raise ValueError(
                f"replacement resume bundle content drift: {coordinate_id}"
            )
        if bundle.status == "complete":
            completed_from_bundles.append(coordinate_id)
        else:
            unavailable_from_bundles[coordinate_id] = str(bundle.unavailable_reason)

        attempt = _read_attempt_receipt(
            attempt_files[coordinate_id],
            coordinate=coordinate,
            preview_sha=preview_sha,
        )
        if attempt is None:  # pragma: no cover - the indexed file is already required
            raise ValueError(
                f"replacement resume bundle has no attempt receipt: {coordinate_id}"
            )
        expected_attempt = _attempt_from_bundle(
            coordinate=coordinate,
            preview_sha=preview_sha,
            bundle=bundle,
            started_at=attempt.get("started_at"),
            continue_frozen_attempt_failures=(
                continue_frozen_attempt_failures
            ),
        )
        for field in (
            "status",
            "bundle_sha256",
            "observed_cost_usd",
            "ambiguous_cost_reserve_usd",
            "cost_exposure_upper_bound_usd",
            "cost_uncertainty_reasons",
        ):
            if attempt.get(field) != expected_attempt.get(field):
                raise ValueError(
                    "replacement resume attempt does not reconcile bundle: "
                    f"{coordinate_id}:{field}"
                )

        observed, ambiguous, reasons = _cost_evidence(bundle.model_calls)
        observed_cost = round(observed_cost + observed, 12)
        ambiguous_cost = round(ambiguous_cost + ambiguous, 12)
        uncertainty_reasons.extend(reasons)

    if completed_from_bundles != completed:
        raise ValueError(
            "replacement resume completed coordinates do not match retained bundles"
        )
    for coordinate_id, reason in unavailable.items():
        if coordinate_id in unavailable_from_bundles:
            if unavailable_from_bundles[coordinate_id] != reason:
                raise ValueError(
                    "replacement resume unavailable reason does not match bundle: "
                    f"{coordinate_id}"
                )
        elif reason != "cohort cost cap reached":
            raise ValueError(
                "replacement resume unavailable coordinate has no retained bundle: "
                f"{coordinate_id}"
            )
    if set(unavailable_from_bundles) != {
        coordinate_id
        for coordinate_id, reason in unavailable.items()
        if reason != "cohort cost cap reached"
    }:
        raise ValueError(
            "replacement resume unavailable coordinates do not match retained bundles"
        )
    if abs(ledger.observed_cost_usd - observed_cost) > 1e-12:
        raise ValueError("replacement resume observed cost does not reconcile bundles")
    if abs(ledger.ambiguous_cost_reserve_usd - ambiguous_cost) > 1e-12:
        raise ValueError("replacement resume ambiguous cost does not reconcile bundles")
    if ledger.cost_uncertainty_reasons != uncertainty_reasons:
        raise ValueError(
            "replacement resume uncertainty reasons do not reconcile bundles"
        )


def _ensure_live_telemetry_marker(
    *,
    cohort_dir: Path,
    cohort: str,
    preview_sha: str,
) -> None:
    """Bind a fresh run directory to its one live Agent Lens destination."""

    destination = LIVE_AGENT_LENS_PROJECTS.get(cohort)
    if destination is None:
        raise ValueError(f"no fixed live Agent Lens destination for {cohort!r}")
    expected = {
        "schema_version": 1,
        "mode": "live_application_instrumentation",
        "preview_sha256": preview_sha,
        "destination_project": destination,
    }
    marker_path = cohort_dir / "live-agent-lens.json"
    if marker_path.exists() or marker_path.is_symlink():
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError("live Agent Lens marker must be a regular file")
        if load_json(marker_path) != expected:
            raise ValueError("live Agent Lens marker identity drift")
        return
    if cohort_dir.exists() and any(cohort_dir.iterdir()):
        raise ValueError(
            "live Agent Lens cannot be enabled after run artifacts already exist"
        )
    cohort_dir.mkdir(parents=True, exist_ok=True)
    _write_exclusive(marker_path, expected)


def _run_cohort_impl(
    cohort: str,
    *,
    approve_preview_sha: str,
    max_new_coordinates: int | None = None,
    concurrency: int = 4,
    live_telemetry: LiveTelemetrySession | None = None,
) -> RunLedger:
    # The operator-approved digest selects the retained runtime directly. This
    # deliberately avoids rebuilding a preview from mutable authoring inputs on
    # resume; every paid input is loaded from and re-hashed in the snapshot.
    runtime = verify_runtime_snapshot(approve_preview_sha)
    preview = dict(runtime.preview)
    continue_frozen_attempt_failures = continues_frozen_attempt_failures(preview)
    cohorts = preview.get("cohorts")
    if not isinstance(cohorts, dict) or cohort not in cohorts:
        raise ValueError("cohort is not present in the approved preview")
    if approve_preview_sha != preview["preview_sha256"]:
        raise ValueError("approval digest does not match the exact current preview")
    if max_new_coordinates is not None and max_new_coordinates <= 0:
        raise ValueError("max_new_coordinates must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    coordinates = [
        Coordinate.model_validate(value)
        for value in cohorts[cohort]["coordinates"]
    ]
    tasks = list(runtime.tasks)
    tasks_by_id = {str(task["id"]): task for task in tasks}
    cohort_dir = RUNS_DIR / cohort / preview["preview_sha256"]
    ledger_path = cohort_dir / "ledger.json"
    planned_coordinate_ids = [item.coordinate_id for item in coordinates]
    resume_ledger = _load_resume_ledger(
        ledger_path,
        cohort=cohort,
        preview_sha=preview["preview_sha256"],
        planned_coordinate_ids=planned_coordinate_ids,
    )
    if (
        live_telemetry is not None
        and resume_ledger is not None
        and resume_ledger.status == "uncertain"
    ):
        raise RuntimeError(
            "live telemetry delivery is uncertain; this run cannot be resumed"
        )
    canary_execution = preview.get("canary_execution")
    execution_generation = (
        int(canary_execution.get("execution_generation", 1))
        if isinstance(canary_execution, dict)
        else 1
    )
    if cohort == "canary" and execution_generation > 1 and resume_ledger is None:
        _require_fresh_replacement_directory(cohort_dir)
    elif cohort == "canary" and execution_generation > 1:
        _validate_replacement_resume_artifacts(
            cohort_dir=cohort_dir,
            ledger=resume_ledger,
            coordinates=coordinates,
            preview_sha=preview["preview_sha256"],
            continue_frozen_attempt_failures=(
                continue_frozen_attempt_failures
            ),
        )

    if live_telemetry is not None:
        _ensure_live_telemetry_marker(
            cohort_dir=cohort_dir,
            cohort=cohort,
            preview_sha=preview["preview_sha256"],
        )

    bundle_dir = cohort_dir / "bundles"
    attempt_dir = cohort_dir / "attempts"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    completed: list[str] = []
    unavailable: dict[str, str] = {}
    bundle_digests: dict[str, str] = {}
    cap = float(preview["cost_caps_usd"][cohort])
    budget = CostBudget(
        cap,
        price_lock=dict(runtime.pricing_lock),
    )
    pending: list[Coordinate] = []
    unreconciled_coordinates: list[str] = []
    final_status: str | None = None

    for coordinate in coordinates:
        bundle_path = bundle_dir / f"{coordinate.coordinate_id}.json"
        attempt_path = attempt_dir / f"{coordinate.coordinate_id}.json"
        attempt = _read_attempt_receipt(
            attempt_path,
            coordinate=coordinate,
            preview_sha=preview["preview_sha256"],
        )
        reused_bundle = bundle_path.exists() or bundle_path.is_symlink()
        if reused_bundle:
            if pending:
                raise ValueError(
                    "retained bundles must form a deterministic coordinate prefix"
                )
            if bundle_path.is_symlink() or not bundle_path.is_file():
                raise ValueError(
                    f"existing bundle must be a regular file: {coordinate.coordinate_id}"
                )
            bundle = ConversationBundle.model_validate(load_json(bundle_path))
            if bundle.coordinate != coordinate:
                raise ValueError(
                    f"existing bundle coordinate drift: {coordinate.coordinate_id}"
                )
            if bundle.bundle_sha256 != digest_value(bundle.unsigned_payload()):
                raise ValueError(
                    f"existing bundle content drift: {coordinate.coordinate_id}"
                )
            if attempt is None:
                raise ValueError(
                    f"existing bundle has no attempt receipt: {coordinate.coordinate_id}"
                )
            expected_attempt = _attempt_from_bundle(
                coordinate=coordinate,
                preview_sha=preview["preview_sha256"],
                bundle=bundle,
                started_at=attempt.get("started_at"),
                reconciled_at=_now(),
                continue_frozen_attempt_failures=(
                    continue_frozen_attempt_failures
                ),
            )
            if attempt.get("status") in {"started", "uncertain"}:
                write_json_atomic(attempt_path, expected_attempt)
                attempt = expected_attempt
            else:
                for field in (
                    "status",
                    "bundle_sha256",
                    "observed_cost_usd",
                    "ambiguous_cost_reserve_usd",
                    "cost_exposure_upper_bound_usd",
                    "cost_uncertainty_reasons",
                ):
                    if attempt.get(field) != expected_attempt.get(field):
                        raise ValueError(
                            "attempt receipt does not reconcile frozen bundle: "
                            f"{coordinate.coordinate_id}:{field}"
                        )

            _record_bundle_cost(budget, bundle)
            bundle_digests[coordinate.coordinate_id] = bundle.bundle_sha256
            if bundle.status == "complete":
                completed.append(coordinate.coordinate_id)
            else:
                unavailable[coordinate.coordinate_id] = str(
                    bundle.unavailable_reason
                )

            has_disallowed_model_evidence = bundle_has_disallowed_model_evidence(
                bundle
            )
            if (
                not continue_frozen_attempt_failures
                and (
                    budget.is_uncertain
                    or has_disallowed_model_evidence
                )
            ) or (
                continue_frozen_attempt_failures
                and has_disallowed_model_evidence
            ):
                final_status = "uncertain"
            elif (
                final_status is None
                and bundle.status == "unavailable"
                and "cost cap would be exceeded"
                in str(bundle.unavailable_reason)
            ):
                final_status = "blocked_budget"
        else:
            if attempt is not None:
                if attempt["status"] in {"started", "uncertain"}:
                    reason = (
                        "prior paid attempt has no frozen bundle; remote outcome "
                        "is uncertain and automatic regeneration is forbidden"
                    )
                    uncertain = _seal_attempt(
                        {
                            **attempt,
                            "status": "uncertain",
                            "detected_at": _now(),
                            "cost_exposure_unknown": True,
                        }
                    )
                    write_json_atomic(attempt_path, uncertain)
                    unavailable[coordinate.coordinate_id] = reason
                    unreconciled_coordinates.append(coordinate.coordinate_id)
                    final_status = "uncertain"
                    continue
                raise ValueError(
                    f"attempt receipt has no matching bundle: {coordinate.coordinate_id}"
                )
            pending.append(coordinate)

    if unreconciled_coordinates:
        remaining_reserve = round(
            cap - budget.cost_exposure_upper_bound_usd,
            12,
        )
        if remaining_reserve <= 0:
            raise RuntimeError(
                "unknown coordinate cost cannot be reserved within the cohort cap"
            )
        for index, coordinate_id in enumerate(sorted(unreconciled_coordinates)):
            budget.record_ambiguous(
                remaining_reserve if index == 0 else 0.0,
                reason=(
                    "unreconciled paid attempt reserved to the remaining "
                    f"cohort cap: {coordinate_id}"
                ),
            )

    selected_pending = pending
    if max_new_coordinates is not None:
        selected_pending = pending[:max_new_coordinates]

    if final_status is None and selected_pending:
        worker_count = min(concurrency, len(selected_pending))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="retail-coordinate",
        ) as executor:
            next_index = 0
            while next_index < len(selected_pending):
                remaining_cap = round(
                    cap - budget.cost_exposure_upper_bound_usd,
                    12,
                )
                if remaining_cap <= 0:
                    coordinate = selected_pending[next_index]
                    unavailable[coordinate.coordinate_id] = "cohort cost cap reached"
                    final_status = "blocked_budget"
                    break

                wave = selected_pending[
                    next_index : next_index + worker_count
                ]
                coordinate_cap = remaining_cap / len(wave)

                # Re-hash the retained inputs before preparing each paid wave.
                # The snapshot is immutable, and every coordinate then receives
                # an independent inference client and MCP subprocess/database.
                for _coordinate in wave:
                    current_runtime = verify_runtime_snapshot(
                        preview["preview_sha256"]
                    )
                    if current_runtime.manifest != runtime.manifest:
                        raise ValueError(
                            "runtime snapshot changed during cohort execution"
                        )

                if _traffic_rows(runtime) is None:
                    clients: list[
                        WandbInferenceClient | _CoordinateInferenceClients
                    ] = [
                        WandbInferenceClient(
                            cost_cap_usd=coordinate_cap,
                            price_lock=dict(runtime.pricing_lock),
                        )
                        for _coordinate in wave
                    ]
                else:
                    clients = [
                        _coordinate_inference_clients(
                            coordinate,
                            runtime=runtime,
                            cost_cap_usd=coordinate_cap,
                        )
                        for coordinate in wave
                    ]
                started_at_by_id: dict[str, str] = {}
                for coordinate in wave:
                    started_at = _now()
                    started_at_by_id[coordinate.coordinate_id] = started_at
                    _write_exclusive(
                        attempt_dir / f"{coordinate.coordinate_id}.json",
                        _seal_attempt(
                            {
                                **_receipt_identity(
                                    coordinate,
                                    preview["preview_sha256"],
                                ),
                                "status": "started",
                                "started_at": started_at,
                            }
                        ),
                    )

                futures: list[
                    tuple[
                        Coordinate,
                        WandbInferenceClient | _CoordinateInferenceClients,
                        Future[ConversationBundle],
                    ]
                ] = []
                for coordinate, client in zip(wave, clients, strict=True):
                    if isinstance(client, _CoordinateInferenceClients):
                        future = executor.submit(
                            _run_coordinate_with_client,
                            coordinate,
                            user_client=client.user,
                            support_client=client.support,
                            budget=client.budget,
                            tasks_by_id=tasks_by_id,
                            runtime=runtime,
                            live_telemetry=live_telemetry,
                        )
                    else:
                        future = executor.submit(
                            _run_coordinate_with_client,
                            coordinate,
                            client=client,
                            tasks_by_id=tasks_by_id,
                            runtime=runtime,
                            live_telemetry=live_telemetry,
                        )
                    futures.append((coordinate, client, future))

                wave_status: str | None = None
                unknown_cost_coordinates: list[str] = []
                for coordinate, client, future in futures:
                    bundle_path = bundle_dir / f"{coordinate.coordinate_id}.json"
                    attempt_path = attempt_dir / f"{coordinate.coordinate_id}.json"
                    try:
                        bundle = future.result()
                    except Exception as exc:
                        reason = (
                            "prior paid attempt has no frozen bundle; coordinate "
                            "worker failed after its paid-attempt boundary, so "
                            "remote cost and response outcome are uncertain: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        unavailable[coordinate.coordinate_id] = reason
                        unknown_cost_coordinates.append(
                            coordinate.coordinate_id
                        )
                        write_json_atomic(
                            attempt_path,
                            _seal_attempt(
                                {
                                    **_receipt_identity(
                                        coordinate,
                                        preview["preview_sha256"],
                                    ),
                                    "status": "uncertain",
                                    "started_at": started_at_by_id[
                                        coordinate.coordinate_id
                                    ],
                                    "detected_at": _now(),
                                    "bundle_sha256": None,
                                    "observed_cost_usd": 0.0,
                                    "ambiguous_cost_reserve_usd": coordinate_cap,
                                    "cost_exposure_upper_bound_usd": coordinate_cap,
                                    "cost_uncertainty_reasons": [reason],
                                    "cost_exposure_unknown": True,
                                }
                            ),
                        )
                        wave_status = "uncertain"
                    else:
                        # Only the coordinator thread freezes public artifacts and
                        # mutates aggregate accounting, preserving deterministic
                        # preview order even when workers finish out of order.
                        write_json_atomic(
                            bundle_path,
                            bundle.model_dump(mode="json"),
                        )
                        write_json_atomic(
                            attempt_path,
                            _attempt_from_bundle(
                                coordinate=coordinate,
                                preview_sha=preview["preview_sha256"],
                                bundle=bundle,
                                started_at=started_at_by_id[
                                    coordinate.coordinate_id
                                ],
                                completed_at=_now(),
                                continue_frozen_attempt_failures=(
                                    continue_frozen_attempt_failures
                                ),
                            ),
                        )
                        _record_bundle_cost(budget, bundle)
                        bundle_digests[coordinate.coordinate_id] = (
                            bundle.bundle_sha256
                        )
                        if bundle.status == "complete":
                            completed.append(coordinate.coordinate_id)
                        else:
                            unavailable[coordinate.coordinate_id] = str(
                                bundle.unavailable_reason
                            )

                        has_disallowed_model_evidence = (
                            bundle_has_disallowed_model_evidence(bundle)
                        )
                        if (
                            not continue_frozen_attempt_failures
                            and (
                                budget.is_uncertain
                                or has_disallowed_model_evidence
                            )
                        ) or (
                            continue_frozen_attempt_failures
                            and has_disallowed_model_evidence
                        ):
                            wave_status = "uncertain"
                        elif (
                            wave_status is None
                            and bundle.status == "unavailable"
                            and "cost cap would be exceeded"
                            in str(bundle.unavailable_reason)
                        ):
                            wave_status = "blocked_budget"

                    if wave_status is None:
                        _write_ledger(
                            path=ledger_path,
                            cohort=cohort,
                            preview_sha=preview["preview_sha256"],
                            coordinates=coordinates,
                            completed=completed,
                            unavailable=unavailable,
                            bundle_digests=bundle_digests,
                            budget=budget,
                            status="running",
                        )

                if live_telemetry is not None:
                    try:
                        live_telemetry.force_flush()
                    except RuntimeError:
                        _write_ledger(
                            path=ledger_path,
                            cohort=cohort,
                            preview_sha=preview["preview_sha256"],
                            coordinates=coordinates,
                            completed=completed,
                            unavailable=unavailable,
                            bundle_digests=bundle_digests,
                            budget=budget,
                            status="uncertain",
                        )
                        raise

                if unknown_cost_coordinates:
                    remaining_reserve = round(
                        cap - budget.cost_exposure_upper_bound_usd,
                        12,
                    )
                    if remaining_reserve <= 0:
                        raise RuntimeError(
                            "unknown coordinate cost cannot be reserved within "
                            "the cohort cap"
                        )
                    for index, coordinate_id in enumerate(
                        sorted(unknown_cost_coordinates)
                    ):
                        budget.record_ambiguous(
                            remaining_reserve if index == 0 else 0.0,
                            reason=(
                                "unreconciled paid attempt reserved to the remaining "
                                f"cohort cap: {coordinate_id}"
                            ),
                        )

                if wave_status is not None:
                    _write_ledger(
                        path=ledger_path,
                        cohort=cohort,
                        preview_sha=preview["preview_sha256"],
                        coordinates=coordinates,
                        completed=completed,
                        unavailable=unavailable,
                        bundle_digests=bundle_digests,
                        budget=budget,
                        status=wave_status,
                    )

                next_index += len(wave)
                if wave_status is not None:
                    final_status = wave_status
                    break

    if final_status is None:
        final_status = (
            "paused"
            if len(selected_pending) < len(pending)
            else "complete"
        )
    return _write_ledger(
        path=ledger_path,
        cohort=cohort,
        preview_sha=preview["preview_sha256"],
        coordinates=coordinates,
        completed=completed,
        unavailable=unavailable,
        bundle_digests=bundle_digests,
        budget=budget,
        status=final_status,
    )


def run_cohort(
    cohort: str,
    *,
    approve_preview_sha: str,
    max_new_coordinates: int | None = None,
    concurrency: int = 4,
    live_agent_lens: bool = False,
) -> RunLedger:
    """Run a cohort, requiring execution-time telemetry when preview-bound."""

    runtime = verify_runtime_snapshot(approve_preview_sha)
    telemetry = runtime.preview.get("telemetry")
    if not isinstance(telemetry, dict):
        if live_agent_lens:
            raise ValueError("approved preview has no telemetry contract")
        return _run_cohort_impl(
            cohort,
            approve_preview_sha=approve_preview_sha,
            max_new_coordinates=max_new_coordinates,
            concurrency=concurrency,
        )
    live_required = telemetry.get("run_requires_live_agent_lens") is True
    if live_required and not live_agent_lens:
        raise ValueError(
            "approved preview requires --live-agent-lens; replay is not permitted"
        )
    if live_agent_lens:
        if telemetry.get("mode") != "live_application_instrumentation":
            raise ValueError("approved preview is not bound to live Agent Lens telemetry")
        destinations = telemetry.get("destinations")
        if not isinstance(destinations, dict) or destinations != dict(
            LIVE_AGENT_LENS_PROJECTS
        ):
            raise ValueError("approved preview live destinations drifted")
        with LiveTelemetrySession.open(
            cohort=cohort,
            agent_version=approve_preview_sha,
        ) as session:
            return _run_cohort_impl(
                cohort,
                approve_preview_sha=approve_preview_sha,
                max_new_coordinates=max_new_coordinates,
                concurrency=concurrency,
                live_telemetry=session,
            )
    return _run_cohort_impl(
        cohort,
        approve_preview_sha=approve_preview_sha,
        max_new_coordinates=max_new_coordinates,
        concurrency=concurrency,
    )
