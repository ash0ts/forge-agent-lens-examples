from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from .canonical import digest_value
from .confirmation import MutationConfirmationController
from .constants import PACKAGE_ROOT, SUPPORT_MAX_OUTPUT_TOKENS
from .inference import (
    AnthropicInferenceClient,
    InferenceCallError,
    SupportInferenceClient,
    WandbInferenceClient,
)
from .jev import (
    JevCallRunner,
    JevConfig,
    JevDecisionEngine,
    JevRoutingDecision,
    JevTurnAssessment,
    SystemOneClient,
)
from .live_telemetry import (
    _EXPORT_OVERRIDE_ENVIRONMENT,
    _EXPORT_TIMEOUT_SECONDS,
    _OTLP_ENDPOINT,
    LiveAgentTurn,
    LiveTelemetrySession,
)
from .mcp_protocol import MCPProcessClient, MCPTimeoutError, MCPToolError
from .schemas import AgentProfile, ModelCall, ModelRoute, ToolCallRecord
from .traffic_config import (
    ALL_TOOL_NAMES,
    build_agent_profiles,
    build_behavior_profile,
    build_model_routes,
    route_agent_profile_id,
)

DEFAULT_AGENT_LENS_PROJECT = "wandb/agent-lens-retail-support-live-v10"
DEFAULT_INFERENCE_PROJECT = "wandb/agent-lens-retail-chat-demo-v1"
DEFAULT_ROUTE_ID = "wandb-glm-5-3-flash"
DEFAULT_CUSTOMER_ID = "retail_user_0001"
DEFAULT_CUSTOMER_NAME = "Avery Arbor"
DEFAULT_CUSTOMER_EMAIL = "customer-0001@example.invalid"
DEFAULT_GREETING = "Hi! How can I help with your order today?"
PROVISIONAL_AGENT_NAME = "retail-support-router"
DEFAULT_IDLE_TTL_SECONDS = 30 * 60
DEFAULT_SESSION_COST_CAP_USD = 20.0
MAX_MESSAGE_CHARACTERS = 4_000
MAX_TOOL_STEPS_PER_MESSAGE = 16

InferenceFactory = Callable[[ModelRoute, float, str], SupportInferenceClient]
MCPFactory = Callable[..., MCPProcessClient]
TelemetryFactory = Callable[[str], LiveTelemetrySession | None]


class ChatRuntimeError(RuntimeError):
    """A bounded error safe for the HTTP adapter to translate."""


class SessionNotFound(ChatRuntimeError):
    pass


class SessionBusy(ChatRuntimeError):
    pass


def _project_url(project: str) -> str:
    return f"https://forge.coreweave.com/agent-lens/{project}"


def _valid_project(value: str, *, field: str) -> str:
    project = value.strip()
    if not project.startswith("wandb/") or project.count("/") != 1:
        raise ValueError(f"{field} must be a wandb/project path")
    return project


def _open_live_telemetry(project: str) -> LiveTelemetrySession | None:
    if os.environ.get("AGENT_LENS_ENABLED", "true").casefold() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        return None
    if api_key != api_key.strip() or "\n" in api_key or "\r" in api_key:
        raise ValueError("WANDB_API_KEY must be a non-empty single-line value")
    configured = sorted(
        name
        for name in _EXPORT_OVERRIDE_ENVIRONMENT
        if os.environ.get(name, "").strip()
    )
    if configured:
        raise ValueError(
            "Agent Lens telemetry rejects OTLP transport overrides; unset: "
            + ", ".join(configured)
        )
    from coreweave.forge.agentlens import tracing

    session = tracing.init(
        project,
        api_key=api_key,
        service_name="agent-lens-retail-chat-demo",
        autopatch_integrations=False,
    )
    return LiveTelemetrySession.from_provider(
        session.provider,
        agent_version="retail-chat-demo-v1",
    )


def _default_inference_factory(
    route: ModelRoute,
    cost_cap_usd: float,
    inference_project: str,
) -> SupportInferenceClient:
    if route.provider_name == "wandb":
        return WandbInferenceClient(
            cost_cap_usd=cost_cap_usd,
            inference_project=inference_project,
        )
    if route.provider_name == "anthropic":
        return AnthropicInferenceClient(cost_cap_usd=cost_cap_usd)
    raise ValueError(f"unsupported provider: {route.provider_name}")


def _route_public(
    route: ModelRoute, *, available: bool | None = None
) -> dict[str, Any]:
    provider_label = "Forge Inference" if route.provider_name == "wandb" else "Anthropic"
    value: dict[str, Any] = {
        "id": route.route_id,
        "provider": route.provider_name,
        "provider_label": provider_label,
        "model": route.model,
    }
    if available is not None:
        value["available"] = available
    return value


def _profile_public(profile: AgentProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "id": profile.profile_id,
        "name": profile.agent_name,
        "role": profile.role,
        "purpose": profile.purpose,
    }


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "password",
    "payment_method",
    "secret",
    "token",
    "user_id",
)


def _public_value(value: Any, *, key: str = "") -> Any:
    normalized_key = key.casefold()
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(child_key): _public_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_public_value(child, key=key) for child in value]
    if isinstance(value, tuple):
        return [_public_value(child, key=key) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_summary(result: Any) -> dict[str, Any] | list[Any] | str | None:
    projected = _public_value(result)
    if isinstance(projected, dict):
        preferred = (
            "status",
            "success",
            "message",
            "order_id",
            "product_name",
            "action_prepared",
        )
        selected = {key: projected[key] for key in preferred if key in projected}
        if selected:
            return selected
        return {"fields": sorted(projected)[:12]}
    if isinstance(projected, list):
        return {"items": len(projected)}
    if isinstance(projected, str):
        return projected[:240]
    return projected


def _usage_public(records: list[ModelCall]) -> dict[str, Any]:
    observed = [
        call.usage.observed_cost_usd
        for call in records
        if call.usage.observed_cost_usd is not None
    ]
    return {
        "model_calls": len(records),
        "input_tokens": sum(call.usage.input_tokens for call in records),
        "cached_input_tokens": sum(call.usage.cached_input_tokens for call in records),
        "output_tokens": sum(call.usage.output_tokens for call in records),
        "total_tokens": sum(call.usage.total_tokens for call in records),
        "observed_cost_usd": round(sum(observed), 8) if observed else None,
    }


class ChatSession:
    """One synchronous, stateful customer-support conversation."""

    def __init__(
        self,
        *,
        route: ModelRoute,
        profiles: Mapping[str, AgentProfile],
        db_source: Path,
        policy_path: Path,
        support_prompt: str,
        customer_id: str,
        customer: Mapping[str, str],
        behavior_profile: Mapping[str, Any],
        inference_project: str,
        lens_project: str,
        inference_factory: InferenceFactory,
        mcp_factory: MCPFactory,
        telemetry: LiveTelemetrySession | None,
        jev_engine: JevDecisionEngine,
        jev_call_runner: JevCallRunner | None,
        cost_cap_usd: float,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.route = route
        self.profile: AgentProfile | None = None
        self.customer_id = customer_id
        self.customer = dict(customer)
        self.inference_project = inference_project
        self.lens_project = lens_project
        self.messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": DEFAULT_GREETING}
        ]
        self.model_calls: list[ModelCall] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.tool_events: list[dict[str, Any]] = []
        self.confirmation = MutationConfirmationController(conversation_id=self.id)
        self.created_at = time.time()
        self.last_access_monotonic = time.monotonic()
        self.lock = threading.Lock()
        self._profiles = dict(profiles)
        self._policy_path = policy_path
        self._policy = policy_path.read_text(encoding="utf-8")
        self._support_prompt = support_prompt
        self._behavior_profile = dict(behavior_profile)
        self._mcp_factory = mcp_factory
        self._telemetry = telemetry
        self._jev_engine = jev_engine
        self._jev_call_runner = jev_call_runner
        if self._jev_call_runner is None and self._telemetry is not None:
            self._jev_call_runner = self._telemetry.jev_call_runner(self.id)
        self.jev_routing: JevRoutingDecision | None = None
        self.jev_assessments: list[JevTurnAssessment] = []
        self._mcp: MCPProcessClient | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="retail-chat-")
        self._db_path = Path(self._temporary.name) / "db.json"
        shutil.copy2(db_source, self._db_path)
        self._client = inference_factory(route, cost_cap_usd, inference_project)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            close_client()
        self._temporary.cleanup()

    def _touch(self) -> None:
        self.last_access_monotonic = time.monotonic()

    def _ensure_profile_and_mcp(
        self,
        first_message: str,
        active_turn: LiveAgentTurn | None,
    ) -> None:
        if self.profile is not None:
            return
        deterministic_profile_id = route_agent_profile_id(first_message)
        self.jev_routing = self._jev_engine.route(
            first_message,
            deterministic_profile_id,
            call_runner=self._jev_call_runner,
        )
        self.profile = self._profiles[self.jev_routing.applied_profile_id]
        if active_turn is not None:
            active_turn.resolve_agent(self.profile.agent_name)
        self._mcp = self._mcp_factory(
            db_path=self._db_path,
            policy_path=self._policy_path,
            authenticated_user_id=self.customer_id,
            allowed_tool_names=self.profile.tool_names,
            behavior_profile=self._behavior_profile,
            conversation_id=self.id,
        )
        self._mcp.start()

    def _system_prompt(self) -> str:
        if self.profile is None:  # pragma: no cover - guarded by send
            raise RuntimeError("agent profile has not been selected")
        return (
            f"{self._support_prompt}\n\n"
            f"<retail-policy>\n{self._policy}\n</retail-policy>\n\n"
            "<assigned-support-role>\n"
            f"You are the {self.profile.agent_name} agent. "
            f"{self.profile.purpose}\n"
            "Use only the tools made available for this role.\n"
            "</assigned-support-role>"
        )

    def _telemetry_turn(self, user_message: Mapping[str, Any]) -> LiveAgentTurn | None:
        if self._telemetry is None:
            return None
        return self._telemetry.start_turn(
            self.id,
            [user_message],
            {record.call_id: record for record in self.tool_calls},
            agent_name=(
                self.profile.agent_name
                if self.profile is not None
                else PROVISIONAL_AGENT_NAME
            ),
            requested_model=self.route.model,
            requested_temperature=self.route.temperature,
            provider_name=self.route.provider_name,
            server_address=self.route.server_address,
            server_port=443,
        )

    def _complete(self, request: dict[str, Any]) -> Any:
        try:
            completion = self._client.complete(**request)
        except InferenceCallError as exc:
            self.model_calls.append(exc.record)
            raise
        self.model_calls.append(completion.record)
        return completion

    def _event(self, record: ToolCallRecord) -> dict[str, Any]:
        event = {
            "id": record.call_id,
            "name": record.name,
            "status": record.execution_disposition,
            "duration_ms": round(record.duration_ms, 1),
            "arguments": _public_value(record.arguments),
            "result": _result_summary(record.result),
            "error_type": record.error_type,
        }
        self.tool_events.append(event)
        return event

    def _tool_call(
        self,
        *,
        raw: Mapping[str, Any],
        active_turn: LiveAgentTurn | None,
    ) -> ToolCallRecord:
        if self._mcp is None:
            raise RuntimeError("MCP process has not started")
        call_id = str(raw.get("id") or "")
        function = raw.get("function")
        if not call_id or not isinstance(function, Mapping):
            raise ValueError("model returned a malformed tool call")
        name = str(function.get("name") or "")
        try:
            arguments = json.loads(str(function.get("arguments") or "{}"))
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            arguments = self._mcp.validate_tool_arguments(name, arguments)
        except Exception as exc:
            return ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments={},
                error_type="invalid_tool_arguments",
                error_message=type(exc).__name__,
                execution_disposition="rejected",
                duration_ms=0.0,
            )

        started = time.perf_counter()
        gate = self.confirmation.guard_write(
            tool_name=name,
            arguments=arguments,
            tool_call_id=call_id,
            message_index=len(self.messages) - 1,
        )
        if not gate.authorized and gate.decision == "awaiting_confirmation":
            return ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=arguments,
                error_type="confirmation_still_pending",
                error_message="a new standalone customer confirmation is required",
                execution_disposition="rejected",
                duration_ms=(time.perf_counter() - started) * 1_000,
            )
        if not gate.authorized:
            if gate.public_result is None:  # pragma: no cover - controller invariant
                raise RuntimeError("prepared write has no public result")
            return ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=arguments,
                result=gate.public_result,
                execution_disposition="prepared",
                duration_ms=(time.perf_counter() - started) * 1_000,
            )

        try:
            result = (
                active_turn.execute_mcp(
                    lambda record_request_id: self._mcp.call_tool(
                        name,
                        arguments,
                        on_request_id=record_request_id,
                    ),
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                    session_id=self._mcp.session_id,
                )
                if active_turn is not None
                else self._mcp.call_tool(name, arguments)
            )
            return ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=arguments,
                result=result,
                execution_disposition="executed",
                duration_ms=(time.perf_counter() - started) * 1_000,
            )
        except Exception as exc:
            if isinstance(exc, MCPToolError):
                error_type = exc.code
            elif isinstance(exc, MCPTimeoutError):
                error_type = "timeout"
            else:
                error_type = type(exc).__name__
            return ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=arguments,
                error_type=error_type,
                error_message=error_type,
                execution_disposition="executed",
                duration_ms=(time.perf_counter() - started) * 1_000,
            )

    def send(self, content: str) -> dict[str, Any]:
        message_text = content.strip()
        if not message_text:
            raise ValueError("message must not be empty")
        if len(message_text) > MAX_MESSAGE_CHARACTERS:
            raise ValueError(
                f"message must be at most {MAX_MESSAGE_CHARACTERS} characters"
            )
        if not self.lock.acquire(blocking=False):
            raise SessionBusy("this conversation is already processing a message")
        active_turn: LiveAgentTurn | None = None
        new_events: list[dict[str, Any]] = []
        call_start = len(self.model_calls)
        try:
            if self._closed:
                raise SessionNotFound("conversation is closed")
            self._touch()
            prior_visible_messages = self.public_messages()
            user_message = {"role": "user", "content": message_text}
            self.messages.append(user_message)
            self.confirmation.observe_user_message(
                text=message_text,
                message_index=len(self.messages) - 1,
            )
            active_turn = self._telemetry_turn(user_message)
            bind_turn = getattr(self._jev_call_runner, "bind_turn", None)
            clear_turn = getattr(self._jev_call_runner, "clear_turn", None)
            if active_turn is not None and callable(bind_turn):
                bind_turn(active_turn)
            self._ensure_profile_and_mcp(message_text, active_turn)
            assert self.profile is not None
            assert self._mcp is not None
            tool_defs = self._mcp.list_openai_tools()

            for _ in range(MAX_TOOL_STEPS_PER_MESSAGE):
                request: dict[str, Any] = {
                    "participant": "support_agent",
                    "model": self.route.model,
                    "temperature": self.route.temperature,
                    "enable_thinking": self.route.enable_thinking,
                    "seed": None,
                    "messages": [
                        {"role": "system", "content": self._system_prompt()},
                        *self.messages,
                    ],
                    "tools": tool_defs,
                    "max_tokens": SUPPORT_MAX_OUTPUT_TOKENS,
                }

                def complete() -> Any:
                    return self._complete(request)

                completion = (
                    active_turn.complete_support_call(
                        complete,
                        self.model_calls,
                        request,
                    )
                    if active_turn is not None
                    else complete()
                )
                output = completion.message
                raw_calls = output.get("tool_calls") or []
                if raw_calls:
                    self.messages.append(output)
                    if len(raw_calls) > 1:
                        for raw in raw_calls:
                            function = raw.get("function") or {}
                            record = ToolCallRecord(
                                call_id=str(raw.get("id") or ""),
                                name=str(function.get("name") or ""),
                                arguments={},
                                error_type="parallel_tool_calls_rejected",
                                error_message="one tool call is allowed at a time",
                                execution_disposition="rejected",
                                duration_ms=0.0,
                            )
                            self.tool_calls.append(record)
                            new_events.append(self._event(record))
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": record.call_id,
                                    "content": json.dumps(
                                        {"error": record.error_type},
                                        sort_keys=True,
                                    ),
                                }
                            )
                        continue
                    record = self._tool_call(
                        raw=raw_calls[0],
                        active_turn=active_turn,
                    )
                    self.tool_calls.append(record)
                    new_events.append(self._event(record))
                    content_value = (
                        record.result
                        if record.error_type is None
                        else {
                            "error": record.error_type,
                            "message": record.error_message,
                        }
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": record.call_id,
                            "content": json.dumps(
                                content_value,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }
                    )
                    continue

                assistant = str(output.get("content") or "").strip()
                if not assistant:
                    raise RuntimeError(
                        "support agent returned neither content nor a tool call"
                    )
                assistant_message = {"role": "assistant", "content": assistant}
                self.messages.append(assistant_message)
                self.confirmation.observe_assistant_message(
                    text=assistant,
                    message_index=len(self.messages) - 1,
                )
                assessment = self._jev_engine.assess_turn(
                    user_message=message_text,
                    assistant_message=assistant,
                    recent_messages=prior_visible_messages,
                    tool_events=new_events,
                    call_runner=self._jev_call_runner,
                )
                if assessment is not None:
                    self.jev_assessments.append(assessment)
                if active_turn is not None:
                    active_turn.finish(
                        assistant_message,
                        {record.call_id: record for record in self.tool_calls},
                    )
                    active_turn = None
                    assert self._telemetry is not None
                    self._telemetry.force_flush()
                self._touch()
                return self.response(
                    assistant=assistant,
                    tool_events=new_events,
                    usage_records=self.model_calls[call_start:],
                )
            raise RuntimeError("per-message tool step cap reached")
        except BaseException as exc:
            if active_turn is not None:
                active_turn.fail(exc)
            raise
        finally:
            try:
                clear_turn = getattr(self._jev_call_runner, "clear_turn", None)
                if callable(clear_turn):
                    clear_turn()
            finally:
                self.lock.release()

    def public_messages(self) -> list[dict[str, str]]:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in self.messages
            if message.get("role") in {"user", "assistant"}
            and message.get("content")
            and not message.get("tool_calls")
        ]

    def response(
        self,
        *,
        assistant: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        usage_records: list[ModelCall] | None = None,
    ) -> dict[str, Any]:
        return {
            "conversation_id": self.id,
            "assistant": assistant,
            "profile": _profile_public(self.profile),
            "route": _route_public(self.route),
            "customer": dict(self.customer),
            "tool_events": list(tool_events or self.tool_events),
            "usage": _usage_public(
                self.model_calls if usage_records is None else usage_records
            ),
            "telemetry": {
                "enabled": self._telemetry is not None,
                "project": self.lens_project,
                "url": _project_url(self.lens_project),
                "conversation_id": self.id,
            },
            "jev": {
                "routing": (
                    self.jev_routing.model_dump(mode="json")
                    if self.jev_routing is not None
                    else None
                ),
                "latest_assessment": (
                    self.jev_assessments[-1].model_dump(mode="json")
                    if self.jev_assessments
                    else None
                ),
            },
            "messages": self.public_messages(),
        }


class ChatRuntime:
    """Single-process session registry for the interactive demo.

    Sessions contain an open MCP subprocess and mutable conversation state. Run
    this demo with one application worker. Idle sessions are reaped on API
    access; a durable or multi-worker deployment should use an external store.
    """

    def __init__(
        self,
        *,
        package_root: Path = PACKAGE_ROOT,
        lens_project: str = DEFAULT_AGENT_LENS_PROJECT,
        inference_project: str = DEFAULT_INFERENCE_PROJECT,
        default_route_id: str = DEFAULT_ROUTE_ID,
        idle_ttl_seconds: int = DEFAULT_IDLE_TTL_SECONDS,
        session_cost_cap_usd: float = DEFAULT_SESSION_COST_CAP_USD,
        inference_factory: InferenceFactory = _default_inference_factory,
        mcp_factory: MCPFactory = MCPProcessClient,
        telemetry_factory: TelemetryFactory = _open_live_telemetry,
        jev_config: JevConfig | None = None,
        jev_client: SystemOneClient | None = None,
        jev_call_runner: JevCallRunner | None = None,
    ) -> None:
        if idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        if session_cost_cap_usd <= 0:
            raise ValueError("session_cost_cap_usd must be positive")
        self.package_root = package_root.resolve()
        self.lens_project = _valid_project(lens_project, field="lens_project")
        self.inference_project = _valid_project(
            inference_project,
            field="inference_project",
        )
        self.default_route_id = default_route_id
        self.idle_ttl_seconds = idle_ttl_seconds
        self.session_cost_cap_usd = session_cost_cap_usd
        self._inference_factory = inference_factory
        self._mcp_factory = mcp_factory
        self._routes = {route.route_id: route for route in build_model_routes()}
        if default_route_id not in self._routes:
            raise ValueError("default route is not configured")
        prompt_path = self.package_root / "prompts" / "support-agent.md"
        self._support_prompt = prompt_path.read_text(encoding="utf-8")
        instructions_sha256 = digest_value(self._support_prompt)
        self._profiles = {
            profile.profile_id: profile
            for profile in build_agent_profiles(
                instructions_sha256=instructions_sha256,
                manifest_tool_names=ALL_TOOL_NAMES,
            )
        }
        self._behavior_profile = build_behavior_profile().model_dump(mode="json")
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self._telemetry = telemetry_factory(self.lens_project)
        self._jev_engine = JevDecisionEngine(jev_config, client=jev_client)
        self._jev_call_runner = jev_call_runner

    @classmethod
    def from_environment(cls, *, package_root: Path = PACKAGE_ROOT) -> "ChatRuntime":
        return cls(
            package_root=package_root,
            lens_project=os.environ.get(
                "AGENT_LENS_PROJECT",
                DEFAULT_AGENT_LENS_PROJECT,
            ),
            inference_project=os.environ.get(
                "WANDB_INFERENCE_PROJECT",
                DEFAULT_INFERENCE_PROJECT,
            ),
            default_route_id=os.environ.get(
                "RETAIL_DEMO_DEFAULT_MODEL_ROUTE",
                DEFAULT_ROUTE_ID,
            ),
            idle_ttl_seconds=int(
                os.environ.get(
                    "RETAIL_DEMO_IDLE_TTL_SECONDS",
                    str(DEFAULT_IDLE_TTL_SECONDS),
                )
            ),
            session_cost_cap_usd=float(
                os.environ.get(
                    "RETAIL_DEMO_SESSION_COST_CAP_USD",
                    str(DEFAULT_SESSION_COST_CAP_USD),
                )
            ),
            jev_config=JevConfig.from_environment(),
        )

    def _credential_available(self, route: ModelRoute) -> bool:
        return bool(os.environ.get(route.credential_env_var, "").strip())

    def config(self) -> dict[str, Any]:
        return {
            "app_name": "Agent Lens Retail Support",
            "lens": {
                "enabled": self._telemetry is not None,
                "project": self.lens_project,
                "url": _project_url(self.lens_project),
            },
            "credentials": {
                "wandb": bool(os.environ.get("WANDB_API_KEY", "").strip()),
                "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
                "values_exposed_to_browser": False,
            },
            "routes": [
                _route_public(
                    route,
                    available=self._credential_available(route),
                )
                for route in self._routes.values()
            ],
            "agents": [_profile_public(profile) for profile in self._profiles.values()],
            "jev": self._jev_engine.public_config(),
            "demo_account": {
                "name": DEFAULT_CUSTOMER_NAME,
                "email": DEFAULT_CUSTOMER_EMAIL,
            },
            "suggestions": [
                f"My email is {DEFAULT_CUSTOMER_EMAIL}. Can you check my latest order?",
                f"I am {DEFAULT_CUSTOMER_EMAIL}. I need to return a damaged item.",
                f"Use {DEFAULT_CUSTOMER_EMAIL}. Can I change a pending order address?",
            ],
            "defaults": {
                "route_id": self.default_route_id,
                "idle_ttl_seconds": self.idle_ttl_seconds,
                "runtime": "single-process",
            },
        }

    def _reap_expired_locked(self) -> None:
        cutoff = time.monotonic() - self.idle_ttl_seconds
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.last_access_monotonic < cutoff and not session.lock.locked()
        ]
        for session_id in expired:
            self._sessions.pop(session_id).close()

    def create(self, *, route_id: str | None = None) -> dict[str, Any]:
        selected_id = route_id or self.default_route_id
        route = self._routes.get(selected_id)
        if route is None:
            raise ValueError("unknown model route")
        if not self._credential_available(route):
            raise ValueError(
                f"{route.provider_name} credentials are not configured on the server"
            )
        session = ChatSession(
            route=route,
            profiles=self._profiles,
            db_source=self.package_root / "prepared" / "db.json",
            policy_path=self.package_root / "prepared" / "policy.md",
            support_prompt=self._support_prompt,
            customer_id=DEFAULT_CUSTOMER_ID,
            customer={
                "name": DEFAULT_CUSTOMER_NAME,
                "email": DEFAULT_CUSTOMER_EMAIL,
            },
            behavior_profile=self._behavior_profile,
            inference_project=self.inference_project,
            lens_project=self.lens_project,
            inference_factory=self._inference_factory,
            mcp_factory=self._mcp_factory,
            telemetry=self._telemetry,
            jev_engine=self._jev_engine,
            jev_call_runner=self._jev_call_runner,
            cost_cap_usd=self.session_cost_cap_usd,
        )
        with self._lock:
            self._reap_expired_locked()
            self._sessions[session.id] = session
        return {
            "id": session.id,
            "greeting": DEFAULT_GREETING,
            "route": _route_public(route),
            "profile": None,
            "customer": dict(session.customer),
        }

    def _get(self, session_id: str) -> ChatSession:
        with self._lock:
            self._reap_expired_locked()
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound("conversation was not found or expired")
        session._touch()
        return session

    def get(self, session_id: str) -> dict[str, Any]:
        return self._get(session_id).response()

    def send(self, session_id: str, content: str) -> dict[str, Any]:
        return self._get(session_id).send(content)

    def delete(self, session_id: str) -> dict[str, bool]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound("conversation was not found or expired")
        if not session.lock.acquire(blocking=False):
            raise SessionBusy("this conversation is already processing a message")
        try:
            with self._lock:
                current = self._sessions.get(session_id)
                if current is not session:
                    raise SessionNotFound("conversation was not found or expired")
                self._sessions.pop(session_id)
            session.close()
        finally:
            session.lock.release()
        return {"deleted": True}

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._reap_expired_locked()
            active_sessions = len(self._sessions)
        return {
            "status": "ok",
            "active_sessions": active_sessions,
            "telemetry_enabled": self._telemetry is not None,
        }

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            with session.lock:
                session.close()
        if self._telemetry is not None:
            self._telemetry.shutdown()
            self._telemetry = None
        self._jev_engine.close()
