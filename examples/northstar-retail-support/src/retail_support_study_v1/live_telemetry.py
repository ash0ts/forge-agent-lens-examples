from __future__ import annotations

import base64
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from .constants import (
    SUPPORT_MAX_OUTPUT_TOKENS,
    SUPPORT_MODEL,
    SUPPORT_TEMPERATURE,
)
from .mcp_protocol import MCPProtocolError, MCPTimeoutError, MCPToolError
from .schemas import ModelCall, ToolCallRecord
from .telemetry import (
    MAX_PUBLIC_ATTRIBUTE_CHARACTERS,
    _json_string,
    _public_message,
    _validate_public_value,
)

_DEFAULT_AGENT_NAME = "retail-support-agent"
_INSTRUMENTATION_NAME = "retail-support-study.live"
_INSTRUMENTATION_VERSION = "1"
_OTLP_ENDPOINT = "https://trace.wandb.ai/agents/otel/v1/traces"
_DEFAULT_PROVIDER_NAME = "wandb"
_DEFAULT_INFERENCE_SERVER_ADDRESS = "api.inference.wandb.ai"
_DEFAULT_INFERENCE_SERVER_PORT = 443
_TYPESAFE_SERVER_ADDRESS = "api.typesafe.ai"
_TYPESAFE_SERVER_PORT = 443
_MCP_PROTOCOL_VERSION = "2025-06-18"
_FORCE_FLUSH_TIMEOUT_MILLIS = 120_000
_BATCH_QUEUE_SIZE = 8_192
_BATCH_EXPORT_SIZE = 512
_BATCH_SCHEDULE_DELAY_MILLIS = 1_000
_EXPORT_TIMEOUT_SECONDS = 30
_MACHINE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")

_EXPORT_OVERRIDE_ENVIRONMENT = frozenset(
    {
        "OTEL_EXPORTER_OTLP_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_CLIENT_KEY",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TIMEOUT",
        "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
    }
)


def _error_type(exc: BaseException) -> str:
    return type(exc).__name__


def _error(span: Span, exc: BaseException) -> None:
    error_type = _error_type(exc)
    span.set_attribute("error.type", error_type)
    # Do not record the exception. Provider and tool errors may contain raw
    # response bodies or other text that is outside the public telemetry plan.
    span.set_status(Status(StatusCode.ERROR, error_type))


def _mcp_error(span: Span, exc: BaseException) -> None:
    """Record MCP errors using the protocol's low-cardinality conventions."""

    response_status_code = (
        exc.response_status_code if isinstance(exc, MCPProtocolError) else None
    )
    if isinstance(exc, MCPToolError):
        error_type = "tool_error"
        status_description = error_type
    elif isinstance(exc, MCPTimeoutError):
        error_type = "timeout"
        status_description = error_type
    elif response_status_code is not None:
        error_type = response_status_code
        span.set_attribute("rpc.response.status_code", response_status_code)
        status_description = exc.response_status_description or error_type
    else:
        error_type = _error_type(exc)
        status_description = error_type
    span.set_attribute("error.type", error_type)
    # Provider/tool error messages may contain sensitive response content. The
    # stable error identifier is sufficient for the public trace.
    span.set_status(Status(StatusCode.ERROR, status_description))


def _public_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
) -> list[dict[str, Any]]:
    projected = [
        _public_message(message, tools_by_id=tools_by_id)
        for message in messages
        if message.get("role") != "system"
    ]
    # Reuse the reviewed public-payload path shape so machine-issued tool-call
    # IDs are validated as identifiers while natural-language content still
    # receives the full PII scan.
    _validate_public_value({"turns": [{"input_messages": projected}]})
    return projected


def _public_output(
    message: Mapping[str, Any],
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
) -> list[dict[str, Any]]:
    projected = [_public_message(message, tools_by_id=tools_by_id)]
    _validate_public_value({"turns": [{"output_messages": projected}]})
    return projected


def _public_value(value: Any) -> Any:
    _validate_public_value(value)
    return value


def _validate_span_attributes(
    attributes: Mapping[str, Any],
    *,
    structured_json: Mapping[str, Any] | None = None,
    machine_identifiers: Sequence[str] = (),
) -> None:
    """Validate the values represented by OTLP attributes exactly once.

    GenAI message and tool attributes are JSON strings because Python OTLP
    attributes cannot carry nested structures. Their structured values have
    already been projected through the public-data allowlist, so validate that
    structure rather than scanning its serialized form a second time. The
    second scan used to mistake numeric UUID fragments for payment cards.

    Provider and model tool-call IDs are opaque machine identifiers, not user
    content. Restrict their alphabet and length instead of applying PII pattern
    matching intended for natural-language fields.
    """

    structured = dict(structured_json or {})
    identifiers = set(machine_identifiers)
    overlap = set(structured) & identifiers
    if overlap:
        raise ValueError(f"span attribute validation overlaps: {sorted(overlap)}")
    if missing := (set(structured) | identifiers) - set(attributes):
        raise ValueError(f"span attributes missing reviewed values: {sorted(missing)}")

    for key, value in structured.items():
        if attributes[key] != _json_string(value):
            raise ValueError(f"serialized span attribute drift: {key}")

    for key in identifiers:
        value = attributes[key]
        if not isinstance(value, str) or not _MACHINE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"unsafe machine identifier: {key}")

    ordinary = {
        key: value
        for key, value in attributes.items()
        if key not in structured and key not in identifiers
    }
    _validate_public_value(ordinary)


def _record_from_result(
    result: Any,
    records: Sequence[ModelCall],
    *,
    previous_count: int,
) -> ModelCall:
    record = getattr(result, "record", None)
    if isinstance(record, ModelCall):
        return record
    if len(records) == previous_count + 1 and isinstance(records[-1], ModelCall):
        return records[-1]
    raise RuntimeError("support inference did not produce one model-call record")


def _record_from_exception(
    exc: BaseException,
    records: Sequence[ModelCall],
    *,
    previous_count: int,
) -> ModelCall | None:
    record = getattr(exc, "record", None)
    if isinstance(record, ModelCall):
        return record
    if len(records) == previous_count + 1 and isinstance(records[-1], ModelCall):
        return records[-1]
    return None


def _chat_request_attributes(
    conversation_id: str,
    request: Mapping[str, Any],
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
    provider_name: str,
    server_address: str,
    server_port: int,
) -> dict[str, Any]:
    model = str(request.get("model") or "")
    if not model:
        raise ValueError("support inference request requires a model")
    messages = request.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("support inference request requires messages")
    public_messages = _public_messages(messages, tools_by_id=tools_by_id)
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider_name,
        "gen_ai.conversation.id": conversation_id,
        "gen_ai.request.model": model,
        "server.address": server_address,
        "server.port": server_port,
        "gen_ai.input.messages": _json_string(public_messages),
    }
    temperature = request.get("temperature")
    if temperature is not None:
        attributes["gen_ai.request.temperature"] = float(temperature)
    max_tokens = request.get("max_tokens")
    if max_tokens is not None:
        attributes["gen_ai.request.max_tokens"] = int(max_tokens)
    seed = request.get("seed")
    if seed is not None:
        attributes["gen_ai.request.seed"] = int(seed)
    _validate_span_attributes(
        attributes,
        structured_json={"gen_ai.input.messages": public_messages},
        machine_identifiers=("gen_ai.conversation.id",),
    )
    return attributes


def _set_chat_response_attributes(
    span: Span,
    record: ModelCall,
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
) -> None:
    if record.participant != "support_agent":
        raise ValueError("live Agent Lens telemetry cannot emit simulator calls")
    public_output = _public_output(record.output_message, tools_by_id=tools_by_id)
    attributes: dict[str, Any] = {
        "gen_ai.response.model": record.returned_model,
        "gen_ai.output.messages": _json_string(public_output),
        "gen_ai.usage.input_tokens": record.usage.input_tokens,
        "gen_ai.usage.output_tokens": record.usage.output_tokens,
    }
    if record.response_id:
        attributes["gen_ai.response.id"] = record.response_id
    if record.finish_reason:
        attributes["gen_ai.response.finish_reasons"] = (record.finish_reason,)
    _validate_span_attributes(
        attributes,
        structured_json={"gen_ai.output.messages": public_output},
        machine_identifiers=(
            ("gen_ai.response.id",) if "gen_ai.response.id" in attributes else ()
        ),
    )
    for key, value in attributes.items():
        span.set_attribute(key, value)


def _jev_request_attributes(
    conversation_id: str,
    *,
    operation: str,
    backend: str,
    requested_model: str,
    question_names: Sequence[str],
) -> dict[str, Any]:
    from .jev import (
        ASSESSMENT_QUESTION_SET_ID,
        ASSESSMENT_QUESTION_SET_SHA256,
        ROUTE_QUESTION_SET_ID,
        ROUTE_QUESTION_SET_SHA256,
    )

    question_set = {
        "route": (ROUTE_QUESTION_SET_ID, ROUTE_QUESTION_SET_SHA256),
        "assessment": (
            ASSESSMENT_QUESTION_SET_ID,
            ASSESSMENT_QUESTION_SET_SHA256,
        ),
    }.get(operation)
    if question_set is None:
        raise ValueError(f"unsupported Jev operation: {operation}")
    if not requested_model:
        raise ValueError("Jev requested_model is required")
    names = tuple(sorted(str(name) for name in question_names))
    if not names:
        raise ValueError("Jev questions are required")
    if backend not in {"live", "mock"}:
        raise ValueError(f"unsupported Jev backend: {backend}")
    prefix = "typesafe" if backend == "live" else "demo.mock_jev"
    attributes: dict[str, Any] = {
        # `evaluate` is deliberately distinct from `chat`: Jev produces typed
        # decisions, not customer-facing language. Agent Lens can still retain
        # it in the trace without inflating the support-model call count.
        "gen_ai.operation.name": "evaluate",
        "gen_ai.provider.name": (
            "typesafe" if backend == "live" else "retail_demo.mock"
        ),
        "gen_ai.conversation.id": conversation_id,
        "gen_ai.request.model": requested_model,
        "demo.decision.backend": backend,
        f"{prefix}.operation.name": (
            "system_one" if backend == "live" else "system_one_contract"
        ),
        f"{prefix}.decision.purpose": operation,
        f"{prefix}.question_set.id": question_set[0],
        f"{prefix}.question_set.sha256": question_set[1],
        f"{prefix}.question.names": names,
    }
    if backend == "live":
        attributes["server.address"] = _TYPESAFE_SERVER_ADDRESS
        attributes["server.port"] = _TYPESAFE_SERVER_PORT
    _validate_span_attributes(
        attributes,
        machine_identifiers=("gen_ai.conversation.id",),
    )
    return attributes


def _set_jev_response_attributes(
    span: Span,
    response: Any,
    *,
    backend: str,
) -> None:
    model = str(getattr(response, "model", ""))
    if not model:
        raise ValueError("Jev response model is required")
    answers = getattr(response, "answers", None)
    if not isinstance(answers, Mapping):
        raise ValueError("Jev response answers are required")
    public_answers: dict[str, Any] = {}
    for name, answer in sorted(answers.items()):
        answer_type = str(getattr(answer, "type", ""))
        if answer_type == "choice":
            public_answers[str(name)] = {
                "type": answer_type,
                "choice": str(answer.choice),
                "confidence": float(answer.confidence),
                "probabilities": {
                    str(key): float(value)
                    for key, value in answer.probabilities.items()
                },
            }
        elif answer_type == "noul":
            public_answers[str(name)] = {
                "type": answer_type,
                "noul": float(answer.noul),
            }
        elif answer_type == "score":
            public_answers[str(name)] = {
                "type": answer_type,
                "score": float(answer.score),
                "confidence": float(answer.confidence),
                "probabilities": {
                    str(key): float(value)
                    for key, value in answer.probabilities.items()
                },
            }
        else:
            raise ValueError(f"unsupported Jev answer type: {answer_type}")
    usage = getattr(response, "usage", None)
    answers_attribute = (
        "typesafe.decision.answers"
        if backend == "live"
        else "demo.mock_jev.decision.answers"
    )
    attributes: dict[str, Any] = {
        "gen_ai.response.model": model,
        answers_attribute: _json_string(public_answers),
    }
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = int(output_tokens)
    _validate_span_attributes(
        attributes,
        structured_json={answers_attribute: public_answers},
    )
    for key, value in attributes.items():
        span.set_attribute(key, value)


class _CheckedExporter(SpanExporter):
    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter
        self._lock = threading.Lock()
        self._failure: BaseException | str | None = None

    @property
    def failure(self) -> BaseException | str | None:
        with self._lock:
            return self._failure

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        try:
            result = self._exporter.export(spans)
        except BaseException as exc:
            with self._lock:
                self._failure = exc
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            with self._lock:
                self._failure = f"export returned {result!r}"
        return result

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            result = self._exporter.force_flush(timeout_millis)
        except BaseException as exc:
            with self._lock:
                self._failure = exc
            return False
        if not result:
            with self._lock:
                self._failure = "exporter force_flush timed out"
        return result

    def shutdown(self) -> None:
        try:
            self._exporter.shutdown()
        except BaseException as exc:
            with self._lock:
                self._failure = exc


class LiveTelemetrySession:
    """One thread-safe, execution-time Agent Lens telemetry session.

    A session owns no global OpenTelemetry state. One instance is shared by all
    coordinate workers in a cohort and flushed by the coordinator between waves.
    """

    def __init__(
        self,
        *,
        provider: TracerProvider,
        agent_version: str,
        checked_exporter: _CheckedExporter | None = None,
        owns_provider: bool = False,
    ) -> None:
        if not agent_version:
            raise ValueError("agent_version is required")
        self._provider = provider
        self._tracer: Tracer = provider.get_tracer(
            _INSTRUMENTATION_NAME,
            _INSTRUMENTATION_VERSION,
        )
        self.agent_version = agent_version
        self._checked_exporter = checked_exporter
        self._owns_provider = owns_provider
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def from_provider(
        cls,
        provider: TracerProvider,
        *,
        agent_version: str,
        owns_provider: bool = False,
    ) -> Self:
        return cls(
            provider=provider,
            agent_version=agent_version,
            owns_provider=owns_provider,
        )

    @classmethod
    def from_exporter(
        cls,
        exporter: SpanExporter,
        *,
        agent_version: str,
        resource_attributes: Mapping[str, Any] | None = None,
    ) -> Self:
        attributes = {
            "service.name": "retail-support-application",
            "service.namespace": "retail-support-study",
            "service.version": agent_version,
            "deployment.environment.name": "test",
            **dict(resource_attributes or {}),
        }
        _validate_public_value(attributes)
        provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource(attributes=attributes),
            shutdown_on_exit=False,
            span_limits=SpanLimits(
                max_attributes=128,
                max_events=128,
                max_links=128,
                max_span_attributes=128,
                max_event_attributes=128,
                max_link_attributes=128,
                max_attribute_length=MAX_PUBLIC_ATTRIBUTE_CHARACTERS,
                max_span_attribute_length=MAX_PUBLIC_ATTRIBUTE_CHARACTERS,
            ),
        )
        checked = _CheckedExporter(exporter)
        provider.add_span_processor(
            BatchSpanProcessor(
                checked,
                max_queue_size=_BATCH_QUEUE_SIZE,
                schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MILLIS,
                max_export_batch_size=_BATCH_EXPORT_SIZE,
                export_timeout_millis=_FORCE_FLUSH_TIMEOUT_MILLIS,
            )
        )
        return cls(
            provider=provider,
            agent_version=agent_version,
            checked_exporter=checked,
            owns_provider=True,
        )

    @classmethod
    def open(cls, *, cohort: str, agent_version: str) -> Self:
        from . import constants

        projects = getattr(constants, "LIVE_AGENT_LENS_PROJECTS", None)
        if not isinstance(projects, Mapping) or cohort not in projects:
            raise ValueError(f"no reviewed live Agent Lens project for cohort {cohort!r}")
        project = str(projects[cohort])
        if not project.startswith("wandb/") or project.count("/") != 1:
            raise ValueError("live Agent Lens destination is malformed")
        configured = sorted(
            name
            for name in _EXPORT_OVERRIDE_ENVIRONMENT
            if os.environ.get(name, "").strip()
        )
        if configured:
            raise ValueError(
                "live Agent Lens telemetry rejects OTLP transport overrides; unset: "
                + ", ".join(configured)
            )
        api_key = os.environ.get("WANDB_API_KEY", "")
        if (
            not api_key
            or api_key != api_key.strip()
            or "\n" in api_key
            or "\r" in api_key
        ):
            raise ValueError("WANDB_API_KEY must be a non-empty single-line value")
        encoded = base64.b64encode(f"api:{api_key}".encode()).decode("ascii")
        exporter = OTLPSpanExporter(
            endpoint=_OTLP_ENDPOINT,
            headers={
                "Authorization": f"Basic {encoded}",
                "project_id": project,
            },
            timeout=_EXPORT_TIMEOUT_SECONDS,
            compression=Compression.Gzip,
        )
        return cls.from_exporter(exporter, agent_version=agent_version)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: Any,
    ) -> None:
        try:
            self.shutdown()
        except BaseException as shutdown_error:
            if _exc is None:
                raise
            _exc.add_note(
                "live telemetry shutdown also failed: "
                f"{type(shutdown_error).__name__}"
            )

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("live telemetry session is closed")

    def start_turn(
        self,
        conversation_id: str,
        input_messages: Sequence[Mapping[str, Any]],
        tools_by_id: Mapping[str, ToolCallRecord],
        *,
        agent_name: str = _DEFAULT_AGENT_NAME,
        requested_model: str = SUPPORT_MODEL,
        requested_temperature: float | None = SUPPORT_TEMPERATURE,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        server_address: str = _DEFAULT_INFERENCE_SERVER_ADDRESS,
        server_port: int = _DEFAULT_INFERENCE_SERVER_PORT,
    ) -> LiveAgentTurn:
        self._require_open()
        if not conversation_id:
            raise ValueError("conversation_id is required")
        for field, value in (
            ("agent_name", agent_name),
            ("requested_model", requested_model),
            ("provider_name", provider_name),
            ("server_address", server_address),
        ):
            if not value:
                raise ValueError(f"{field} is required")
        if server_port <= 0:
            raise ValueError("server_port must be positive")
        public_input = _public_messages(input_messages, tools_by_id=tools_by_id)
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": conversation_id,
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.version": self.agent_version,
            "gen_ai.request.model": requested_model,
            "gen_ai.request.max_tokens": SUPPORT_MAX_OUTPUT_TOKENS,
            "gen_ai.input.messages": _json_string(public_input),
        }
        if requested_temperature is not None:
            attributes["gen_ai.request.temperature"] = requested_temperature
        _validate_span_attributes(
            attributes,
            structured_json={"gen_ai.input.messages": public_input},
            machine_identifiers=("gen_ai.conversation.id",),
        )
        # An empty context deliberately makes every actual user -> agent turn a
        # parentless trace. The stable conversation ID joins those turns in Lens.
        root = self._tracer.start_span(
            f"invoke_agent {agent_name}",
            context=context.Context(),
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        )
        return LiveAgentTurn(
            tracer=self._tracer,
            root=root,
            conversation_id=conversation_id,
            tools_by_id=tools_by_id,
            provider_name=provider_name,
            server_address=server_address,
            server_port=server_port,
        )

    def jev_call_runner(self, conversation_id: str) -> LiveJevCallRunner:
        """Create a live wrapper for TypeSafe System One calls in a conversation.

        The chat runtime binds the runner to the active customer turn before
        routing. Both the routing decision and the post-turn behavior check
        therefore sit beside the model and MCP calls they influenced. A caller
        may still use the runner without a turn for a standalone decision.
        """

        self._require_open()
        if not conversation_id:
            raise ValueError("conversation_id is required")
        return LiveJevCallRunner(
            tracer=self._tracer,
            conversation_id=conversation_id,
        )

    def force_flush(self) -> None:
        self._require_open()
        if not self._provider.force_flush(_FORCE_FLUSH_TIMEOUT_MILLIS):
            raise RuntimeError("live telemetry provider did not flush before deadline")
        if self._checked_exporter is not None and self._checked_exporter.failure:
            raise RuntimeError("live telemetry export failed")

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
        failure: BaseException | None = None
        try:
            self.force_flush()
        except BaseException as exc:
            failure = exc
        try:
            if self._owns_provider:
                self._provider.shutdown()
        except BaseException as exc:
            if failure is None:
                failure = exc
        if (
            failure is None
            and self._checked_exporter is not None
            and self._checked_exporter.failure
        ):
            failure = RuntimeError("live telemetry export failed during shutdown")
        with self._lock:
            self._closed = True
        if failure is not None:
            raise failure


class LiveAgentTurn:
    """A single real user -> support-agent turn and its child operations."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        root: Span,
        conversation_id: str,
        tools_by_id: Mapping[str, ToolCallRecord],
        provider_name: str,
        server_address: str,
        server_port: int,
    ) -> None:
        self._tracer = tracer
        self._root = root
        self._conversation_id = conversation_id
        self._tools_by_id = dict(tools_by_id)
        self._provider_name = provider_name
        self._server_address = server_address
        self._server_port = server_port
        self._child_context = trace.set_span_in_context(root, context.Context())
        self._ended = False
        self._lock = threading.Lock()

    def _require_active(self) -> None:
        with self._lock:
            if self._ended:
                raise RuntimeError("live agent turn has already ended")

    def resolve_agent(self, agent_name: str) -> None:
        """Replace the provisional dispatcher identity with the applied agent.

        Routing is part of the customer turn, so the root must exist before the
        routing call. OpenTelemetry spans are mutable until they end; updating
        the name and reviewed agent attribute preserves chronological
        parentage while Agent Lens ultimately shows the agent that handled the
        request.
        """

        self._require_active()
        if not agent_name:
            raise ValueError("agent_name is required")
        _validate_public_value(agent_name)
        self._root.update_name(f"invoke_agent {agent_name}")
        self._root.set_attribute("gen_ai.agent.name", agent_name)

    def complete_support_call(
        self,
        operation: Callable[[], Any],
        records: Sequence[ModelCall],
        request: Mapping[str, Any],
    ) -> Any:
        self._require_active()
        request_attributes = _chat_request_attributes(
            self._conversation_id,
            request,
            tools_by_id=self._tools_by_id,
            provider_name=self._provider_name,
            server_address=self._server_address,
            server_port=self._server_port,
        )
        requested_model = str(request["model"])
        span = self._tracer.start_span(
            f"chat {requested_model}",
            context=self._child_context,
            kind=SpanKind.CLIENT,
            attributes=request_attributes,
        )
        previous_count = len(records)
        try:
            result = operation()
            record = _record_from_result(
                result,
                records,
                previous_count=previous_count,
            )
            _set_chat_response_attributes(
                span,
                record,
                tools_by_id=self._tools_by_id,
            )
            return result
        except BaseException as exc:
            record = _record_from_exception(
                exc,
                records,
                previous_count=previous_count,
            )
            if record is not None:
                try:
                    _set_chat_response_attributes(
                        span,
                        record,
                        tools_by_id=self._tools_by_id,
                    )
                except (TypeError, ValueError):
                    # Keep the real operation failure primary. Invalid response
                    # content is not attached to telemetry.
                    pass
            _error(span, exc)
            raise
        finally:
            span.end()

    def execute_mcp(
        self,
        operation: Callable[[Callable[[int], None]], Any],
        *,
        name: str,
        arguments: Mapping[str, Any],
        call_id: str,
        session_id: str,
    ) -> Any:
        self._require_active()
        if not name or not call_id or not session_id:
            raise ValueError(
                "executed MCP call requires a tool name, call id, and session id"
            )
        public_arguments = dict(arguments)
        _public_value(public_arguments)
        attributes: dict[str, Any] = {
            "mcp.method.name": "tools/call",
            "mcp.protocol.version": _MCP_PROTOCOL_VERSION,
            "mcp.session.id": session_id,
            "network.transport": "pipe",
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.conversation.id": self._conversation_id,
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.call.arguments": _json_string(public_arguments),
        }
        _validate_span_attributes(
            attributes,
            structured_json={"gen_ai.tool.call.arguments": public_arguments},
            machine_identifiers=(
                "gen_ai.conversation.id",
                "gen_ai.tool.call.id",
                "mcp.session.id",
            ),
        )
        span = self._tracer.start_span(
            f"tools/call {name}",
            context=self._child_context,
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )
        # Make the CLIENT span current only while the stdio request is in
        # flight. MCPProcessClient injects this W3C trace context into
        # params._meta so a configured server tracer can use it as its remote
        # parent without receiving the Agent Lens export credential.
        span_context = trace.set_span_in_context(span, self._child_context)
        attach_token = context.attach(span_context)
        started = time.perf_counter()
        request_id_recorded = False

        def record_request_id(request_id: int) -> None:
            nonlocal request_id_recorded
            if request_id_recorded:
                raise RuntimeError("MCP request ID was recorded more than once")
            if not isinstance(request_id, int) or request_id < 0:
                raise ValueError("MCP request ID must be a non-negative integer")
            span.set_attribute("jsonrpc.request.id", str(request_id))
            request_id_recorded = True

        try:
            result = operation(record_request_id)
            if not request_id_recorded:
                raise RuntimeError("executed MCP call did not expose its request ID")
            _public_value(result)
            span.set_attribute("gen_ai.tool.call.result", _json_string(result))
            self._tools_by_id[call_id] = ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=public_arguments,
                result=result,
                execution_disposition="executed",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return result
        except BaseException as exc:
            self._tools_by_id[call_id] = ToolCallRecord(
                call_id=call_id,
                name=name,
                arguments=public_arguments,
                error_type=_error_type(exc),
                execution_disposition="executed",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            _mcp_error(span, exc)
            raise
        finally:
            context.detach(attach_token)
            span.end()

    def finish(
        self,
        output_message: Mapping[str, Any],
        tools_by_id: Mapping[str, ToolCallRecord],
    ) -> None:
        self._require_active()
        self._tools_by_id.update(tools_by_id)
        try:
            output = _public_output(
                output_message,
                tools_by_id=self._tools_by_id,
            )
            self._root.set_attribute("gen_ai.output.messages", _json_string(output))
        except BaseException as exc:
            _error(self._root, exc)
            self._end()
            raise
        self._end()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._ended:
                return
        _error(self._root, exc)
        self._end()

    def _end(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        self._root.end()


class LiveJevCallRunner:
    """Wrap live or explicitly mocked decisions in content-minimized spans."""

    def __init__(self, *, tracer: Tracer, conversation_id: str) -> None:
        self._tracer = tracer
        self._conversation_id = conversation_id
        self._parent_context: context.Context | None = None
        self._lock = threading.Lock()

    def bind_turn(self, turn: LiveAgentTurn) -> None:
        turn._require_active()
        with self._lock:
            self._parent_context = turn._child_context

    def clear_turn(self) -> None:
        with self._lock:
            self._parent_context = None

    def __call__(
        self,
        *,
        operation: str,
        state: Any,
        questions: Mapping[str, Any],
        backend: Any,
        requested_model: str,
        timeout_seconds: float,
        invoke: Callable[[], Any],
    ) -> Any:
        # `state` is intentionally not attached to telemetry. It contains the
        # customer's visible message and recent turn context, which Agent Lens
        # already retains on the reviewed invoke_agent trace.
        del state
        if timeout_seconds <= 0:
            raise ValueError("Jev timeout_seconds must be positive")
        request_attributes = _jev_request_attributes(
            self._conversation_id,
            operation=operation,
            backend=str(backend),
            requested_model=requested_model,
            question_names=tuple(questions),
        )
        prefix = "typesafe" if str(backend) == "live" else "demo.mock_jev"
        request_attributes[f"{prefix}.request.timeout_seconds"] = float(
            timeout_seconds
        )
        _validate_span_attributes(
            request_attributes,
            machine_identifiers=("gen_ai.conversation.id",),
        )
        with self._lock:
            parent_context = self._parent_context
        purpose = "routing decision" if operation == "route" else "behavior check"
        span = self._tracer.start_span(
            f"evaluate {purpose} with {requested_model}",
            context=parent_context or context.Context(),
            kind=(SpanKind.CLIENT if str(backend) == "live" else SpanKind.INTERNAL),
            attributes=request_attributes,
        )
        try:
            result = invoke()
            _set_jev_response_attributes(span, result, backend=str(backend))
            return result
        except BaseException as exc:
            _error(span, exc)
            raise
        finally:
            span.end()
