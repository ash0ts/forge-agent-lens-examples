from __future__ import annotations

import copy
import json
import os
import re
import select
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.propagators.textmap import CarrierT
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, ValidationError

# Pinned tau3 imports LiteLLM, whose default behavior can fetch a mutable remote
# price map at import time. The study's preview and MCP checks must stay offline.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from tau2.domains.retail.data_model import RetailDB

from .behavior import RuntimeBehaviorProfile, select_tool_behavior
from .constants import WRITE_TOOLS
from .retail_tools import StudyRetailTools

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_RESPONSE_TIMEOUT_SECONDS = 15.0
SERVER_NAME = "retail-support-study-v1"
SERVER_VERSION = "0.1.0"
_SERVER_INSTRUMENTATION_NAME = "retail-support-study.mcp-server"
_SERVER_INSTRUMENTATION_VERSION = "1"
_TRACE_CONTEXT_PROPAGATOR = TraceContextTextMapPropagator()

_MCP_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "CONDA_PREFIX",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "VIRTUAL_ENV",
    }
)
_SOURCE_SCHEMA_EXAMPLES = {
    "#W0000000": "#R1234567",
    "1008292230": "item_99999",
    "item id, such as '6086499569'": "item id, such as 'item_99999'",
    "product id, such as '6086499569'": "product id, such as 'prod_9999'",
    "sara_doe_496": "retail_user_9999",
    "gift_card_0000000": "gift_card_999999",
    "credit_card_0000000": "credit_card_999999",
}
_SOURCE_CONFIRMATION_SENTENCE = re.compile(
    r"\s*The agent needs to explain the "
    r"(?:cancellation|exchange|modification|return) detail\s+"
    r"and ask for explicit user confirmation \(yes/no\) to proceed\.",
    re.IGNORECASE,
)
_PREPARE_CONFIRM_SUFFIX = (
    "Runtime confirmation contract: the first call only prepares this exact "
    "action and does not change store state. Call it once before asking for "
    "confirmation. After the tool returns `action_prepared`, explain that one "
    "action and its consequences, ask for a new yes/no confirmation, and wait. "
    "After the customer confirms, repeat the exact same call once to execute it. "
    "Do not ask for confirmation before the preparation call."
)

_ORDER_SCOPED_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "get_order_details",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)
_USER_SCOPED_TOOLS = frozenset({"get_user_details", "modify_user_address"})
_IDENTITY_LOOKUP_TOOLS = frozenset(
    {"find_user_id_by_email", "find_user_id_by_name_zip"}
)


class AccountScopeError(Exception):
    """Raised when a tool tries to leave the authenticated account."""


class InvalidToolArguments(Exception):
    """Raised when an MCP tool call does not match its JSON schema."""


_JSONRPC_STATUS_DESCRIPTIONS = {
    "-32700": "Parse error",
    "-32600": "Invalid Request",
    "-32601": "Method not found",
    "-32602": "Invalid params",
    "-32603": "Internal error",
    "-32002": "Server not initialized",
}


class MCPProtocolError(RuntimeError):
    """Raised by the process client for a JSON-RPC or transport failure."""

    def __init__(
        self,
        message: str,
        *,
        response_status_code: str | None = None,
        response_status_description: str | None = None,
    ) -> None:
        super().__init__(message)
        self.response_status_code = response_status_code
        self.response_status_description = response_status_description or (
            _JSONRPC_STATUS_DESCRIPTIONS.get(response_status_code)
            if response_status_code is not None
            else None
        )


class MCPTimeoutError(MCPProtocolError):
    """Raised when the stdio MCP transport exceeds its response deadline."""


class MCPToolError(RuntimeError):
    """Raised by the process client when an MCP tool returns ``isError``."""

    def __init__(self, code: str, message: str | None = None) -> None:
        if message is None:
            message = code
            code = "tool_error"
        if not code:
            raise ValueError("MCP tool error requires a stable code")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.public_message = message


def _mcp_subprocess_environment(
    environ: Mapping[str, str] | None = None,
    *,
    package_source_dir: Path | None = None,
) -> dict[str, str]:
    """Build the minimum environment needed by the local MCP subprocess.

    The retail tool server never calls W&B or a model provider. In particular,
    it must not inherit the parent's API keys, OTLP headers, or unrelated
    credentials merely because the paid runner has them in its environment.
    """

    source = os.environ if environ is None else environ
    child = {
        name: value
        for name, value in source.items()
        if name in _MCP_CHILD_ENV_ALLOWLIST and value
    }
    child["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    if package_source_dir is not None:
        child["PYTHONPATH"] = str(package_source_dir.resolve())
    return child


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"tool returned unsupported value type: {type(value).__name__}")


def _rewrite_schema_examples(value: Any) -> Any:
    """Keep tool documentation aligned with the remapped runtime identifiers."""

    if isinstance(value, dict):
        return {key: _rewrite_schema_examples(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_schema_examples(child) for child in value]
    if isinstance(value, str):
        rewritten = value
        for source, target in _SOURCE_SCHEMA_EXAMPLES.items():
            rewritten = rewritten.replace(source, target)
        return rewritten
    return value


def _runtime_tool_description(name: str, description: str) -> str:
    """Align source tool docs with the study's prepare-confirm boundary."""

    if name not in WRITE_TOOLS:
        return description
    source = _SOURCE_CONFIRMATION_SENTENCE.sub("", description).strip()
    return f"{source}\n\n{_PREPARE_CONFIRM_SUFFIX}"


def _rpc_error(
    request_id: Any,
    code: int,
    message: str,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if detail:
        error["data"] = {"detail": detail}
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def parse_error_response() -> dict[str, Any]:
    """Return the JSON-RPC parse-error response used by the stdio loop."""

    return _rpc_error(None, -32700, "Parse error")


class RetailMCPProtocol:
    """Small MCP JSON-RPC dispatcher over one isolated τ³ retail database.

    The tracer is injectable because the paid study keeps the isolated stdio
    subprocess credential-free. A deployment that configures a server-side SDK
    and exporter receives standards-correct SERVER spans; the paid study itself
    exports only the outer CLIENT ``execute_tool`` span.
    """

    def __init__(
        self,
        db: RetailDB,
        *,
        authenticated_user_id: str | None = None,
        session_id: str | None = None,
        allowed_tool_names: Sequence[str] | None = None,
        behavior_profile: RuntimeBehaviorProfile | None = None,
        conversation_id: str | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if authenticated_user_id is not None and authenticated_user_id not in db.users:
            raise ValueError("authenticated user is not present in the retail database")
        self.db = db
        self.authenticated_user_id = authenticated_user_id
        self.session_id = session_id
        self._tracer = tracer or trace.get_tracer(
            _SERVER_INSTRUMENTATION_NAME,
            _SERVER_INSTRUMENTATION_VERSION,
        )
        self.tools = StudyRetailTools(db)
        definitions = self.tools.get_tools()
        if allowed_tool_names is None:
            self._tool_definitions = definitions
        else:
            allowed = frozenset(allowed_tool_names)
            if not allowed:
                raise ValueError("allowed tool manifest cannot be empty")
            unknown = allowed - set(definitions)
            if unknown:
                raise ValueError(f"allowed tool manifest has unknown names: {sorted(unknown)}")
            self._tool_definitions = {
                name: definition
                for name, definition in definitions.items()
                if name in allowed
            }
        if behavior_profile is not None and not conversation_id:
            raise ValueError("behavior profile requires a conversation ID")
        self._behavior_profile = behavior_profile
        self._conversation_id = conversation_id
        self._tool_call_ordinal = 0
        self._fault_injected = False
        self._initialized = False

    @classmethod
    def from_db_path(
        cls,
        db_path: str | Path,
        *,
        authenticated_user_id: str | None = None,
        session_id: str | None = None,
        allowed_tool_names: Sequence[str] | None = None,
        behavior_profile: RuntimeBehaviorProfile | None = None,
        conversation_id: str | None = None,
        tracer: Tracer | None = None,
    ) -> RetailMCPProtocol:
        # Loading here, rather than at module import, gives each server process its
        # own mutable database and prevents state from leaking between conversations.
        db = RetailDB.load(str(Path(db_path)))
        return cls(
            db,
            authenticated_user_id=authenticated_user_id,
            session_id=session_id,
            allowed_tool_names=allowed_tool_names,
            behavior_profile=behavior_profile,
            conversation_id=conversation_id,
            tracer=tracer,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in sorted(self._tool_definitions):
            openai_schema = copy.deepcopy(
                self._tool_definitions[name].openai_schema["function"]
            )
            openai_schema = _rewrite_schema_examples(openai_schema)
            input_schema = openai_schema["parameters"]
            input_schema["additionalProperties"] = False
            result.append(
                {
                    "name": name,
                    "description": _runtime_tool_description(
                        name, openai_schema.get("description", "")
                    ),
                    "inputSchema": input_schema,
                }
            )
        return result

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        if (
            isinstance(message, dict)
            and message.get("method") == "tools/call"
            and isinstance(message.get("params"), dict)
        ):
            return self._handle_traced_tool_call(message)
        return self._dispatch_message(message)

    def _handle_traced_tool_call(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        params = message["params"]
        metadata = params.get("_meta")
        carrier: CarrierT = metadata if isinstance(metadata, dict) else {}
        remote_context = _TRACE_CONTEXT_PROPAGATOR.extract(carrier=carrier)
        request_id = message.get("id")
        tool_name = params.get("name")
        target = tool_name if isinstance(tool_name, str) and tool_name else None
        attributes: dict[str, Any] = {
            "mcp.method.name": "tools/call",
            "mcp.protocol.version": MCP_PROTOCOL_VERSION,
            "network.transport": "pipe",
            "gen_ai.operation.name": "execute_tool",
        }
        if self.session_id:
            attributes["mcp.session.id"] = self.session_id
        if request_id is not None:
            attributes["jsonrpc.request.id"] = str(request_id)
        if target is not None:
            attributes["gen_ai.tool.name"] = target

        span_name = f"tools/call {target}" if target is not None else "tools/call"
        with self._tracer.start_as_current_span(
            span_name,
            context=remote_context,
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            try:
                response = self._dispatch_message(message)
            except BaseException as exc:
                error_type = type(exc).__name__
                span.set_attribute("error.type", error_type)
                span.set_status(Status(StatusCode.ERROR, error_type))
                raise
            self._record_server_response(span, response)
            return response

    @staticmethod
    def _record_server_response(
        span: Span,
        response: dict[str, Any] | None,
    ) -> None:
        if response is None:
            return
        error = response.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            error_type = str(code) if isinstance(code, (int, str)) else "_OTHER"
            span.set_attribute("error.type", error_type)
            if code is not None:
                span.set_attribute("rpc.response.status_code", str(code))
            description = error.get("message")
            span.set_status(
                Status(
                    StatusCode.ERROR,
                    str(description) if description else error_type,
                )
            )
            return
        result = response.get("result")
        if isinstance(result, dict) and result.get("isError") is True:
            span.set_attribute("error.type", "tool_error")
            span.set_status(Status(StatusCode.ERROR, "tool_error"))

    def _dispatch_message(self, message: Any) -> dict[str, Any] | None:
        request_id = message.get("id") if isinstance(message, dict) else None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _rpc_error(request_id, -32600, "Invalid Request")

        method = message.get("method")
        if not isinstance(method, str):
            return _rpc_error(request_id, -32600, "Invalid Request")

        is_notification = "id" not in message
        params = message.get("params", {})
        if not isinstance(params, dict):
            if is_notification:
                return None
            return _rpc_error(request_id, -32602, "Invalid params")

        if method == "initialize":
            if is_notification:
                return None
            self._initialized = True
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            }

        if method == "notifications/initialized":
            return None

        if is_notification:
            return None

        if not self._initialized:
            return _rpc_error(
                request_id,
                -32002,
                "Server not initialized",
            )

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            cursor = params.get("cursor")
            if cursor not in (None, ""):
                return _rpc_error(
                    request_id,
                    -32602,
                    "Invalid params",
                    detail="this server does not paginate tools",
                )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.list_tools()},
            }
        if method == "tools/call":
            return self._handle_tool_call(request_id, params)
        if method == "study/tools/validate":
            return self._handle_tool_validation(request_id, params)
        return _rpc_error(request_id, -32601, "Method not found")

    def _handle_tool_validation(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize and authorize arguments without invoking a retail tool.

        This private host-control method is not advertised in ``tools/list`` and
        is never available to the model. It lets the confirmation digest bind
        the same Pydantic-normalized business arguments later passed to
        ``tools/call``.
        """

        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name or name not in self._tool_definitions:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail="study/tools/validate requires a known tool name",
            )
        if not isinstance(arguments, dict):
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail="tool arguments must be an object",
            )
        try:
            validated = self._validate_arguments(name, arguments)
            self._authorize(name, validated)
        except InvalidToolArguments as exc:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail=str(exc),
            )
        except AccountScopeError:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail="tool access is limited to the authenticated account",
            )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"arguments": _json_value(validated)},
        }

    def _handle_tool_call(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail="tools/call requires a non-empty name",
            )
        if name not in self._tool_definitions:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail=f"unknown tool: {name}",
            )
        if not isinstance(arguments, dict):
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail="tool arguments must be an object",
            )

        try:
            validated = self._validate_arguments(name, arguments)
        except InvalidToolArguments as exc:
            return _rpc_error(
                request_id,
                -32602,
                "Invalid params",
                detail=str(exc),
            )

        try:
            self._authorize(name, validated)
            if self._behavior_profile is not None:
                self._tool_call_ordinal += 1
                selected = select_tool_behavior(
                    self._behavior_profile,
                    conversation_id=str(self._conversation_id),
                    tool_name=name,
                    call_ordinal=self._tool_call_ordinal,
                    fault_already_injected=self._fault_injected,
                )
                time.sleep(selected.delay_seconds)
                if selected.fault_code is not None:
                    self._fault_injected = True
                    if selected.fault_code == "service_unavailable":
                        message = "the retail service is temporarily unavailable"
                    elif selected.fault_code == "upstream_timeout":
                        message = "the retail service did not respond before its deadline"
                    else:  # pragma: no cover - profile validation owns this
                        raise RuntimeError("unsupported injected fault")
                    return self._tool_error(
                        request_id,
                        code=selected.fault_code,
                        message=message,
                    )
            raw_result = self.tools.use_tool(name, **validated)
            if name in _IDENTITY_LOOKUP_TOOLS and self.authenticated_user_id is None:
                self.authenticated_user_id = str(raw_result)
            result = _json_value(raw_result)
        except AccountScopeError:
            return self._tool_error(
                request_id,
                code="permission_denied",
                message="tool access is limited to the authenticated account",
            )
        except ValueError as exc:
            return self._tool_error(
                request_id,
                code="tool_error",
                message=str(exc),
            )
        except Exception:
            return self._tool_error(
                request_id,
                code="internal_error",
                message="tool execution failed",
            )

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            result,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
                "structuredContent": {"result": result},
                "isError": False,
            },
        }

    def _validate_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        model = self._tool_definitions[name].params
        field_names = set(model.model_fields)
        unexpected = sorted(set(arguments) - field_names)
        if unexpected:
            raise InvalidToolArguments(
                f"unexpected argument(s): {', '.join(unexpected)}"
            )
        try:
            parsed = model.model_validate(arguments)
        except ValidationError as exc:
            details = []
            for error in exc.errors(include_input=False, include_url=False):
                location = ".".join(str(part) for part in error["loc"])
                details.append(f"{location}: {error['msg']}")
            raise InvalidToolArguments("; ".join(details)) from exc
        return parsed.model_dump()

    def _authorize(self, name: str, arguments: dict[str, Any]) -> None:
        if name in _ORDER_SCOPED_TOOLS:
            if self.authenticated_user_id is None:
                raise AccountScopeError
            order = self.db.orders.get(arguments["order_id"])
            if order is None or order.user_id != self.authenticated_user_id:
                raise AccountScopeError
            return

        if name in _USER_SCOPED_TOOLS:
            if self.authenticated_user_id is None:
                raise AccountScopeError
            if arguments["user_id"] != self.authenticated_user_id:
                raise AccountScopeError
            return

        if name in _IDENTITY_LOOKUP_TOOLS:
            if self.authenticated_user_id is None:
                return
            user = self.db.users[self.authenticated_user_id]
            if name == "find_user_id_by_email":
                matches = arguments["email"].casefold() == user.email.casefold()
            else:
                matches = (
                    arguments["first_name"].casefold()
                    == user.name.first_name.casefold()
                    and arguments["last_name"].casefold()
                    == user.name.last_name.casefold()
                    and arguments["zip"] == user.address.zip
                )
            if not matches:
                raise AccountScopeError

    @staticmethod
    def _tool_error(request_id: Any, *, code: str, message: str) -> dict[str, Any]:
        payload = {"error": {"code": code, "message": message}}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
                "isError": True,
            },
        }


class MCPProcessClient:
    """Synchronous MCP stdio client used by one support conversation."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        policy_path: str | Path | None = None,
        authenticated_user_id: str | None = None,
        package_source_dir: str | Path | None = None,
        allowed_tool_names: Sequence[str] | None = None,
        behavior_profile: Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.policy_path = (
            Path(policy_path).resolve() if policy_path is not None else None
        )
        self.authenticated_user_id = authenticated_user_id
        self.package_source_dir = (
            Path(package_source_dir).resolve()
            if package_source_dir is not None
            else None
        )
        self.allowed_tool_names = (
            tuple(sorted(set(allowed_tool_names)))
            if allowed_tool_names is not None
            else None
        )
        self.behavior_profile = (
            dict(behavior_profile) if behavior_profile is not None else None
        )
        self.conversation_id = conversation_id
        if self.behavior_profile is not None:
            RuntimeBehaviorProfile.from_mapping(self.behavior_profile)
            if not conversation_id:
                raise ValueError("behavior profile requires a conversation ID")
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        if self._process is None or self._session_id is None:
            raise MCPProtocolError("MCP process is not running")
        return self._session_id

    def __enter__(self) -> MCPProcessClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None:
            raise MCPProtocolError("MCP process is already running")
        if not self.db_path.is_file():
            raise FileNotFoundError(f"retail database does not exist: {self.db_path}")
        if self.policy_path is not None and not self.policy_path.is_file():
            raise FileNotFoundError(f"retail policy does not exist: {self.policy_path}")
        if self.package_source_dir is not None:
            package = self.package_source_dir / "retail_support_study_v1"
            if package.is_symlink() or not package.is_dir():
                raise FileNotFoundError(
                    f"snapshot package source does not exist: {package}"
                )

        session_id = str(uuid.uuid4())
        command = [
            sys.executable,
            "-m",
            "retail_support_study_v1.mcp_server",
            "--db",
            str(self.db_path),
            "--session-id",
            session_id,
        ]
        if self.authenticated_user_id is not None:
            command.extend(["--authenticated-user-id", self.authenticated_user_id])
        if self.allowed_tool_names is not None:
            for tool_name in self.allowed_tool_names:
                command.extend(["--allowed-tool", tool_name])
        if self.behavior_profile is not None:
            command.extend(
                [
                    "--behavior-profile-json",
                    json.dumps(
                        self.behavior_profile,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "--conversation-id",
                    str(self.conversation_id),
                ]
            )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # The protocol uses stdout only. Leaving stderr as an unread pipe can
            # deadlock a long conversation if an imported dependency is noisy.
            # Discard it instead of admitting an unbounded, unreviewed side channel
            # into the frozen study artifacts.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=_mcp_subprocess_environment(
                package_source_dir=self.package_source_dir,
            ),
            cwd=(
                str(self.package_source_dir.parent)
                if self.package_source_dir is not None
                else None
            ),
        )
        self._session_id = session_id
        try:
            response = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            if response.get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise MCPProtocolError("MCP protocol-version negotiation failed")
            self._notify("notifications/initialized", {})
        except BaseException:
            # __enter__ has not returned yet, so the context manager cannot run
            # __exit__. Reap the process here on every failed startup path.
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        self._process = None
        self._session_id = None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    # The child may already have closed the pipe. It still needs
                    # to be waited on so it cannot remain as a zombie.
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPProtocolError("tools/list returned a malformed result")
        return tools

    def list_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["inputSchema"],
                },
            }
            for tool in self.list_tools()
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_request_id: Callable[[int], None] | None = None,
    ) -> Any:
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            on_request_id=on_request_id,
        )
        if result.get("isError"):
            payload = self._content_payload(result)
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                code = str(error.get("code", "tool_error"))
                message = str(error.get("message", "tool call failed"))
                raise MCPToolError(code, message)
            raise MCPToolError("tool_error", "tool call failed")
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return self._content_payload(result)

    def validate_tool_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Return server-normalized arguments without executing the tool."""

        result = self._request(
            "study/tools/validate",
            {"name": name, "arguments": arguments},
        )
        validated = result.get("arguments")
        if not isinstance(validated, dict):
            raise MCPProtocolError("tool validation returned malformed arguments")
        return validated

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        on_request_id: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        if on_request_id is not None:
            on_request_id(request_id)
        wire_params = dict(params)
        metadata = wire_params.get("_meta")
        if metadata is None:
            trace_metadata: dict[str, str] = {}
        elif isinstance(metadata, dict):
            trace_metadata = {
                str(key): str(value) for key, value in metadata.items()
            }
        else:
            raise MCPProtocolError("MCP params._meta must be an object")
        _TRACE_CONTEXT_PROPAGATOR.inject(trace_metadata)
        if trace_metadata:
            wire_params["_meta"] = trace_metadata
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": wire_params,
            }
        )
        response = self._read()
        if response.get("id") != request_id:
            raise MCPProtocolError("MCP response id does not match the request")
        error = response.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "JSON-RPC request failed"))
            code = error.get("code")
            response_status_code = (
                str(code) if isinstance(code, (int, str)) else None
            )
            response_status_description = _JSONRPC_STATUS_DESCRIPTIONS.get(
                response_status_code,
                "JSON-RPC error",
            )
            data = error.get("data")
            if isinstance(data, dict) and data.get("detail"):
                message = f"{message}: {data['detail']}"
            raise MCPProtocolError(
                message,
                response_status_code=response_status_code,
                response_status_description=response_status_description,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP response has no object result")
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise MCPProtocolError("MCP process has no stdin")
        try:
            process.stdin.write(
                json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPProtocolError("MCP process closed its input") from exc

    def _read(self) -> dict[str, Any]:
        process = self._require_process()
        if process.stdout is None:
            raise MCPProtocolError("MCP process has no stdout")
        ready, _, _ = select.select(
            [process.stdout], [], [], MCP_RESPONSE_TIMEOUT_SECONDS
        )
        if not ready:
            raise MCPTimeoutError(
                f"MCP process did not respond within "
                f"{MCP_RESPONSE_TIMEOUT_SECONDS:g} seconds"
            )
        line = process.stdout.readline()
        if not line:
            raise MCPProtocolError("MCP process exited unexpectedly")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPProtocolError("MCP process returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise MCPProtocolError("MCP process returned a non-object response")
        return response

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise MCPProtocolError("MCP process is not running")
        return self._process

    @staticmethod
    def _content_payload(result: dict[str, Any]) -> Any:
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise MCPProtocolError("tool result has no content")
        first = content[0]
        if not isinstance(first, dict) or not isinstance(first.get("text"), str):
            raise MCPProtocolError("tool result has malformed text content")
        try:
            return json.loads(first["text"])
        except json.JSONDecodeError:
            return first["text"]
