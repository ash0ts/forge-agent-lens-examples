from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import context, trace
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from .canonical import canonical_bytes, digest_value
from .constants import (
    DESTINATION_PROJECT,
    FORBIDDEN_PUBLIC_KEY_FRAGMENTS,
    SCHEMA_VERSION,
    STUDY_ID,
)
from .schemas import ConversationBundle, ModelCall, RunLedger, ToolCallRecord

_PROVIDER_NAME = "wandb"
_NETWORK_TRANSPORT = "pipe"
_MCP_PROTOCOL_VERSION = "2025-06-18"
MAX_PUBLIC_ATTRIBUTE_CHARACTERS = 1_000_000

ROOT_ATTRIBUTE_KEYS = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.conversation.id",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
    }
)
CHAT_ATTRIBUTE_KEYS = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.conversation.id",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.request.temperature",
        "gen_ai.request.seed",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)
MCP_ATTRIBUTE_KEYS = frozenset(
    {
        "mcp.method.name",
        "mcp.protocol.version",
        "network.transport",
        "gen_ai.operation.name",
        "gen_ai.conversation.id",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "jsonrpc.request.id",
        "error.type",
    }
)

_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "access_token",
    "auth_token",
    "refresh_token",
    "service_token",
    "secret",
)
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?:file://|/Users/|/home/|/private/(?:tmp|var)/|/tmp/|/workspace/|"
        r"[A-Za-z]:\\\\)"
    ),
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9.-]{0,61}[A-Z0-9])?\.[A-Z]{2,63}\b",
    re.IGNORECASE,
)
_ALLOWED_EMAIL_DOMAINS = frozenset(
    {
        "example.com",
        "example.invalid",
        "example.net",
        "example.org",
    }
)
_PAYMENT_CARD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[ -]?){12,18}\d(?![A-Za-z0-9])"
)
_CANONICAL_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\+?1[ .-]?)?"
    r"(?:\(\d{3}\)[ .-]?|\d{3}[ .-]?)"
    r"\d{3}[ .-]?\d{4}(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class EmissionCounts:
    conversations: int
    turns: int
    chat_spans: int
    tool_spans: int

    @property
    def total_spans(self) -> int:
        return self.turns + self.chat_spans + self.tool_spans


def _json_string(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def _normalize_bundles(
    bundles: Mapping[str, ConversationBundle] | Sequence[ConversationBundle],
) -> dict[str, ConversationBundle]:
    if isinstance(bundles, Mapping):
        result = dict(bundles)
    else:
        result = {}
        for bundle in bundles:
            coordinate_id = bundle.coordinate.coordinate_id
            if coordinate_id in result:
                raise ValueError(f"duplicate bundle coordinate: {coordinate_id}")
            result[coordinate_id] = bundle
    return result


def _has_duplicates(values: Sequence[str]) -> bool:
    return len(values) != len(set(values))


def validate_frozen_inputs(
    ledger: RunLedger,
    bundles: Mapping[str, ConversationBundle] | Sequence[ConversationBundle],
) -> dict[str, ConversationBundle]:
    """Reconcile a complete frozen ledger with its exact conversation bundles."""

    normalized = _normalize_bundles(bundles)
    if ledger.schema_version != SCHEMA_VERSION or ledger.study_id != STUDY_ID:
        raise ValueError("ledger does not belong to the reviewed study schema")
    if ledger.status != "complete":
        raise ValueError("only a complete cohort can be emitted")
    if _has_duplicates(ledger.planned_coordinate_ids):
        raise ValueError("ledger contains duplicate planned coordinates")
    if _has_duplicates(ledger.completed_coordinate_ids):
        raise ValueError("ledger contains duplicate completed coordinates")
    planned = set(ledger.planned_coordinate_ids)
    completed = set(ledger.completed_coordinate_ids)
    unavailable = set(ledger.unavailable)
    if completed & unavailable:
        raise ValueError("coordinate is both complete and unavailable")
    if completed | unavailable != planned:
        raise ValueError(
            "planned coordinates do not reconcile with completed and unavailable"
        )
    if ledger.completed_coordinate_ids != [
        coordinate_id
        for coordinate_id in ledger.planned_coordinate_ids
        if coordinate_id in completed
    ]:
        raise ValueError("completed coordinate order must follow the exact plan")
    if set(ledger.bundle_digests) != set(ledger.planned_coordinate_ids):
        raise ValueError("ledger bundle digests do not cover the exact plan")
    if set(normalized) != set(ledger.planned_coordinate_ids):
        raise ValueError("loaded bundles do not cover the exact frozen ledger")
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("ledger digest does not match its canonical payload")

    conversation_ids: set[str] = set()
    for coordinate_id in ledger.planned_coordinate_ids:
        bundle = normalized[coordinate_id]
        if bundle.schema_version != SCHEMA_VERSION or bundle.study_id != STUDY_ID:
            raise ValueError(f"bundle schema drift: {coordinate_id}")
        if bundle.coordinate.coordinate_id != coordinate_id:
            raise ValueError(f"bundle coordinate drift: {coordinate_id}")
        if coordinate_id in completed:
            if bundle.status != "complete" or not bundle.terminal_response:
                raise ValueError(
                    f"bundle is not terminal and complete: {coordinate_id}"
                )
            if bundle.unavailable_reason is not None:
                raise ValueError(
                    f"complete bundle has unavailable reason: {coordinate_id}"
                )
        else:
            if bundle.status != "unavailable" or not bundle.unavailable_reason:
                raise ValueError(f"bundle is not frozen unavailable: {coordinate_id}")
            if ledger.unavailable[coordinate_id] != bundle.unavailable_reason:
                raise ValueError(f"unavailable reason drift: {coordinate_id}")
        calculated = digest_value(bundle.unsigned_payload())
        if bundle.bundle_sha256 != calculated:
            raise ValueError(f"bundle digest drift: {coordinate_id}")
        if ledger.bundle_digests[coordinate_id] != calculated:
            raise ValueError(f"ledger points to another bundle: {coordinate_id}")
        if bundle.conversation_id in conversation_ids:
            raise ValueError(f"duplicate conversation id: {bundle.conversation_id}")
        conversation_ids.add(bundle.conversation_id)

        if coordinate_id in completed:
            # This also validates transcript/call reconciliation and the strict
            # public projection before a network-capable exporter is constructed.
            _build_public_conversation(bundle)

    return normalized


def _safe_key(key: str) -> None:
    normalized = key.casefold().replace("-", "_")
    if any(fragment in normalized for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS):
        raise ValueError(f"forbidden public key: {key}")
    if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
        raise ValueError(f"credential-like public key: {key}")


def _without_conversation_prefix(path: tuple[str | int, ...]) -> tuple[str | int, ...]:
    if (
        len(path) >= 2
        and path[0] == "conversations"
        and isinstance(path[1], int)
    ):
        return path[2:]
    return path


def _is_message_part_path(path: tuple[str | int, ...]) -> bool:
    relative = _without_conversation_prefix(path)
    if (
        len(relative) == 6
        and relative[0] == "turns"
        and isinstance(relative[1], int)
        and relative[2] in {"input_messages", "output_messages"}
        and isinstance(relative[3], int)
        and relative[4] == "parts"
        and isinstance(relative[5], int)
    ):
        return True
    return (
        len(relative) == 8
        and relative[0] == "turns"
        and isinstance(relative[1], int)
        and relative[2] == "children"
        and isinstance(relative[3], int)
        and relative[4] in {"input_messages", "output_messages"}
        and isinstance(relative[5], int)
        and relative[6] == "parts"
        and isinstance(relative[7], int)
    )


def _is_mcp_child_path(path: tuple[str | int, ...]) -> bool:
    relative = _without_conversation_prefix(path)
    return (
        len(relative) == 4
        and relative[0] == "turns"
        and isinstance(relative[1], int)
        and relative[2] == "children"
        and isinstance(relative[3], int)
    )


def _is_tool_call_identifier_slot(
    container: Mapping[str, Any],
    *,
    key: str,
    path: tuple[str | int, ...],
) -> bool:
    if key == "id" and _is_message_part_path(path):
        return container.get("type") in {"tool_call", "tool_call_response"}
    if key == "tool_call_id" and _is_mcp_child_path(path):
        return container.get("kind") == "mcp_tool"
    return False


def _validate_public_value(
    value: Any,
    *,
    path: str = "public",
    _path_parts: tuple[str | int, ...] = (),
    _allow_canonical_tool_call_uuid: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _safe_key(key)
            _validate_public_value(
                child,
                path=f"{path}.{key}",
                _path_parts=(*_path_parts, key),
                _allow_canonical_tool_call_uuid=(
                    _is_tool_call_identifier_slot(
                        value,
                        key=key,
                        path=_path_parts,
                    )
                ),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_public_value(
                child,
                path=f"{path}[{index}]",
                _path_parts=(*_path_parts, index),
            )
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if not isinstance(value, str):
        raise ValueError(f"{path} contains unsupported type {type(value).__name__}")
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"{path} contains credential-like or local-only text")
    for match in _EMAIL_PATTERN.finditer(value):
        domain = match.group(0).rsplit("@", 1)[1].casefold()
        if domain not in _ALLOWED_EMAIL_DOMAINS:
            raise ValueError(f"{path} contains a non-fictional email address")
    if _PAYMENT_CARD_PATTERN.search(value) and not (
        _allow_canonical_tool_call_uuid
        and _CANONICAL_UUID_PATTERN.fullmatch(value)
    ):
        raise ValueError(f"{path} contains a payment-card-like number")
    if _SSN_PATTERN.search(value):
        raise ValueError(f"{path} contains an SSN-like value")
    if _PHONE_PATTERN.search(value):
        raise ValueError(f"{path} contains a phone-number-like value")


def _parse_json_object(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON object")
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def _public_message(
    message: Mapping[str, Any],
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
) -> dict[str, Any]:
    role = message.get("role")
    if role == "system":
        raise ValueError("system messages must never enter the public projection")
    if role not in {"user", "assistant", "tool"}:
        raise ValueError(f"unsupported public message role: {role!r}")

    if role == "tool":
        allowed = {"role", "tool_call_id", "content"}
        if set(message) - allowed:
            raise ValueError("tool message contains unsupported public fields")
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool message requires a call id")
        record = tools_by_id.get(call_id)
        if record is None:
            response: Any
            try:
                response = json.loads(str(message.get("content") or "null"))
            except json.JSONDecodeError:
                response = str(message.get("content") or "")
        elif record.error_type:
            response = {"error_type": record.error_type}
        else:
            response = record.result
        return {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": call_id,
                    "response": response,
                }
            ],
        }

    allowed = {"role", "content"}
    if role == "assistant":
        allowed.add("tool_calls")
    if set(message) - allowed:
        raise ValueError(f"{role} message contains unsupported public fields")
    parts: list[dict[str, Any]] = []
    content = message.get("content")
    if content not in (None, ""):
        if not isinstance(content, str):
            raise ValueError(f"{role} message content must be text")
        parts.append({"type": "text", "content": content})

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("assistant tool_calls must be a list")
    for raw in raw_calls:
        if not isinstance(raw, Mapping) or set(raw) != {"id", "type", "function"}:
            raise ValueError("assistant tool call has an unsupported shape")
        function = raw["function"]
        if (
            raw["type"] != "function"
            or not isinstance(raw["id"], str)
            or not isinstance(function, Mapping)
            or set(function) != {"name", "arguments"}
            or not isinstance(function["name"], str)
        ):
            raise ValueError("assistant tool call is malformed")
        parts.append(
            {
                "type": "tool_call",
                "id": raw["id"],
                "name": function["name"],
                "arguments": _parse_json_object(
                    function["arguments"], label="tool-call arguments"
                ),
            }
        )
    if not parts:
        raise ValueError(f"{role} message has no public content")
    return {"role": role, "parts": parts}


def _raw_message_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_bytes(dict(left)) == canonical_bytes(dict(right))


def _find_output_index(
    messages: Sequence[Mapping[str, Any]],
    call: ModelCall,
    *,
    start: int,
) -> int:
    for index in range(start, len(messages)):
        if _raw_message_equal(messages[index], call.output_message):
            return index
    raise ValueError(f"support-agent call is absent from transcript: {call.call_id}")


def _turn_index_for_message(user_indexes: Sequence[int], message_index: int) -> int:
    matches = [
        index for index, value in enumerate(user_indexes) if value < message_index
    ]
    if not matches:
        raise ValueError("agent activity occurred before the first user request")
    return matches[-1]


def _support_call_payload(
    call: ModelCall,
    *,
    tools_by_id: Mapping[str, ToolCallRecord],
) -> dict[str, Any]:
    public_input = [
        _public_message(message, tools_by_id=tools_by_id)
        for message in call.input_messages
        if message.get("role") != "system"
    ]
    public_output = [_public_message(call.output_message, tools_by_id=tools_by_id)]
    payload: dict[str, Any] = {
        "kind": "chat",
        "requested_model": call.requested_model,
        "returned_model": call.returned_model,
        "input_messages": public_input,
        "output_messages": public_output,
        "usage": {
            "input_tokens": call.usage.input_tokens,
            "output_tokens": call.usage.output_tokens,
        },
    }
    if call.temperature is not None:
        payload["temperature"] = call.temperature
    if call.seed is not None:
        payload["seed"] = call.seed
    return payload


def _tool_payload(record: ToolCallRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "mcp_tool",
        "tool_call_id": record.call_id,
        "name": record.name,
        "arguments": record.arguments,
    }
    if record.error_type:
        payload["error_type"] = record.error_type
    else:
        payload["result"] = record.result
    return payload


def _build_public_conversation(bundle: ConversationBundle) -> dict[str, Any]:
    messages = [dict(message) for message in bundle.messages]
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if not user_indexes:
        raise ValueError("complete bundle contains no user request")

    tools_by_id: dict[str, ToolCallRecord] = {}
    tool_message_index: dict[str, int] = {}
    for record in bundle.tool_calls:
        if record.call_id in tools_by_id:
            raise ValueError(f"duplicate model tool-call id: {record.call_id}")
        tools_by_id[record.call_id] = record
        occurrences: list[int] = []
        response_indexes: list[int] = []
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for raw in message.get("tool_calls") or []:
                    if isinstance(raw, Mapping) and raw.get("id") == record.call_id:
                        occurrences.append(index)
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == record.call_id
            ):
                response_indexes.append(index)
        if len(occurrences) != 1 or len(response_indexes) != 1:
            raise ValueError(
                f"tool record does not reconcile with one call and response: "
                f"{record.call_id}"
            )
        if response_indexes[0] <= occurrences[0]:
            raise ValueError(f"tool response precedes its call: {record.call_id}")
        tool_message_index[record.call_id] = occurrences[0]

    children_by_turn: list[list[tuple[int, int, dict[str, Any]]]] = [
        [] for _ in user_indexes
    ]
    search_start = 0
    for call in bundle.model_calls:
        if call.participant != "support_agent":
            continue
        output_index = _find_output_index(messages, call, start=search_start)
        search_start = output_index + 1
        raw_public_input = [
            dict(message)
            for message in call.input_messages
            if message.get("role") != "system"
        ]
        if len(raw_public_input) != output_index or any(
            not _raw_message_equal(actual, expected)
            for actual, expected in zip(messages[:output_index], raw_public_input)
        ):
            raise ValueError(
                f"support-agent input does not match transcript: {call.call_id}"
            )
        turn_index = _turn_index_for_message(user_indexes, output_index)
        children_by_turn[turn_index].append(
            (
                output_index,
                0,
                _support_call_payload(call, tools_by_id=tools_by_id),
            )
        )

    for order, record in enumerate(bundle.tool_calls):
        # A prepared write is an in-process confirmation control, not an MCP
        # execution. Its attempted call and action_prepared response remain in
        # the surrounding chat evidence, but emitting an execute_tool span
        # would falsely claim that the store tool ran.
        if record.execution_disposition != "executed":
            continue
        message_index = tool_message_index[record.call_id]
        turn_index = _turn_index_for_message(user_indexes, message_index)
        children_by_turn[turn_index].append(
            (message_index, order + 1, _tool_payload(record))
        )

    turns: list[dict[str, Any]] = []
    for turn_index, user_index in enumerate(user_indexes):
        end = (
            user_indexes[turn_index + 1]
            if turn_index + 1 < len(user_indexes)
            else len(messages)
        )
        final_index = next(
            (
                index
                for index in range(end - 1, user_index, -1)
                if messages[index].get("role") == "assistant"
                and messages[index].get("content") not in (None, "")
            ),
            None,
        )
        if final_index is None:
            raise ValueError(
                f"user turn {turn_index + 1} has no final assistant answer"
            )
        turns.append(
            {
                "turn_index": turn_index + 1,
                "input_messages": [
                    _public_message(message, tools_by_id=tools_by_id)
                    for message in messages[: user_index + 1]
                ],
                "output_messages": [
                    _public_message(messages[final_index], tools_by_id=tools_by_id)
                ],
                "children": [
                    payload for _, _, payload in sorted(children_by_turn[turn_index])
                ],
            }
        )

    conversation = {
        "conversation_id": bundle.conversation_id,
        "turns": turns,
    }
    _validate_public_value(conversation)
    return conversation


def build_public_payload(
    ledger: RunLedger,
    bundles: Mapping[str, ConversationBundle] | Sequence[ConversationBundle],
) -> dict[str, Any]:
    """Build the canonical, public-only telemetry plan and bind it by digest."""

    normalized = validate_frozen_inputs(ledger, bundles)
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "destination_project": DESTINATION_PROJECT,
        "conversations": [
            _build_public_conversation(normalized[coordinate_id])
            for coordinate_id in ledger.completed_coordinate_ids
        ],
    }
    _validate_public_value(unsigned)
    payload = {**unsigned, "payload_sha256": digest_value(unsigned)}
    validate_public_payload(payload)
    return payload


def validate_public_payload(payload: Mapping[str, Any]) -> None:
    allowed = {
        "schema_version",
        "destination_project",
        "conversations",
        "payload_sha256",
    }
    if set(payload) != allowed:
        raise ValueError("public payload has an unsupported top-level shape")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported public payload schema")
    if payload["destination_project"] != DESTINATION_PROJECT:
        raise ValueError("public payload points at an unreviewed destination")
    expected = digest_value(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    if payload["payload_sha256"] != expected:
        raise ValueError("public payload digest drift")
    _validate_public_value(payload)
    for conversation in payload["conversations"]:
        conversation_id = str(conversation["conversation_id"])
        for turn in conversation["turns"]:
            attribute_sets = [_root_attributes(conversation_id, turn)]
            for child in turn["children"]:
                if child["kind"] == "chat":
                    attribute_sets.append(_chat_attributes(conversation_id, child))
                elif child["kind"] == "mcp_tool":
                    attribute_sets.append(_mcp_attributes(conversation_id, child))
                else:
                    raise ValueError(f"unsupported child span kind: {child['kind']}")
            for attributes in attribute_sets:
                for key, value in attributes.items():
                    if (
                        isinstance(value, str)
                        and len(value) > MAX_PUBLIC_ATTRIBUTE_CHARACTERS
                    ):
                        raise ValueError(
                            f"public span attribute exceeds reviewed limit: {key}"
                        )


def _root_attributes(conversation_id: str, turn: Mapping[str, Any]) -> dict[str, Any]:
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.conversation.id": conversation_id,
        "gen_ai.input.messages": _json_string(turn["input_messages"]),
        "gen_ai.output.messages": _json_string(turn["output_messages"]),
    }
    if set(attributes) - ROOT_ATTRIBUTE_KEYS:  # pragma: no cover - defensive
        raise AssertionError("root telemetry exceeded its public allowlist")
    return attributes


def _chat_attributes(conversation_id: str, child: Mapping[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": _PROVIDER_NAME,
        "gen_ai.conversation.id": conversation_id,
        "gen_ai.request.model": child["requested_model"],
        "gen_ai.response.model": child["returned_model"],
        "gen_ai.input.messages": _json_string(child["input_messages"]),
        "gen_ai.output.messages": _json_string(child["output_messages"]),
        "gen_ai.usage.input_tokens": child["usage"]["input_tokens"],
        "gen_ai.usage.output_tokens": child["usage"]["output_tokens"],
    }
    if "temperature" in child:
        attributes["gen_ai.request.temperature"] = child["temperature"]
    if "seed" in child:
        attributes["gen_ai.request.seed"] = child["seed"]
    if set(attributes) - CHAT_ATTRIBUTE_KEYS:  # pragma: no cover - defensive
        raise AssertionError("chat telemetry exceeded its public allowlist")
    return attributes


def _mcp_attributes(conversation_id: str, child: Mapping[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "mcp.method.name": "tools/call",
        "mcp.protocol.version": _MCP_PROTOCOL_VERSION,
        "network.transport": _NETWORK_TRANSPORT,
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.conversation.id": conversation_id,
        "gen_ai.tool.name": child["name"],
        "gen_ai.tool.call.id": child["tool_call_id"],
        "gen_ai.tool.call.arguments": _json_string(child["arguments"]),
    }
    # The frozen record contains the model-issued tool-call ID, but the stdio
    # client's JSON-RPC request ID is not persisted. Omit it rather than invent it.
    if "result" in child:
        attributes["gen_ai.tool.call.result"] = _json_string(child["result"])
    if "error_type" in child:
        attributes["error.type"] = child["error_type"]
    if set(attributes) - MCP_ATTRIBUTE_KEYS:  # pragma: no cover - defensive
        raise AssertionError("MCP telemetry exceeded its public allowlist")
    return attributes


def emit_public_conversation(tracer: Tracer, conversation: Mapping[str, Any]) -> int:
    conversation_id = str(conversation["conversation_id"])
    span_count = 0
    for turn in conversation["turns"]:
        # An empty Context is deliberate: each actual user -> assistant turn is
        # one Agent Lens trace, while gen_ai.conversation.id joins the turns.
        with tracer.start_as_current_span(
            "invoke_agent",
            context=context.Context(),
            kind=SpanKind.INTERNAL,
            attributes=_root_attributes(conversation_id, turn),
        ) as root:
            span_count += 1
            parent_context = trace.set_span_in_context(root, context.Context())
            for child in turn["children"]:
                if child["kind"] == "chat":
                    with tracer.start_as_current_span(
                        f"chat {child['requested_model']}",
                        context=parent_context,
                        kind=SpanKind.CLIENT,
                        attributes=_chat_attributes(conversation_id, child),
                    ) as chat_span:
                        chat_span.set_status(Status(StatusCode.OK))
                    span_count += 1
                    continue
                if child["kind"] != "mcp_tool":
                    raise ValueError(f"unsupported child span kind: {child['kind']}")
                with tracer.start_as_current_span(
                    f"tools/call {child['name']}",
                    context=parent_context,
                    kind=SpanKind.CLIENT,
                    attributes=_mcp_attributes(conversation_id, child),
                ) as tool_span:
                    if "error_type" in child:
                        tool_span.set_status(
                            Status(StatusCode.ERROR, str(child["error_type"]))
                        )
                    else:
                        tool_span.set_status(Status(StatusCode.OK))
                span_count += 1
            root.set_status(Status(StatusCode.OK))
    return span_count


def count_public_conversations(
    conversations: Sequence[Mapping[str, Any]],
) -> EmissionCounts:
    """Count the spans in an already validated public conversation slice."""

    turns = 0
    chat_spans = 0
    tool_spans = 0
    for conversation in conversations:
        for turn in conversation["turns"]:
            turns += 1
            chat_spans += sum(child["kind"] == "chat" for child in turn["children"])
            tool_spans += sum(child["kind"] == "mcp_tool" for child in turn["children"])
    return EmissionCounts(
        conversations=len(conversations),
        turns=turns,
        chat_spans=chat_spans,
        tool_spans=tool_spans,
    )


def emit_public_payload(tracer: Tracer, payload: Mapping[str, Any]) -> EmissionCounts:
    validate_public_payload(payload)
    conversations = payload["conversations"]
    counts = count_public_conversations(conversations)
    emitted = 0
    for conversation in conversations:
        emitted += emit_public_conversation(tracer, conversation)
    if emitted != counts.total_spans:  # pragma: no cover - defensive
        raise AssertionError("telemetry span count drift")
    return counts


def emit_bundle(tracer: Tracer, bundle: ConversationBundle) -> EmissionCounts:
    """Emit one already-validated bundle without exposing its study coordinate."""

    if bundle.status != "complete" or not bundle.terminal_response:
        raise ValueError("only a complete terminal bundle can be emitted")
    if bundle.bundle_sha256 != digest_value(bundle.unsigned_payload()):
        raise ValueError("bundle digest drift")
    conversation = _build_public_conversation(bundle)
    turns = len(conversation["turns"])
    chat_spans = sum(
        child["kind"] == "chat"
        for turn in conversation["turns"]
        for child in turn["children"]
    )
    tool_spans = sum(
        child["kind"] == "mcp_tool"
        for turn in conversation["turns"]
        for child in turn["children"]
    )
    emitted = emit_public_conversation(tracer, conversation)
    counts = EmissionCounts(1, turns, chat_spans, tool_spans)
    if emitted != counts.total_spans:  # pragma: no cover - defensive
        raise AssertionError("telemetry span count drift")
    return counts
