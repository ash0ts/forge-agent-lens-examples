from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode

from retail_support_study_v1.inference import Completion, InferenceCallError
from retail_support_study_v1.live_telemetry import LiveTelemetrySession
from retail_support_study_v1.mcp_protocol import MCPProtocolError, MCPToolError
from retail_support_study_v1.schemas import ModelCall, Usage


def _session() -> tuple[LiveTelemetrySession, TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return (
        LiveTelemetrySession.from_provider(
            provider,
            agent_version="live-test-version",
        ),
        provider,
        exporter,
    )


def _model_call(
    request: dict[str, Any],
    *,
    call_id: str,
    output: dict[str, Any],
    participant: str = "support_agent",
) -> ModelCall:
    return ModelCall(
        call_id=call_id,
        participant=participant,
        requested_model=str(request["model"]),
        returned_model=str(request["model"]),
        response_id=f"response-{call_id}",
        finish_reason="tool_calls" if output.get("tool_calls") else "stop",
        temperature=float(request["temperature"]),
        requested_max_tokens=int(request["max_tokens"]),
        seed=request.get("seed"),
        input_messages=list(request["messages"]),
        output_message=output,
        usage=Usage(input_tokens=12, output_tokens=4, total_tokens=16),
    )


def _complete(
    records: list[ModelCall],
    request: dict[str, Any],
    *,
    call_id: str,
    output: dict[str, Any],
    timing: dict[str, int] | None = None,
) -> Completion:
    if timing is not None:
        timing["start"] = time.time_ns()
        time.sleep(0.001)
    record = _model_call(request, call_id=call_id, output=output)
    records.append(record)
    if timing is not None:
        timing["end"] = time.time_ns()
    return Completion(message=output, record=record)


def _request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "participant": "support_agent",
        "model": "zai-org/GLM-5.3-Flash",
        "temperature": 0.0,
        "seed": None,
        "messages": [
            {"role": "system", "content": "PRIVATE support policy"},
            *messages,
        ],
        "max_tokens": 4096,
    }


def test_live_turn_wraps_real_support_and_mcp_operations() -> None:
    session, provider, exporter = _session()
    conversation_id = "conversation-live-001"
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Where is order R0000042?"},
    ]
    turn = session.start_turn(conversation_id, messages, {})

    records: list[ModelCall] = []
    tool_call = {
        "id": "tool-call-001",
        "type": "function",
        "function": {
            "name": "get_order_details",
            "arguments": json.dumps({"order_id": "R0000042"}),
        },
    }
    first_output = {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call],
    }
    chat_timing: dict[str, int] = {}
    first_request = _request(messages)
    first = turn.complete_support_call(
        lambda: _complete(
            records,
            first_request,
            call_id="support-001",
            output=first_output,
            timing=chat_timing,
        ),
        records,
        first_request,
    )
    messages.append(first.message)

    tool_timing: dict[str, int] = {}
    active_tool_span_ids: list[int] = []

    def execute_tool(record_request_id: Any) -> dict[str, Any]:
        active_tool_span_ids.append(
            trace.get_current_span().get_span_context().span_id
        )
        record_request_id(42)
        tool_timing["start"] = time.time_ns()
        time.sleep(0.001)
        result = {"order_id": "R0000042", "status": "shipped"}
        tool_timing["end"] = time.time_ns()
        return result

    result = turn.execute_mcp(
        execute_tool,
        name="get_order_details",
        arguments={"order_id": "R0000042"},
        call_id="tool-call-001",
        session_id="mcp-session-001",
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "tool-call-001",
            "content": json.dumps(result),
        }
    )

    final_output = {
        "role": "assistant",
        "content": "Order R0000042 has shipped.",
    }
    final_request = _request(messages)
    final = turn.complete_support_call(
        lambda: _complete(
            records,
            final_request,
            call_id="support-002",
            output=final_output,
        ),
        records,
        final_request,
    )
    turn.finish(final.message, {})
    session.force_flush()
    session.shutdown()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    chats = [span for span in spans if span.name.startswith("chat ")]
    tools = [span for span in spans if span.name.startswith("tools/call ")]
    assert len(tools) == 1
    tool = tools[0]

    assert root.name == "invoke_agent retail-support-agent"
    assert root.kind is SpanKind.INTERNAL
    assert root.parent is None
    assert root.status.status_code is StatusCode.UNSET
    assert root.attributes["gen_ai.agent.name"] == "retail-support-agent"
    assert root.attributes["gen_ai.agent.version"] == "live-test-version"
    assert root.attributes["gen_ai.request.model"] == "zai-org/GLM-5.3-Flash"
    assert root.attributes["gen_ai.request.max_tokens"] == 4096
    assert root.attributes["gen_ai.request.temperature"] == 0.0
    assert "gen_ai.usage.input_tokens" not in root.attributes
    assert "gen_ai.usage.output_tokens" not in root.attributes
    assert len(chats) == 2
    assert all(span.kind is SpanKind.CLIENT for span in [*chats, tool])
    assert all(span.parent.span_id == root.context.span_id for span in [*chats, tool])
    assert all(span.context.trace_id == root.context.trace_id for span in [*chats, tool])
    assert all(span.status.status_code is StatusCode.UNSET for span in chats)
    assert sum(span.attributes["gen_ai.usage.input_tokens"] for span in chats) == 24
    assert sum(span.attributes["gen_ai.usage.output_tokens"] for span in chats) == 8
    assert tool.status.status_code is StatusCode.UNSET
    assert tool.attributes["mcp.method.name"] == "tools/call"
    assert tool.attributes["mcp.protocol.version"] == "2025-06-18"
    assert tool.attributes["mcp.session.id"] == "mcp-session-001"
    assert tool.attributes["network.transport"] == "pipe"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    # Agent identity belongs to the invoke_agent root. Adding only a name to
    # child MCP spans creates a second, blank-version agent attribution bucket.
    assert "gen_ai.agent.name" not in tool.attributes
    assert "gen_ai.agent.version" not in tool.attributes
    assert tool.attributes["gen_ai.tool.type"] == "function"
    assert tool.attributes["gen_ai.tool.call.id"] == "tool-call-001"
    assert tool.attributes["jsonrpc.request.id"] == "42"
    assert active_tool_span_ids == [tool.context.span_id]
    assert all(
        span.attributes["server.address"] == "api.inference.wandb.ai"
        and span.attributes["server.port"] == 443
        for span in chats
    )

    first_chat = next(
        span
        for span in chats
        if span.attributes["gen_ai.response.id"] == "response-support-001"
    )
    assert first_chat.start_time <= chat_timing["start"]
    assert chat_timing["end"] <= first_chat.end_time
    assert first_chat.end_time > first_chat.start_time
    assert tool.start_time <= tool_timing["start"]
    assert tool_timing["end"] <= tool.end_time
    assert tool.end_time > tool.start_time

    encoded = json.dumps([dict(span.attributes) for span in spans], sort_keys=True)
    assert "PRIVATE support policy" not in encoded
    assert "user_simulator" not in encoded
    assert conversation_id in encoded


def test_jev_decisions_are_live_non_chat_spans_with_reviewed_parentage() -> None:
    session, provider, exporter = _session()
    conversation_id = "conversation-jev-001"
    runner = session.jev_call_runner(conversation_id)
    messages = [{"role": "user", "content": "Please return the damaged cup."}]
    turn = session.start_turn(
        conversation_id,
        messages,
        {},
        agent_name="retail-support-router",
    )
    runner.bind_turn(turn)
    private_state = {"first_customer_message": "PRIVATE routing message"}
    route_response = SimpleNamespace(
        model="jev-1.13.0",
        usage=SimpleNamespace(input_tokens=18, output_tokens=0),
        answers={
            "agent_profile": SimpleNamespace(
                type="choice",
                choice="returns_resolution",
                confidence=0.91,
                probabilities={
                    "order_management": 0.03,
                    "returns_resolution": 0.91,
                    "generalist": 0.06,
                },
            )
        },
    )
    route_timing: dict[str, int] = {}

    def route_call() -> Any:
        route_timing["start"] = time.time_ns()
        time.sleep(0.001)
        route_timing["end"] = time.time_ns()
        return route_response

    assert runner(
        operation="route",
        state=private_state,
        questions={"agent_profile": object()},
        backend="live",
        requested_model="jev-1.13.0",
        timeout_seconds=2.0,
        invoke=route_call,
    ) is route_response

    turn.resolve_agent("retail-returns-resolution")
    assessment_response = SimpleNamespace(
        model="jev-1.13.0",
        usage=SimpleNamespace(input_tokens=44, output_tokens=0),
        answers={
            name: SimpleNamespace(type="noul", noul=probability)
            for name, probability in {
                "redundant_confirmation": 0.84,
                "partial_completion": 0.08,
                "unsupported_definitive_answer": 0.03,
            }.items()
        },
    )
    assert runner(
        operation="assessment",
        state={"current_turn": "PRIVATE assessment state"},
        questions={name: object() for name in assessment_response.answers},
        backend="live",
        requested_model="jev-1.13.0",
        timeout_seconds=2.0,
        invoke=lambda: assessment_response,
    ) is assessment_response
    runner.clear_turn()
    turn.finish(
        {"role": "assistant", "content": "I can help with that return."},
        {},
    )
    session.force_flush()
    session.shutdown()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    decisions = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "evaluate"
    ]
    assert len(decisions) == 2
    route = next(
        span
        for span in decisions
        if span.attributes["typesafe.decision.purpose"] == "route"
    )
    assessment = next(
        span
        for span in decisions
        if span.attributes["typesafe.decision.purpose"] == "assessment"
    )

    assert route.kind is SpanKind.CLIENT
    assert route.name == "evaluate routing decision with jev-1.13.0"
    assert route.parent.span_id == root.context.span_id
    assert route.context.trace_id == root.context.trace_id
    assert route.start_time <= route_timing["start"]
    assert route_timing["end"] <= route.end_time
    assert assessment.parent.span_id == root.context.span_id
    assert assessment.context.trace_id == root.context.trace_id
    assert assessment.name == "evaluate behavior check with jev-1.13.0"
    assert root.name == "invoke_agent retail-returns-resolution"
    assert root.attributes["gen_ai.agent.name"] == "retail-returns-resolution"
    assert all(
        span.attributes["gen_ai.operation.name"] == "evaluate"
        and span.attributes["gen_ai.provider.name"] == "typesafe"
        and span.attributes["server.address"] == "api.typesafe.ai"
        and span.attributes["gen_ai.conversation.id"] == conversation_id
        for span in decisions
    )
    assert route.attributes["typesafe.question_set.id"].endswith("route-v1")
    assert assessment.attributes["typesafe.question_set.id"].endswith(
        "assessment-v1"
    )
    assert route.attributes["gen_ai.response.model"] == "jev-1.13.0"
    assert route.attributes["gen_ai.usage.input_tokens"] == 18
    assert assessment.attributes["gen_ai.usage.input_tokens"] == 44
    assert "returns_resolution" in route.attributes["typesafe.decision.answers"]
    encoded = json.dumps([dict(span.attributes) for span in spans], sort_keys=True)
    assert "PRIVATE routing message" not in encoded
    assert "PRIVATE assessment state" not in encoded


def test_mock_jev_decision_is_explicitly_local_and_never_attributed_to_typesafe() -> None:
    session, provider, exporter = _session()
    runner = session.jev_call_runner("conversation-mock-jev-001")
    response = SimpleNamespace(
        model="mock-jev-decision-v1",
        usage=SimpleNamespace(input_tokens=12, output_tokens=0),
        answers={
            "agent_profile": SimpleNamespace(
                type="choice",
                choice="generalist",
                confidence=0.82,
                probabilities={
                    "order_management": 0.09,
                    "returns_resolution": 0.09,
                    "generalist": 0.82,
                },
            )
        },
    )

    runner(
        operation="route",
        state={"first_customer_message": "PRIVATE mock routing message"},
        questions={"agent_profile": object()},
        backend="mock",
        requested_model="mock-jev-decision-v1",
        timeout_seconds=2.0,
        invoke=lambda: response,
    )
    session.force_flush()
    session.shutdown()
    provider.shutdown()

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "evaluate"
    )
    assert span.kind is SpanKind.INTERNAL
    assert span.name == "evaluate routing decision with mock-jev-decision-v1"
    assert span.attributes["gen_ai.provider.name"] == "retail_demo.mock"
    assert span.attributes["demo.decision.backend"] == "mock"
    assert span.attributes["demo.mock_jev.decision.purpose"] == "route"
    assert "server.address" not in span.attributes
    assert "typesafe.operation.name" not in span.attributes
    encoded = json.dumps(dict(span.attributes), sort_keys=True)
    assert "PRIVATE mock routing message" not in encoded
    assert "api.typesafe.ai" not in encoded


def test_live_json_attributes_preserve_reviewed_machine_ids_without_rescanning() -> None:
    session, provider, exporter = _session()
    # This deterministic UUID was rejected by the natural-language PII scanner
    # during the first live cohort because one numeric fragment happened to look
    # like a payment-card number. Conversation IDs are opaque OTel identifiers,
    # so validate their syntax rather than treating them as message content.
    conversation_id = "d45b4b3b-4a57-5127-8377-311063889590"
    tool_call_id = "12345678-1234-4234-8234-123456789012"
    response_id = "12345678-1234-5234-8234-123456789012"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Please find order R0000042."}
    ]
    turn = session.start_turn(conversation_id, messages, {})
    request = _request(messages)
    output = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "arguments": json.dumps({"order_id": "R0000042"}),
                },
            }
        ],
    }
    records: list[ModelCall] = []

    def complete() -> Completion:
        record = _model_call(
            request,
            call_id="machine-identifier-response",
            output=output,
        )
        record.response_id = response_id
        records.append(record)
        return Completion(message=output, record=record)

    completion = turn.complete_support_call(complete, records, request)
    messages.append(completion.message)
    result = turn.execute_mcp(
        lambda record_request_id: (
            record_request_id(7),
            {"order_id": "R0000042", "status": "shipped"},
        )[1],
        name="get_order_details",
        arguments={"order_id": "R0000042"},
        call_id=tool_call_id,
        session_id="mcp-session-machine-identifiers",
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result),
        }
    )
    turn.finish({"role": "assistant", "content": "The order has shipped."}, {})
    session.force_flush()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if span.name.startswith("chat "))
    tool = next(span for span in spans if span.name.startswith("tools/call "))
    assert chat.attributes["gen_ai.response.id"] == response_id
    assert tool.attributes["gen_ai.tool.call.id"] == tool_call_id
    assert tool_call_id in chat.attributes["gen_ai.output.messages"]


def test_live_messages_still_reject_payment_card_content() -> None:
    session, provider, _ = _session()
    with pytest.raises(ValueError, match="payment-card-like"):
        session.start_turn(
            "conversation-private-card",
            [{"role": "user", "content": "My card is 4111 1111 1111 1111."}],
            {},
        )
    provider.shutdown()


def test_recovered_mcp_error_marks_only_the_tool_span() -> None:
    session, provider, exporter = _session()
    messages = [{"role": "user", "content": "Please find my order."}]
    turn = session.start_turn("conversation-tool-error", messages, {})

    def fail_tool(record_request_id: Any) -> None:
        record_request_id(43)
        raise RuntimeError("private backend response must not be exported")

    with pytest.raises(RuntimeError, match="private backend"):
        turn.execute_mcp(
            fail_tool,
            name="get_order_details",
            arguments={"order_id": "R0000042"},
            call_id="tool-error-001",
            session_id="mcp-session-tool-error",
        )

    output = {"role": "assistant", "content": "I could not verify the order."}
    request = _request(messages)
    records: list[ModelCall] = []
    completion = turn.complete_support_call(
        lambda: _complete(
            records,
            request,
            call_id="support-after-tool-error",
            output=output,
        ),
        records,
        request,
    )
    turn.finish(completion.message, {})
    session.force_flush()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    tool = next(span for span in spans if span.name.startswith("tools/call "))
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.status.description == "RuntimeError"
    assert tool.attributes["error.type"] == "RuntimeError"
    assert root.status.status_code is StatusCode.UNSET
    encoded = json.dumps([dict(span.attributes) for span in spans])
    assert "private backend response" not in encoded


def test_mcp_tool_result_error_uses_semantic_convention_error_type() -> None:
    session, provider, exporter = _session()
    turn = session.start_turn(
        "conversation-mcp-tool-error",
        [{"role": "user", "content": "Please find my order."}],
        {},
    )

    def fail_tool(record_request_id: Any) -> None:
        record_request_id(44)
        raise MCPToolError("private tool failure detail")

    with pytest.raises(MCPToolError):
        turn.execute_mcp(
            fail_tool,
            name="get_order_details",
            arguments={"order_id": "R0000042"},
            call_id="tool-error-002",
            session_id="mcp-session-tool-result-error",
        )
    turn.fail(RuntimeError("stop after tool error"))
    session.force_flush()
    provider.shutdown()

    tool = next(
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("tools/call ")
    )
    assert tool.attributes["error.type"] == "tool_error"
    assert "rpc.response.status_code" not in tool.attributes
    assert "private tool failure detail" not in json.dumps(dict(tool.attributes))


def test_mcp_jsonrpc_error_preserves_response_status_code() -> None:
    session, provider, exporter = _session()
    turn = session.start_turn(
        "conversation-mcp-rpc-error",
        [{"role": "user", "content": "Please find my order."}],
        {},
    )

    def fail_request(record_request_id: Any) -> None:
        record_request_id(45)
        raise MCPProtocolError(
            "private invalid arguments detail",
            response_status_code="-32602",
        )

    with pytest.raises(MCPProtocolError):
        turn.execute_mcp(
            fail_request,
            name="get_order_details",
            arguments={"order_id": "R0000042"},
            call_id="tool-error-003",
            session_id="mcp-session-jsonrpc-error",
        )
    turn.fail(RuntimeError("stop after protocol error"))
    session.force_flush()
    provider.shutdown()

    tool = next(
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("tools/call ")
    )
    assert tool.attributes["error.type"] == "-32602"
    assert tool.attributes["rpc.response.status_code"] == "-32602"
    assert tool.status.description == "Invalid params"
    assert "private invalid arguments detail" not in json.dumps(
        dict(tool.attributes)
    )


def test_support_failure_ends_chat_and_root_as_errors_without_raw_text() -> None:
    session, provider, exporter = _session()
    messages = [{"role": "user", "content": "Please find my order."}]
    turn = session.start_turn("conversation-chat-error", messages, {})
    request = _request(messages)
    records: list[ModelCall] = []
    record = _model_call(
        request,
        call_id="failed-support-call",
        output={"role": "assistant", "content": "unusable response"},
    )
    record.response_validation_error = "PRIVATE provider envelope"

    def fail_inference() -> None:
        records.append(record)
        raise InferenceCallError(
            "PRIVATE provider envelope and credentials",
            record=record,
        )

    with pytest.raises(InferenceCallError):
        turn.complete_support_call(fail_inference, records, request)
    turn.fail(RuntimeError("PRIVATE outer failure detail"))
    session.force_flush()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    chat = next(span for span in spans if span.name.startswith("chat "))
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes["error.type"] == "RuntimeError"
    assert chat.status.status_code is StatusCode.ERROR
    assert chat.attributes["error.type"] == "InferenceCallError"
    encoded = json.dumps([dict(span.attributes) for span in spans])
    assert "PRIVATE provider envelope" not in encoded
    assert "PRIVATE outer failure" not in encoded


def test_explicit_contexts_do_not_cross_concurrent_conversations() -> None:
    session, provider, exporter = _session()

    def run_one(index: int) -> None:
        conversation_id = f"conversation-{index}"
        messages = [{"role": "user", "content": f"Request {index}"}]
        turn = session.start_turn(conversation_id, messages, {})
        request = _request(messages)
        records: list[ModelCall] = []
        output = {"role": "assistant", "content": f"Answer {index}"}
        completion = turn.complete_support_call(
            lambda: _complete(
                records,
                request,
                call_id=f"support-{index}",
                output=output,
            ),
            records,
            request,
        )
        turn.finish(completion.message, {})

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_one, range(2)))

    session.force_flush()
    provider.shutdown()
    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.name.startswith("invoke_agent ")]
    chats = [span for span in spans if span.name.startswith("chat ")]
    assert len(roots) == 2
    assert len(chats) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id
    roots_by_trace = {root.context.trace_id: root for root in roots}
    for chat in chats:
        root = roots_by_trace[chat.context.trace_id]
        assert chat.parent.span_id == root.context.span_id
        assert (
            chat.attributes["gen_ai.conversation.id"]
            == root.attributes["gen_ai.conversation.id"]
        )


def test_session_from_exporter_does_not_replace_global_provider() -> None:
    before = trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    with LiveTelemetrySession.from_exporter(
        exporter,
        agent_version="owned-provider-version",
    ) as session:
        turn = session.start_turn(
            "conversation-owned-provider",
            [{"role": "user", "content": "Hello"}],
            {},
        )
        turn.finish({"role": "assistant", "content": "Hi"}, {})
        session.force_flush()

    assert trace.get_tracer_provider() is before
    assert len(exporter.get_finished_spans()) == 1


class _FailingExporter(InMemorySpanExporter):
    def export(self, spans: Any) -> SpanExportResult:
        super().export(spans)
        return SpanExportResult.FAILURE


def test_checked_export_failure_surfaces_on_force_flush() -> None:
    session = LiveTelemetrySession.from_exporter(
        _FailingExporter(),
        agent_version="failing-exporter-version",
    )
    turn = session.start_turn(
        "conversation-export-failure",
        [{"role": "user", "content": "Hello"}],
        {},
    )
    turn.finish({"role": "assistant", "content": "Hi"}, {})

    with pytest.raises(RuntimeError, match="export failed"):
        session.force_flush()


def test_turn_records_resolved_agent_model_and_provider_route() -> None:
    session, provider, exporter = _session()
    messages = [{"role": "user", "content": "Please return this item."}]
    turn = session.start_turn(
        "conversation-routed-agent",
        messages,
        {},
        agent_name="retail-returns-resolution",
        requested_model="claude-haiku-4-5-20251001",
        provider_name="anthropic",
        server_address="api.anthropic.com",
    )
    request = {
        **_request(messages),
        "model": "claude-haiku-4-5-20251001",
    }
    records: list[ModelCall] = []
    completion = turn.complete_support_call(
        lambda: _complete(
            records,
            request,
            call_id="anthropic-support-call",
            output={"role": "assistant", "content": "I can help with that."},
        ),
        records,
        request,
    )
    turn.finish(completion.message, {})
    session.force_flush()
    provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    chat = next(span for span in spans if span.name.startswith("chat "))
    assert root.name == "invoke_agent retail-returns-resolution"
    assert root.attributes["gen_ai.agent.name"] == "retail-returns-resolution"
    assert root.attributes["gen_ai.request.model"] == (
        "claude-haiku-4-5-20251001"
    )
    assert chat.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat.attributes["server.address"] == "api.anthropic.com"
    assert chat.attributes["gen_ai.request.model"] == (
        "claude-haiku-4-5-20251001"
    )
