from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from tau2.domains.retail.data_model import RetailDB

import retail_support_study_v1.mcp_protocol as mcp_protocol_module
from retail_support_study_v1.behavior import RuntimeBehaviorProfile
from retail_support_study_v1.constants import PREPARED_DIR, WRITE_TOOLS
from retail_support_study_v1.mcp_protocol import (
    MCP_PROTOCOL_VERSION,
    MCPProcessClient,
    MCPProtocolError,
    MCPToolError,
    RetailMCPProtocol,
    _mcp_subprocess_environment,
)
from retail_support_study_v1.mcp_server import serve_stdio


def _address(number: int) -> dict[str, str]:
    return {
        "address1": f"{number} Lantern Way",
        "address2": "",
        "city": "Northstar",
        "country": "USA",
        "state": "NY",
        "zip": f"{10000 + number:05d}",
    }


@pytest.fixture
def retail_db() -> RetailDB:
    return RetailDB.model_validate(
        {
            "products": {},
            "users": {
                "retail_user_0001": {
                    "user_id": "retail_user_0001",
                    "name": {"first_name": "Avery", "last_name": "Arbor"},
                    "address": _address(1),
                    "email": "customer-0001@example.invalid",
                    "payment_methods": {
                        "credit_card_000001": {
                            "source": "credit_card",
                            "id": "credit_card_000001",
                            "brand": "visa",
                            "last_four": "1111",
                        }
                    },
                    "orders": ["#R0000001"],
                },
                "retail_user_0002": {
                    "user_id": "retail_user_0002",
                    "name": {"first_name": "Blair", "last_name": "Brook"},
                    "address": _address(2),
                    "email": "customer-0002@example.invalid",
                    "payment_methods": {
                        "credit_card_000002": {
                            "source": "credit_card",
                            "id": "credit_card_000002",
                            "brand": "mastercard",
                            "last_four": "2222",
                        }
                    },
                    "orders": ["#R0000002"],
                },
            },
            "orders": {
                "#R0000001": {
                    "order_id": "#R0000001",
                    "user_id": "retail_user_0001",
                    "address": _address(1),
                    "items": [],
                    "status": "pending",
                    "fulfillments": [],
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 42.5,
                            "payment_method_id": "credit_card_000001",
                        }
                    ],
                },
                "#R0000002": {
                    "order_id": "#R0000002",
                    "user_id": "retail_user_0002",
                    "address": _address(2),
                    "items": [],
                    "status": "pending",
                    "fulfillments": [],
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 75.0,
                            "payment_method_id": "credit_card_000002",
                        }
                    ],
                },
            },
        }
    )


def _behavior_profile(*, service_rate: float) -> RuntimeBehaviorProfile:
    return RuntimeBehaviorProfile.from_mapping(
        {
            "profile_id": "test-production-shaped-v1",
            "seed": "test-production-shaped-seed",
            "latency_by_family": {
                family: {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30}
                for family in (
                    "calculation",
                    "identity_lookup",
                    "entity_read",
                    "catalog_list",
                    "mutation",
                    "human_transfer",
                )
            },
            "service_unavailable_rate": service_rate,
            "upstream_timeout_rate": 0,
            "service_unavailable_delay_ms": [180, 450],
            "upstream_timeout_delay_ms": [1500, 2500],
        }
    )


def _request(
    protocol: RetailMCPProtocol,
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        message["params"] = params
    response = protocol.handle_message(message)
    assert response is not None
    return response


def _initialize(protocol: RetailMCPProtocol) -> None:
    response = _request(
        protocol,
        1,
        "initialize",
        {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1"},
        },
    )
    assert response["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    notification = protocol.handle_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )
    assert notification is None


def _tool_call(
    protocol: RetailMCPProtocol,
    request_id: int,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _request(
        protocol,
        request_id,
        "tools/call",
        {"name": name, "arguments": arguments},
    )


def test_stdio_handshake_discovery_and_pydantic_serialization(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
    )
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_order_details",
                "arguments": {"order_id": "#R0000001"},
            },
        },
    ]
    input_stream = io.StringIO(
        "".join(json.dumps(message) + "\n" for message in messages)
    )
    output_stream = io.StringIO()

    serve_stdio(
        protocol,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert responses[0]["result"]["capabilities"] == {"tools": {"listChanged": False}}

    tools = responses[1]["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == sorted(names)
    assert {
        "calculate",
        "cancel_pending_order",
        "get_order_details",
        "get_user_details",
        "return_delivered_order_items",
    }.issubset(names)
    get_order = next(tool for tool in tools if tool["name"] == "get_order_details")
    assert get_order["inputSchema"]["additionalProperties"] is False
    assert "order_id" in get_order["inputSchema"]["required"]

    tool_result = responses[2]["result"]
    assert tool_result["isError"] is False
    assert tool_result["structuredContent"]["result"]["order_id"] == "#R0000001"
    assert json.loads(tool_result["content"][0]["text"])["status"] == "pending"


def test_argument_protocol_and_execution_errors(retail_db: RetailDB) -> None:
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
    )

    before_initialize = _request(protocol, 1, "tools/list")
    assert before_initialize["error"]["code"] == -32002
    _initialize(protocol)

    missing = _tool_call(protocol, 2, "get_order_details", {})
    assert missing["error"]["code"] == -32602
    assert "order_id" in missing["error"]["data"]["detail"]

    extra = _tool_call(
        protocol,
        3,
        "get_order_details",
        {"order_id": "#R0000001", "user_id": "retail_user_0001"},
    )
    assert extra["error"]["code"] == -32602
    assert "unexpected argument" in extra["error"]["data"]["detail"]

    unknown = _tool_call(protocol, 4, "not_a_tool", {})
    assert unknown["error"]["code"] == -32602
    assert "unknown tool" in unknown["error"]["data"]["detail"]

    tool_error = _tool_call(
        protocol,
        5,
        "calculate",
        {"expression": "2 + forbidden"},
    )
    assert tool_error["result"]["isError"] is True
    payload = json.loads(tool_error["result"]["content"][0]["text"])
    assert payload["error"]["code"] == "tool_error"


def test_mcp_server_extracts_w3c_context_and_records_a_remote_child_span(
    retail_db: RetailDB,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("retail-support-study.test")
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
        session_id="stdio-session-001",
        tracer=tracer,
    )
    _initialize(protocol)

    with tracer.start_as_current_span(
        "mcp-client",
        kind=SpanKind.CLIENT,
    ) as client_span:
        carrier: dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier)
        response = _request(
            protocol,
            42,
            "tools/call",
            {
                "name": "get_order_details",
                "arguments": {"order_id": "#R0000001"},
                "_meta": carrier,
            },
        )
        client_context = client_span.get_span_context()

    provider.shutdown()
    assert response["result"]["isError"] is False
    spans = exporter.get_finished_spans()
    server = next(span for span in spans if span.kind is SpanKind.SERVER)
    assert server.name == "tools/call get_order_details"
    assert server.context.trace_id == client_context.trace_id
    assert server.parent is not None
    assert server.parent.span_id == client_context.span_id
    assert server.parent.is_remote is True
    assert server.status.status_code is StatusCode.UNSET
    assert server.attributes == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "get_order_details",
        "jsonrpc.request.id": "42",
        "mcp.method.name": "tools/call",
        "mcp.protocol.version": "2025-06-18",
        "mcp.session.id": "stdio-session-001",
        "network.transport": "pipe",
    }


def test_mcp_server_records_jsonrpc_and_tool_result_errors(
    retail_db: RetailDB,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
        session_id="stdio-session-errors",
        tracer=provider.get_tracer("retail-support-study.test"),
    )
    _initialize(protocol)

    missing = _tool_call(protocol, 42, "get_order_details", {})
    tool_error = _tool_call(
        protocol,
        43,
        "calculate",
        {"expression": "2 + forbidden"},
    )
    provider.shutdown()

    assert missing["error"]["code"] == -32602
    assert tool_error["result"]["isError"] is True
    spans = {
        span.attributes["jsonrpc.request.id"]: span
        for span in exporter.get_finished_spans()
        if span.kind is SpanKind.SERVER
    }
    invalid = spans["42"]
    assert invalid.attributes["error.type"] == "-32602"
    assert invalid.attributes["rpc.response.status_code"] == "-32602"
    assert invalid.status.status_code is StatusCode.ERROR
    assert invalid.status.description == "Invalid params"
    failed_tool = spans["43"]
    assert failed_tool.attributes["error.type"] == "tool_error"
    assert "rpc.response.status_code" not in failed_tool.attributes
    assert failed_tool.status.status_code is StatusCode.ERROR


def test_private_argument_validation_normalizes_without_mutating(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
    )
    _initialize(protocol)
    arguments = {
        "order_id": "#R0000001",
        "reason": "ordered by mistake",
    }

    response = _request(
        protocol,
        2,
        "study/tools/validate",
        {"name": "cancel_pending_order", "arguments": arguments},
    )

    assert response["result"]["arguments"] == arguments
    assert retail_db.orders["#R0000001"].status == "pending"

    denied = _request(
        protocol,
        3,
        "study/tools/validate",
        {
            "name": "cancel_pending_order",
            "arguments": {
                "order_id": "#R0000002",
                "reason": "ordered by mistake",
            },
        },
    )
    assert denied["error"]["code"] == -32602
    assert retail_db.orders["#R0000002"].status == "pending"

    malformed_output = io.StringIO()
    serve_stdio(
        protocol,
        input_stream=io.StringIO("{not-json}\n"),
        output_stream=malformed_output,
    )
    parse_error = json.loads(malformed_output.getvalue())
    assert parse_error["id"] is None
    assert parse_error["error"]["code"] == -32700


def test_cross_account_user_order_lookup_and_write_are_denied(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
    )
    _initialize(protocol)

    attempts = [
        ("get_user_details", {"user_id": "retail_user_0002"}),
        ("get_order_details", {"order_id": "#R0000002"}),
        (
            "find_user_id_by_email",
            {"email": "customer-0002@example.invalid"},
        ),
        (
            "cancel_pending_order",
            {"order_id": "#R0000002", "reason": "ordered by mistake"},
        ),
    ]

    for request_id, (name, arguments) in enumerate(attempts, start=10):
        response = _tool_call(protocol, request_id, name, arguments)
        assert response["result"]["isError"] is True
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload == {
            "error": {
                "code": "permission_denied",
                "message": "tool access is limited to the authenticated account",
            }
        }

    assert retail_db.orders["#R0000002"].status == "pending"
    own_order = _tool_call(
        protocol,
        20,
        "get_order_details",
        {"order_id": "#R0000001"},
    )
    assert own_order["result"]["isError"] is False


def test_identity_lookup_binds_an_unauthenticated_session(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(retail_db)
    _initialize(protocol)

    before_authentication = _tool_call(
        protocol,
        2,
        "get_order_details",
        {"order_id": "#R0000001"},
    )
    assert before_authentication["result"]["isError"] is True

    authenticated = _tool_call(
        protocol,
        3,
        "find_user_id_by_email",
        {"email": "customer-0001@example.invalid"},
    )
    assert authenticated["result"]["structuredContent"]["result"] == (
        "retail_user_0001"
    )
    assert protocol.authenticated_user_id == "retail_user_0001"

    own_order = _tool_call(
        protocol,
        4,
        "get_order_details",
        {"order_id": "#R0000001"},
    )
    assert own_order["result"]["isError"] is False
    other_order = _tool_call(
        protocol,
        5,
        "get_order_details",
        {"order_id": "#R0000002"},
    )
    assert other_order["result"]["isError"] is True


def test_loading_protocol_twice_produces_independent_databases(
    tmp_path: Any,
    retail_db: RetailDB,
) -> None:
    db_path = tmp_path / "db.json"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )
    first = RetailMCPProtocol.from_db_path(
        db_path,
        authenticated_user_id="retail_user_0001",
    )
    second = RetailMCPProtocol.from_db_path(
        db_path,
        authenticated_user_id="retail_user_0001",
    )
    _initialize(first)
    _initialize(second)

    cancelled = _tool_call(
        first,
        2,
        "cancel_pending_order",
        {"order_id": "#R0000001", "reason": "ordered by mistake"},
    )
    assert cancelled["result"]["isError"] is False
    assert first.db.orders["#R0000001"].status == "cancelled"
    assert second.db.orders["#R0000001"].status == "pending"


def test_process_client_uses_real_stdio_and_a_fresh_process_database(
    tmp_path: Any,
    retail_db: RetailDB,
) -> None:
    db_path = tmp_path / "db.json"
    policy_path = tmp_path / "policy.md"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )
    policy_path.write_text("test retail policy\n", encoding="utf-8")

    with MCPProcessClient(db_path=db_path, policy_path=policy_path) as first:
        first_session_id = first.session_id
        observed_request_ids: list[int] = []
        tool_names = {
            definition["function"]["name"] for definition in first.list_openai_tools()
        }
        assert "get_order_details" in tool_names
        with pytest.raises(MCPToolError, match="permission_denied"):
            first.call_tool("get_order_details", {"order_id": "#R0000001"})
        assert (
            first.call_tool(
                "find_user_id_by_email",
                {"email": "customer-0001@example.invalid"},
                on_request_id=observed_request_ids.append,
            )
            == "retail_user_0001"
        )
        cancelled = first.call_tool(
            "cancel_pending_order",
            {"order_id": "#R0000001", "reason": "ordered by mistake"},
        )
        assert cancelled["status"] == "cancelled"
        assert len(observed_request_ids) == 1
        assert isinstance(observed_request_ids[0], int)
        assert first.session_id == first_session_id

    with MCPProcessClient(db_path=db_path, policy_path=policy_path) as second:
        second_session_id = second.session_id
        second.call_tool(
            "find_user_id_by_email",
            {"email": "customer-0001@example.invalid"},
        )
        untouched = second.call_tool(
            "get_order_details",
            {"order_id": "#R0000001"},
        )
        assert untouched["status"] == "pending"
        assert second.session_id == second_session_id

    assert first_session_id != second_session_id


def test_process_client_reaps_child_after_normal_context_exit(
    tmp_path: Path,
    retail_db: RetailDB,
) -> None:
    db_path = tmp_path / "db.json"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )

    with MCPProcessClient(db_path=db_path) as client:
        process = client._process
        assert process is not None
        assert process.poll() is None
        assert len(client.list_tools()) == 16

    assert client._process is None
    assert process.poll() is not None


def test_process_client_reaps_child_when_context_body_raises(
    tmp_path: Path,
    retail_db: RetailDB,
) -> None:
    db_path = tmp_path / "db.json"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )
    process: subprocess.Popen[str] | None = None

    with pytest.raises(RuntimeError, match="conversation failed"):
        with MCPProcessClient(db_path=db_path) as client:
            process = client._process
            assert process is not None
            raise RuntimeError("conversation failed")

    assert client._process is None
    assert process is not None
    assert process.poll() is not None


def test_process_client_reaps_child_when_initialize_times_out(
    tmp_path: Path,
    retail_db: RetailDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "db.json"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )
    original_popen = subprocess.Popen
    spawned: list[subprocess.Popen[str]] = []

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(mcp_protocol_module.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        mcp_protocol_module.select,
        "select",
        lambda *_args, **_kwargs: ([], [], []),
    )
    client = MCPProcessClient(db_path=db_path)

    with pytest.raises(MCPProtocolError, match="did not respond"):
        client.start()

    assert client._process is None
    assert len(spawned) == 1
    assert spawned[0].poll() is not None


def test_process_client_injects_current_w3c_context_into_params_meta(
    tmp_path: Any,
    retail_db: RetailDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "db.json"
    db_path.write_text(
        json.dumps(retail_db.model_dump(mode="json")),
        encoding="utf-8",
    )
    client = MCPProcessClient(db_path=db_path)
    messages: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "_write", messages.append)
    monkeypatch.setattr(
        client,
        "_read",
        lambda: {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "structuredContent": {"result": {"status": "pending"}},
                "isError": False,
            },
        },
    )
    provider = TracerProvider(shutdown_on_exit=False)
    tracer = provider.get_tracer("retail-support-study.test")

    with tracer.start_as_current_span("mcp-client", kind=SpanKind.CLIENT) as span:
        expected = span.get_span_context()
        result = client.call_tool(
            "get_order_details",
            {"order_id": "#R0000001"},
        )
    provider.shutdown()

    assert result == {"status": "pending"}
    assert len(messages) == 1
    metadata = messages[0]["params"]["_meta"]
    assert set(metadata) <= {"traceparent", "tracestate"}
    assert "traceparent" in metadata
    extracted = TraceContextTextMapPropagator().extract(carrier=metadata)
    extracted_context = trace.get_current_span(extracted).get_span_context()
    assert extracted_context.trace_id == expected.trace_id
    assert extracted_context.span_id == expected.span_id
    assert extracted_context.is_remote is True


def test_mcp_subprocess_environment_does_not_inherit_credentials() -> None:
    child = _mcp_subprocess_environment(
        {
            "PATH": "/reviewed/bin",
            "LANG": "en_US.UTF-8",
            "WANDB_API_KEY": "must-not-reach-tools",
            "ANTHROPIC_API_KEY": "must-not-reach-tools",
            "OTEL_EXPORTER_OTLP_HEADERS": "authorization=must-not-reach-tools",
            "AWS_SECRET_ACCESS_KEY": "must-not-reach-tools",
        }
    )

    assert child == {
        "PATH": "/reviewed/bin",
        "LANG": "en_US.UTF-8",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    snapshot_child = _mcp_subprocess_environment(
        {"PYTHONPATH": "/mutable/authoring/tree"},
        package_source_dir=Path("/reviewed/runtime-snapshot/src"),
    )
    assert snapshot_child["PYTHONPATH"].endswith("/reviewed/runtime-snapshot/src")
    assert "/mutable/authoring/tree" not in snapshot_child["PYTHONPATH"]


def test_tool_schemas_describe_the_remapped_identifier_formats(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(retail_db)
    encoded = json.dumps(protocol.list_tools(), sort_keys=True)

    assert "#W0000000" not in encoded
    assert "1008292230" not in encoded
    assert "6086499569" not in encoded
    assert "sara_doe_496" not in encoded
    assert "gift_card_0000000" not in encoded
    assert "#R1234567" in encoded
    assert "item_99999" in encoded
    assert "prod_9999" in encoded
    assert "retail_user_9999" in encoded
    by_name = {tool["name"]: tool for tool in protocol.list_tools()}
    assert (
        "item_99999"
        in by_name["get_item_details"]["inputSchema"]["properties"]["item_id"][
            "description"
        ]
    )
    assert (
        "prod_9999"
        in by_name["get_product_details"]["inputSchema"]["properties"][
            "product_id"
        ]["description"]
    )


def test_write_tool_descriptions_match_prepare_confirm_runtime(
    retail_db: RetailDB,
) -> None:
    by_name = {
        tool["name"]: tool["description"]
        for tool in RetailMCPProtocol(retail_db).list_tools()
    }

    for name in WRITE_TOOLS:
        description = by_name[name]
        assert "the first call only prepares this exact action" in description
        assert "Do not ask for confirmation before the preparation call" in description
        assert "needs to explain" not in description


def test_multi_item_modification_binds_each_item_to_its_own_variant() -> None:
    protocol = RetailMCPProtocol.from_db_path(
        PREPARED_DIR / "db.json",
        authenticated_user_id="retail_user_0120",
    )
    _initialize(protocol)

    response = _tool_call(
        protocol,
        2,
        "modify_pending_order_items",
        {
            "order_id": "#R0000987",
            "item_ids": ["item_00574", "item_00023"],
            "new_item_ids": ["item_00201", "item_00027"],
            "payment_method_id": "gift_card_000315",
        },
    )

    assert response["result"]["isError"] is False
    result = response["result"]["structuredContent"]["result"]
    items = {item["item_id"]: item for item in result["items"]}
    assert items["item_00201"] == {
        "item_id": "item_00201",
        "name": "Running Shoes",
        "options": {
            "color": "black",
            "material": "synthetic",
            "size": "9",
            "sole": "rubber",
        },
        "price": 155.33,
        "product_id": "prod_0030",
    }
    assert items["item_00027"] == {
        "item_id": "item_00027",
        "name": "Mechanical Keyboard",
        "options": {
            "backlight": "none",
            "size": "80%",
            "switch type": "linear",
        },
        "price": 268.77,
        "product_id": "prod_0002",
    }
    assert result["payment_history"][-1] == {
        "amount": 41.91999999999999,
        "payment_method_id": "gift_card_000315",
        "transaction_type": "payment",
    }


def test_agent_tool_manifest_is_enforced_by_the_mcp_server(
    retail_db: RetailDB,
) -> None:
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
        allowed_tool_names=("get_user_details",),
    )
    _initialize(protocol)

    assert [tool["name"] for tool in protocol.list_tools()] == [
        "get_user_details"
    ]
    rejected = _tool_call(
        protocol,
        2,
        "cancel_pending_order",
        {"order_id": "#R0000001", "reason": "changed mind"},
    )
    assert rejected["error"]["code"] == -32602
    assert rejected["error"]["message"] == "Invalid params"


def test_injected_read_fault_is_real_delayed_and_limited_to_one(
    retail_db: RetailDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(mcp_protocol_module.time, "sleep", sleeps.append)
    protocol = RetailMCPProtocol(
        retail_db,
        authenticated_user_id="retail_user_0001",
        behavior_profile=_behavior_profile(service_rate=0.999),
        conversation_id="conversation-with-one-fault",
    )
    _initialize(protocol)

    failed = _tool_call(
        protocol,
        2,
        "get_user_details",
        {"user_id": "retail_user_0001"},
    )
    assert failed["result"]["isError"] is True
    payload = json.loads(failed["result"]["content"][0]["text"])
    assert payload["error"]["code"] == "service_unavailable"
    assert 0.180 <= sleeps[0] <= 0.450

    recovered = _tool_call(
        protocol,
        3,
        "get_user_details",
        {"user_id": "retail_user_0001"},
    )
    assert recovered["result"]["isError"] is False
    assert recovered["result"]["structuredContent"]["result"]["user_id"] == (
        "retail_user_0001"
    )
    assert len(sleeps) == 2
