from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from retail_support_study_v1.chat_runtime import ChatRuntime, SessionNotFound
from retail_support_study_v1.inference import Completion
from retail_support_study_v1.jev import JevConfig
from retail_support_study_v1.live_telemetry import LiveTelemetrySession
from retail_support_study_v1.schemas import ModelCall, Usage
from retail_support_study_v1.web import create_app


class _ScriptedClient:
    provider_name = "wandb"
    server_address = "api.inference.wandb.ai"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> Completion:
        self.requests.append(request)
        message = self.responses.pop(0)
        return Completion(
            message=message,
            record=ModelCall(
                call_id=f"model-{len(self.requests)}",
                participant="support_agent",
                requested_model=request["model"],
                returned_model=request["model"],
                finish_reason=("tool_calls" if message.get("tool_calls") else "stop"),
                temperature=request["temperature"],
                requested_max_tokens=request["max_tokens"],
                requested_enable_thinking=request.get("enable_thinking"),
                seed=request["seed"],
                input_messages=request["messages"],
                output_message=message,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    observed_cost_usd=0.0001,
                ),
            ),
        )


class _FakeMCP:
    instances: list["_FakeMCP"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.session_id = "fake-session"
        self.started = False
        self.closed = False
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def list_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "fixture",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self.kwargs["allowed_tool_names"]
        ]

    def validate_tool_arguments(
        self,
        _name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return arguments

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_request_id: Any = None,
    ) -> dict[str, Any]:
        if on_request_id is not None:
            on_request_id(1)
        self.executed.append((name, arguments))
        return {"status": "ok", "user_id": "retail_user_0001"}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _FakeMCP.instances.clear()


def _runtime(
    responses: list[dict[str, Any]],
    *,
    ttl: int = 1_800,
    jev_config: JevConfig | None = None,
    jev_client: Any = None,
    telemetry_factory: Any = None,
) -> ChatRuntime:
    client = _ScriptedClient(responses)
    return ChatRuntime(
        package_root=Path(__file__).resolve().parents[1],
        idle_ttl_seconds=ttl,
        inference_factory=lambda _route, _cap, _project: client,
        mcp_factory=_FakeMCP,
        telemetry_factory=telemetry_factory or (lambda _project: None),
        jev_config=jev_config,
        jev_client=jev_client,
    )


def test_config_create_and_first_message_route_without_exposing_credentials() -> None:
    runtime = _runtime(
        [
            _tool_call(
                "read-1",
                "find_user_id_by_email",
                {"email": "customer-0001@example.invalid"},
            ),
            {"role": "assistant", "content": "I found your account."},
        ]
    )
    try:
        config = runtime.config()
        assert config["credentials"] == {
            "wandb": True,
            "anthropic": False,
            "values_exposed_to_browser": False,
        }
        assert "test-key" not in json.dumps(config)
        created = runtime.create()
        assert created["profile"] is None
        assert created["customer"] == {
            "name": "Avery Arbor",
            "email": "customer-0001@example.invalid",
        }

        response = runtime.send(
            created["id"],
            "My email is customer-0001@example.invalid. "
            "I need to return a damaged item.",
        )

        assert response["profile"]["id"] == "retail-returns-resolution"
        assert response["assistant"] == "I found your account."
        assert response["tool_events"][0]["name"] == "find_user_id_by_email"
        assert response["tool_events"][0]["result"] == {"status": "ok"}
        assert response["usage"]["model_calls"] == 2
        assert _FakeMCP.instances[0].kwargs["authenticated_user_id"] == (
            "retail_user_0001"
        )
        assert response["telemetry"]["project"] == (
            "wandb/agent-lens-retail-support-live-v10"
        )
        assert response["telemetry"]["url"] == (
            "https://forge.coreweave.com/agent-lens/"
            "wandb/agent-lens-retail-support-live-v10"
        )
    finally:
        runtime.close()


def test_write_is_prepared_then_exact_confirmation_allows_execution() -> None:
    arguments = {"order_id": "#R0000368", "reason": "ordered by mistake"}
    runtime = _runtime(
        [
            _tool_call("write-1", "cancel_pending_order", arguments),
            {
                "role": "assistant",
                "content": (
                    "I can cancel order #R0000368 because it was ordered by "
                    "mistake. Do you want me to proceed?"
                ),
            },
            _tool_call("write-2", "cancel_pending_order", arguments),
            {"role": "assistant", "content": "Order #R0000368 is cancelled."},
        ]
    )
    try:
        created = runtime.create()
        prepared = runtime.send(
            created["id"],
            "Cancel order #R0000368; I ordered it by mistake.",
        )
        mcp = _FakeMCP.instances[0]
        assert prepared["tool_events"][0]["status"] == "prepared"
        assert mcp.executed == []

        completed = runtime.send(created["id"], "yes")
        assert completed["tool_events"][0]["status"] == "executed"
        assert mcp.executed == [("cancel_pending_order", arguments)]
    finally:
        runtime.close()


def test_jev_route_cannot_bypass_mutation_confirmation() -> None:
    class FakeJev:
        def system_one(self, *_args: Any, **_kwargs: Any) -> Any:
            return type(
                "Response",
                (),
                {
                    "model": "jev-1.13.0",
                    "usage": type(
                        "Usage",
                        (),
                        {"input_tokens": 12, "output_tokens": 0},
                    )(),
                    "choices": {
                        "agent_profile": type(
                            "Choice",
                            (),
                            {
                                "choice": "generalist",
                                "confidence": 0.99,
                                "probabilities": {
                                    "generalist": 0.99,
                                    "order_management": 0.005,
                                    "returns_resolution": 0.005,
                                },
                            },
                        )()
                    },
                    "nouls": {
                        "redundant_confirmation": type(
                            "Noul",
                            (),
                            {"noul": 0.01},
                        )(),
                        "partial_completion": type(
                            "Noul",
                            (),
                            {"noul": 0.01},
                        )(),
                        "unsupported_definitive_answer": type(
                            "Noul",
                            (),
                            {"noul": 0.01},
                        )(),
                    },
                },
            )()

        def close(self) -> None:
            raise AssertionError("injected Jev client must not be owned by runtime")

    arguments = {"order_id": "#R0000368", "reason": "ordered by mistake"}
    runtime = _runtime(
        [
            _tool_call("write-jev", "cancel_pending_order", arguments),
            {
                "role": "assistant",
                "content": "I can prepare that cancellation. Please confirm.",
            },
        ],
        jev_config=JevConfig(mode="route"),
        jev_client=FakeJev(),
    )
    try:
        created = runtime.create()
        response = runtime.send(
            created["id"],
            "Cancel order #R0000368; I ordered it by mistake.",
        )

        assert response["profile"]["id"] == "retail-support-generalist"
        assert response["jev"]["routing"]["source"] == "jev"
        assert response["jev"]["latest_assessment"]["source"] == "jev"
        assert response["tool_events"][0]["status"] == "prepared"
        assert _FakeMCP.instances[0].executed == []
    finally:
        runtime.close()


def test_runtime_keeps_jev_route_and_assessment_inside_the_customer_turn() -> None:
    assessment_states: list[dict[str, Any]] = []

    class FakeJev:
        def system_one(self, *_args: Any, **kwargs: Any) -> Any:
            questions = kwargs["questions"]
            if "agent_profile" in questions:
                return SimpleNamespace(
                    model="jev-1.13.0",
                    usage=SimpleNamespace(input_tokens=12, output_tokens=0),
                    choices={
                        "agent_profile": SimpleNamespace(
                            # Shadow mode must retain the deterministic returns
                            # route even when Jev suggests the generalist.
                            choice="generalist",
                            confidence=0.96,
                            probabilities={
                                "generalist": 0.96,
                                "order_management": 0.02,
                                "returns_resolution": 0.02,
                            },
                        )
                    },
                )
            assessment_states.append(kwargs["state"])
            return SimpleNamespace(
                model="jev-1.13.0",
                usage=SimpleNamespace(input_tokens=20, output_tokens=0),
                nouls={
                    name: SimpleNamespace(noul=0.05)
                    for name in questions
                },
            )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = LiveTelemetrySession.from_provider(
        provider,
        agent_version="runtime-parentage-test",
    )
    runtime = _runtime(
        [{"role": "assistant", "content": "I can help with that return."}],
        jev_config=JevConfig(mode="shadow"),
        jev_client=FakeJev(),
        telemetry_factory=lambda _project: telemetry,
    )
    try:
        created = runtime.create()
        runtime.send(created["id"], "I need to return a damaged item.")
    finally:
        runtime.close()
        provider.shutdown()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name.startswith("invoke_agent "))
    decisions = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "evaluate"
    ]
    assert root.name == "invoke_agent retail-returns-resolution"
    assert root.attributes["gen_ai.agent.name"] == "retail-returns-resolution"
    assert {span.name for span in decisions} == {
        "evaluate routing decision with jev-1.13.0",
        "evaluate behavior check with jev-1.13.0",
    }
    assert all(span.parent.span_id == root.context.span_id for span in decisions)
    assert all(span.context.trace_id == root.context.trace_id for span in decisions)
    assert assessment_states == [
        {
            "current_turn": {
                "user_message": "I need to return a damaged item.",
                "assistant_message": "I can help with that return.",
                "tool_events": [],
            },
            "recent_visible_messages": [
                {
                    "role": "assistant",
                    "content": "Hi! How can I help with your order today?",
                }
            ],
        }
    ]


def test_delete_closes_isolated_mcp_and_removes_session() -> None:
    runtime = _runtime([{"role": "assistant", "content": "I can help with that."}])
    created = runtime.create()
    runtime.send(created["id"], "I need help with a return.")
    mcp = _FakeMCP.instances[0]
    assert runtime.delete(created["id"]) == {"deleted": True}
    assert mcp.closed is True
    with pytest.raises(SessionNotFound):
        runtime.get(created["id"])
    runtime.close()


def test_fastapi_contract_uses_sync_worker_endpoints() -> None:
    runtime = _runtime([{"role": "assistant", "content": "I can help with that."}])
    app = create_app(runtime, static_dir=Path("/does/not/exist"))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        created = client.post("/api/conversations", json={}).json()
        response = client.post(
            f"/api/conversations/{created['id']}/messages",
            json={"content": "I need help with a return."},
        )
        assert response.status_code == 200
        assert response.json()["assistant"] == "I can help with that."
        assert client.get(f"/api/conversations/{created['id']}").status_code == 200
        assert client.delete(f"/api/conversations/{created['id']}").json() == {
            "deleted": True
        }
