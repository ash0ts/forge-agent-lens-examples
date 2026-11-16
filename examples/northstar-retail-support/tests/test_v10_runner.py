from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from retail_support_study_v1.canonical import digest_value
from retail_support_study_v1.inference import Completion
from retail_support_study_v1.runner import (
    _coordinate_inference_clients,
    run_coordinate,
)
from retail_support_study_v1.runtime_snapshot import RuntimeSnapshot
from retail_support_study_v1.schemas import Coordinate, ModelCall, Usage
from retail_support_study_v1.traffic_config import (
    ALL_TOOL_NAMES,
    RETURNS_AGENT_ID,
    build_agent_profiles,
    build_behavior_profile,
    build_model_routes,
)


class _UserClient:
    provider_name = "wandb"
    server_address = "api.inference.wandb.ai"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> Completion:
        assert request["participant"] == "user_simulator"
        self.calls.append(request)
        content = (
            "I need to return the damaged lamp from my delivered order."
            if len(self.calls) == 1
            else "###STOP###"
        )
        message = {"role": "assistant", "content": content}
        return Completion(
            message=message,
            record=ModelCall(
                call_id=f"user-{len(self.calls)}",
                participant="user_simulator",
                requested_model=request["model"],
                returned_model=request["model"],
                finish_reason="stop",
                temperature=request["temperature"],
                requested_max_tokens=request["max_tokens"],
                requested_enable_thinking=request.get("enable_thinking"),
                seed=request["seed"],
                input_messages=request["messages"],
                output_message=message,
                usage=Usage(),
            ),
        )


class _SupportClient:
    provider_name = "anthropic"
    server_address = "api.anthropic.com"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> Completion:
        assert request["participant"] == "support_agent"
        self.calls.append(request)
        message = {
            "role": "assistant",
            "content": "I can help inspect the delivered item and return options.",
        }
        return Completion(
            message=message,
            record=ModelCall(
                call_id="support-1",
                participant="support_agent",
                requested_model=request["model"],
                returned_model=request["model"],
                finish_reason="stop",
                temperature=request["temperature"],
                requested_max_tokens=request["max_tokens"],
                seed=request["seed"],
                input_messages=request["messages"],
                output_message=message,
                usage=Usage(),
            ),
        )


class _SpecialistMCP:
    session_id = "v10-specialist-session"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> "_SpecialistMCP":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

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
        _name: str,
        _arguments: dict[str, Any],
        *,
        on_request_id: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        if on_request_id is not None:
            on_request_id(1)
        return {"status": "ok"}


class _TelemetryTurn:
    def complete_support_call(
        self,
        operation: Callable[[], Any],
        _records: Any,
        _request: Any,
    ) -> Any:
        return operation()

    def execute_mcp(self, operation: Callable[..., Any], **_kwargs: Any) -> Any:
        return operation(lambda _request_id: None)

    def finish(self, _output: Any, _tools: Any) -> None:
        return None

    def fail(self, _exc: BaseException) -> None:
        return None


class _RouteRecordingTelemetry:
    def __init__(self) -> None:
        self.routes: list[dict[str, Any]] = []

    def start_turn(self, *_args: Any, **route: Any) -> _TelemetryTurn:
        self.routes.append(route)
        return _TelemetryTurn()


def _runtime(tmp_path: Path) -> RuntimeSnapshot:
    prompts = {
        "support-agent.md": "Help the customer using retail policy and tools.",
        "ordinary-user.md": "Speak naturally as the customer.",
        "concise-user.md": "Speak concisely as the customer.",
        "skeptical-user.md": "Ask for evidence when useful.",
    }
    instructions_sha256 = digest_value(prompts["support-agent.md"])
    profiles = build_agent_profiles(
        instructions_sha256=instructions_sha256,
        manifest_tool_names=ALL_TOOL_NAMES,
    )
    routes = build_model_routes()
    return RuntimeSnapshot(
        root=tmp_path,
        preview={
            "preview_sha256": "0" * 64,
            "traffic": {
                "agent_profiles": [
                    profile.model_dump(mode="json") for profile in profiles
                ],
                "model_routes": [route.model_dump(mode="json") for route in routes],
                "behavior_profile": build_behavior_profile().model_dump(mode="json"),
            },
        },
        manifest={},
        tasks=(),
        policy="Follow the retail policy.",
        user_guidelines="Stay in character and reveal information progressively.",
        prompts=prompts,
        pricing_lock={},
    )


def test_v10_routes_after_first_customer_message_and_scopes_mcp(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    selected_route = next(
        route
        for route in build_model_routes()
        if route.route_id == "anthropic-claude-sonnet-5"
    )
    coordinate = Coordinate(
        coordinate_id="v10-coordinate",
        logical_task_id="fixture-task",
        profile="ordinary",
        trial_index=0,
        model_route_id=selected_route.route_id,
        seed=17,
        cohort="canary",
    )
    task = {
        "id": "fixture-task",
        "authenticated_user_id": "retail_user_0001",
        "user_scenario": {"instructions": "Ask for help with a damaged lamp."},
    }
    created: list[_SpecialistMCP] = []

    def factory(**kwargs: Any) -> _SpecialistMCP:
        instance = _SpecialistMCP(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        factory,
    )
    user = _UserClient()
    support = _SupportClient()
    telemetry = _RouteRecordingTelemetry()

    bundle = run_coordinate(
        coordinate,
        user_client=user,  # type: ignore[arg-type]
        support_client=support,  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": task},
        runtime=runtime,
        live_telemetry=telemetry,  # type: ignore[arg-type]
    )

    assert bundle.status == "complete", bundle.unavailable_reason
    assert bundle.requested_models == {
        "support_agent": selected_route.model,
        "user_simulator": "deepseek-ai/DeepSeek-V4-Flash-0731",
    }
    assert len(user.calls) == 2
    assert len(support.calls) == 1
    assert support.calls[0]["model"] == selected_route.model
    assert support.calls[0]["temperature"] is None
    assert support.calls[0]["enable_thinking"] is False
    assert RETURNS_AGENT_ID in support.calls[0]["messages"][0]["content"]
    assert len(created) == 1
    allowed = set(created[0].kwargs["allowed_tool_names"])
    assert "return_delivered_order_items" in allowed
    assert "exchange_delivered_order_items" in allowed
    assert "cancel_pending_order" not in allowed
    assert created[0].kwargs["conversation_id"] == bundle.conversation_id
    assert created[0].kwargs["behavior_profile"]["profile_id"] == (
        "production-shaped-retail-tools-v1"
    )
    assert telemetry.routes == [
        {
            "agent_name": RETURNS_AGENT_ID,
            "requested_model": selected_route.model,
            "requested_temperature": selected_route.temperature,
            "provider_name": "anthropic",
            "server_address": "api.anthropic.com",
            "server_port": 443,
        }
    ]


def test_split_provider_clients_share_one_coordinate_budget(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    runtime = _runtime(tmp_path)
    coordinate = Coordinate(
        coordinate_id="anthropic-coordinate",
        logical_task_id="fixture-task",
        profile="ordinary",
        trial_index=0,
        model_route_id="anthropic-claude-sonnet-5",
        seed=17,
        cohort="canary",
    )

    clients = _coordinate_inference_clients(
        coordinate,
        runtime=runtime,
        cost_cap_usd=5.0,
    )

    assert clients.user.provider_name == "wandb"
    assert clients.support.provider_name == "anthropic"
    assert clients.user.budget is clients.budget
    assert clients.support.budget is clients.budget
