from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from retail_support_study_v1.canonical import digest_value, load_json
from retail_support_study_v1.constants import (
    PACKAGE_ROOT,
    RUNS_DIR,
    SUPPORT_MAX_OUTPUT_TOKENS,
    USER_MAX_OUTPUT_TOKENS,
)
from retail_support_study_v1.inference import Completion, InferenceCallError
from retail_support_study_v1.runner import (
    _task_instructions,
    _usage_total,
    _user_system_prompt,
    run_coordinate,
)
from retail_support_study_v1.runtime_snapshot import RuntimeSnapshot
from retail_support_study_v1.schemas import (
    ConversationBundle,
    Coordinate,
    ModelCall,
    RunLedger,
    Usage,
)
from retail_support_study_v1.simulator_contracts import (
    CONFIRMATION_CONDITION_INSTRUCTION,
    SimulatorChoreography,
)
from retail_support_study_v1.telemetry import build_public_payload


def test_usage_total_canonicalizes_floating_point_cost_sums() -> None:
    def call(call_id: str, observed: float, provider: float) -> ModelCall:
        return ModelCall(
            call_id=call_id,
            participant="support_agent",
            requested_model="model",
            returned_model="model",
            temperature=0.0,
            requested_max_tokens=1,
            input_messages=[],
            output_message={"role": "assistant", "content": "done"},
            usage=Usage(
                provider_reported_cost_usd=provider,
                observed_cost_usd=observed,
            ),
        )

    total = _usage_total(
        [
            call("first", 0.1, 0.1),
            call("second", 0.2, 0.2),
        ]
    )

    assert total.observed_cost_usd == 0.3
    assert total.provider_reported_cost_usd == 0.3


class _FakeMCP:
    calls: list[tuple[str, dict[str, Any]]] = []
    session_id = "fixture-stdio-session"

    def __enter__(self) -> "_FakeMCP":
        type(self).calls = []
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def list_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"tool-{index}",
                    "description": "fixture",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for index in range(16)
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_request_id: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        if on_request_id is not None:
            on_request_id(len(type(self).calls) + 101)
        type(self).calls.append((name, arguments))
        return {"status": "cancelled"}

    def validate_tool_arguments(
        self, _name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return arguments


class _NormalizingMCP(_FakeMCP):
    calls: list[tuple[str, dict[str, Any]]] = []

    def validate_tool_arguments(
        self, _name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {**arguments, "reason": "ordered by mistake"}


class _StopBeforeSupportClient:
    def complete(self, **kwargs: Any) -> Completion:
        assert kwargs["participant"] == "user_simulator"
        output = {"role": "assistant", "content": "###STOP###"}
        return Completion(
            message=output,
            record=ModelCall(
                call_id="simulator-stop",
                participant="user_simulator",
                requested_model=kwargs["model"],
                returned_model=kwargs["model"],
                temperature=kwargs["temperature"],
                requested_max_tokens=kwargs["max_tokens"],
                seed=kwargs["seed"],
                input_messages=kwargs["messages"],
                output_message=output,
                usage=Usage(),
            ),
        )


class _UnavailableClient:
    def complete(self, **_kwargs: Any) -> Completion:
        raise RuntimeError("provider transport remained unavailable")


class _AmbiguousResponseClient:
    def complete(self, **kwargs: Any) -> Completion:
        output = {"role": "assistant", "content": "Hello from the simulator."}
        return Completion(
            message=output,
            record=ModelCall(
                call_id="ambiguous-call",
                participant=kwargs["participant"],
                requested_model=kwargs["model"],
                returned_model=kwargs["model"],
                finish_reason="stop",
                ambiguous_cost_reserve_usd=0.25,
                cost_uncertainty_reasons=["successful response omitted usage"],
                temperature=kwargs["temperature"],
                requested_max_tokens=kwargs["max_tokens"],
                seed=kwargs["seed"],
                input_messages=kwargs["messages"],
                output_message=output,
                usage=Usage(),
            ),
        )


class _RejectedResponseClient:
    def complete(self, **kwargs: Any) -> Completion:
        record = ModelCall(
            call_id="rejected-call",
            participant=kwargs["participant"],
            requested_model=kwargs["model"],
            returned_model=f"{kwargs['model']}-alias",
            finish_reason="stop",
            response_validation_error="returned model identity mismatch",
            temperature=kwargs["temperature"],
            requested_max_tokens=kwargs["max_tokens"],
            requested_enable_thinking=kwargs.get("enable_thinking"),
            seed=kwargs["seed"],
            input_messages=kwargs["messages"],
            output_message={"role": "assistant", "content": "paid response"},
            usage=Usage(observed_cost_usd=0.01),
        )
        raise InferenceCallError("returned model identity mismatch", record=record)


class _RecordingLiveTurn:
    def __init__(self) -> None:
        self.request_ids: list[int] = []
        self.finished = False

    def complete_support_call(
        self,
        operation: Callable[[], Any],
        _records: Any,
        _request: Any,
    ) -> Any:
        return operation()

    def execute_mcp(
        self,
        operation: Callable[[Callable[[int], None]], Any],
        **_kwargs: Any,
    ) -> Any:
        return operation(self.request_ids.append)

    def finish(self, _output: Any, _tools: Any) -> None:
        self.finished = True

    def fail(self, _exc: BaseException) -> None:
        return None


class _RecordingLiveTelemetry:
    def __init__(self) -> None:
        self.inputs: list[list[dict[str, Any]]] = []
        self.turns: list[_RecordingLiveTurn] = []
        self.routes: list[dict[str, Any]] = []

    def start_turn(
        self,
        _conversation_id: str,
        input_messages: list[dict[str, Any]],
        _tools_by_id: Any,
        **route: Any,
    ) -> _RecordingLiveTurn:
        self.inputs.append([dict(message) for message in input_messages])
        self.routes.append(route)
        turn = _RecordingLiveTurn()
        self.turns.append(turn)
        return turn


def _model_completion(
    kwargs: dict[str, Any],
    *,
    call_id: str,
    message: dict[str, Any],
) -> Completion:
    return Completion(
        message=message,
        record=ModelCall(
            call_id=call_id,
            participant=kwargs["participant"],
            requested_model=kwargs["model"],
            returned_model=kwargs["model"],
            temperature=kwargs["temperature"],
            requested_max_tokens=kwargs["max_tokens"],
            requested_enable_thinking=kwargs.get("enable_thinking"),
            seed=kwargs["seed"],
            input_messages=kwargs["messages"],
            output_message=message,
            usage=Usage(),
        ),
    )


class _UnconfirmedWriteClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = (
                "Please cancel order #R0000001."
                if self.user_calls == 1
                else "###STOP###"
            )
            return _model_completion(
                kwargs,
                call_id=f"user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls == 1:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": (
                                '{"order_id":"#R0000001","reason":"no longer needed"}'
                            ),
                        },
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": "I need your confirmation before I can cancel the order.",
            }
        return _model_completion(
            kwargs,
            call_id=f"agent-{self.agent_calls}",
            message=message,
        )


class _ConfirmedWriteClient:
    def __init__(self, *, proposal_reason: str = "no longer needed") -> None:
        self.user_calls = 0
        self.agent_calls = 0
        self.proposal_reason = proposal_reason

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = {
                1: "Please cancel order #R0000001.",
                2: "Yes, please.",
                3: "###STOP###",
            }[self.user_calls]
            return _model_completion(
                kwargs,
                call_id=f"confirmed-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls in {1, 3}:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"confirmed-tool-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": (
                                '{"order_id":"#R0000001",'
                                '"reason":"no longer needed"}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 2:
            message = {
                "role": "assistant",
                "content": (
                    "I can cancel order #R0000001 because it was "
                    f"{self.proposal_reason}. Would you like me to proceed?"
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": "Order #R0000001 is now cancelled.",
            }
        return _model_completion(
            kwargs,
            call_id=f"confirmed-agent-{self.agent_calls}",
            message=message,
        )


class _NaturalReturnConfirmationClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = {
                1: "I'd like to return the jacket from order #R0000042.",
                2: "Yes, please.",
                3: "###STOP###",
            }[self.user_calls]
            return _model_completion(
                kwargs,
                call_id=f"return-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls in {1, 3}:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"return-tool-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "return_delivered_order_items",
                            "arguments": (
                                '{"order_id":"#R0000042",'
                                '"item_ids":["item_00420"],'
                                '"payment_method_id":"paypal_000042"}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 2:
            message = {
                "role": "assistant",
                "content": (
                    "I can start the return for the jacket from order "
                    "#R0000042 and send the refund to your PayPal account. "
                    "You will receive an email with return instructions. "
                    "Would you like me to proceed?"
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": "The return for order #R0000042 is now requested.",
            }
        return _model_completion(
            kwargs,
            call_id=f"return-agent-{self.agent_calls}",
            message=message,
        )


class _MixedReadContentToolClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = (
                "Where is order #R0000001?"
                if self.user_calls == 1
                else "###STOP###"
            )
            return _model_completion(
                kwargs,
                call_id=f"mixed-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )
        self.agent_calls += 1
        if self.agent_calls > 1:
            return _model_completion(
                kwargs,
                call_id="mixed-read-agent-2",
                message={
                    "role": "assistant",
                    "content": "I checked the order record; it is still processing.",
                },
            )
        return _model_completion(
            kwargs,
            call_id="mixed-read-agent-1",
            message={
                "role": "assistant",
                "content": "I will check that order now.",
                "tool_calls": [
                    {
                        "id": "mixed-read-tool-1",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id":"#R0000001"}',
                        },
                    }
                ],
            },
        )


class _MixedWriteContentToolClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = (
                "Please cancel order #R0000001."
                if self.user_calls == 1
                else "###STOP###"
            )
            return _model_completion(
                kwargs,
                call_id=f"mixed-write-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )
        self.agent_calls += 1
        if self.agent_calls > 1:
            return _model_completion(
                kwargs,
                call_id="mixed-write-agent-2",
                message={
                    "role": "assistant",
                    "content": (
                        "I can cancel order #R0000001 because it is no longer "
                        "needed. Would you like me to proceed?"
                    ),
                },
            )
        return _model_completion(
            kwargs,
            call_id="mixed-write-agent-1",
            message={
                "role": "assistant",
                "content": "I will prepare that cancellation.",
                "tool_calls": [
                    {
                        "id": "mixed-write-tool-1",
                        "type": "function",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": (
                                '{"order_id":"#R0000001",'
                                '"reason":"no longer needed"}'
                            ),
                        },
                    }
                ],
            },
        )


class _StopWithVisibleTextClient:
    def __init__(self) -> None:
        self.user_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = (
                "Where is my order?"
                if self.user_calls == 1
                else "Thanks, that answers it. ###STOP###"
            )
            return _model_completion(
                kwargs,
                call_id=f"user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )
        return _model_completion(
            kwargs,
            call_id="agent-1",
            message={"role": "assistant", "content": "It arrives tomorrow."},
        )


class _RecoveredAmbiguousResponseClient(_StopWithVisibleTextClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        completion = super().complete(**kwargs)
        self.calls += 1
        if self.calls == 1:
            completion.record.finish_reason = "stop"
            completion.record.ambiguous_cost_reserve_usd = 0.25
            completion.record.cost_uncertainty_reasons = [
                "retryable transport outcome was ambiguous: attempt=1, "
                "error=InternalServerError"
            ]
        return completion


class _Task21ConfirmationClient:
    def __init__(self, *, repair_text: str) -> None:
        self.user_calls = 0
        self.agent_calls = 0
        self.repair_text = repair_text

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = {
                1: "I need to change the running shoes in my pending order.",
                2: "Yes, please.",
                3: self.repair_text,
                4: "Yes.",
                5: "###STOP###",
            }[self.user_calls]
            return _model_completion(
                kwargs,
                call_id=f"task-21-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls == 1:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"task-21-tool-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": (
                                '{"order_id":"#R0000001",'
                                '"reason":"no longer needed"}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 2:
            message = {
                "role": "assistant",
                "content": (
                    "I can cancel order #R0000001 because it is no longer "
                    "needed. Would you like me to proceed?"
                ),
            }
        elif self.agent_calls in {3, 5}:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"task-21-tool-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "modify_pending_order_items",
                            "arguments": (
                                '{"order_id":"#R0000001",'
                                '"item_ids":["item_shoes","item_keyboard"],'
                                '"new_item_ids":['
                                '"variant_shoes","variant_keyboard"]}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 4:
            message = {
                "role": "assistant",
                "content": (
                    "I can change the running shoes and also change the keyboard "
                    "to the no-backlight, 80% model with linear switches in order "
                    "#R0000001. Would you like me to proceed?"
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": "The requested change is complete.",
            }
        return _model_completion(
            kwargs,
            call_id=f"task-21-agent-{self.agent_calls}",
            message=message,
        )


class _Task21CorrectionBeforePreparationClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            content = {
                1: "I need to change the running shoes in my pending order.",
                2: (
                    "Before I confirm, I'd also like to switch the mechanical "
                    "keyboard to the no-backlight, 80 percent, linear-switch model."
                ),
                3: "Yes.",
                4: "###STOP###",
            }[self.user_calls]
            return _model_completion(
                kwargs,
                call_id=f"task-21-early-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls == 1:
            message = {
                "role": "assistant",
                "content": (
                    "You mentioned a couple of changes. Is there anything else "
                    "in this order that you want to change?"
                ),
            }
        elif self.agent_calls in {2, 4}:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"task-21-early-tool-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "modify_pending_order_items",
                            "arguments": (
                                '{"order_id":"#R0000001",'
                                '"item_ids":["item_shoes","item_keyboard"],'
                                '"new_item_ids":['
                                '"variant_shoes","variant_keyboard"]}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 3:
            message = {
                "role": "assistant",
                "content": (
                    "The full modification for order #R0000001 includes the "
                    "running shoes and the no-backlight, 80% mechanical keyboard "
                    "with linear switches. Shall I proceed?"
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": "The requested change is complete.",
            }
        return _model_completion(
            kwargs,
            call_id=f"task-21-early-agent-{self.agent_calls}",
            message=message,
        )


class _Task84LatentConfirmationChangeClient:
    def __init__(self) -> None:
        self.user_calls = 0
        self.agent_calls = 0
        self.user_system_prompts: list[str] = []

    def complete(self, **kwargs: Any) -> Completion:
        if kwargs["participant"] == "user_simulator":
            self.user_calls += 1
            self.user_system_prompts.append(
                str(kwargs["messages"][0]["content"])
            )
            content = {
                1: (
                    "I received two tablets and only need one. I want to return "
                    "the cheaper one and have the refund sent to my credit card."
                ),
                2: (
                    "Actually, return the more expensive tablet instead and put "
                    "the refund on my gift card."
                ),
                3: (
                    "The cheaper tablet, please, with the refund going back to "
                    "my credit card."
                ),
                4: (
                    "Actually, return the more expensive tablet instead and put "
                    "the refund on my gift card."
                ),
                5: "Yes.",
                6: "###STOP###",
            }[self.user_calls]
            return _model_completion(
                kwargs,
                call_id=f"task-84-user-{self.user_calls}",
                message={"role": "assistant", "content": content},
            )

        self.agent_calls += 1
        if self.agent_calls == 1:
            message = {
                "role": "assistant",
                "content": "Which tablet would you like to return?",
            }
        elif self.agent_calls == 2:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "task-84-cheaper-preparation",
                        "type": "function",
                        "function": {
                            "name": "return_delivered_order_items",
                            "arguments": (
                                '{"order_id":"#R0000223",'
                                '"item_ids":["item_00499"],'
                                '"payment_method_id":"credit_card_000017"}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 3:
            message = {
                "role": "assistant",
                "content": (
                    "I am ready to return the cheaper silver tablet from order "
                    "#R0000223 and refund your credit card. Shall I proceed?"
                ),
            }
        elif self.agent_calls in {4, 6}:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"task-84-gift-card-{self.agent_calls}",
                        "type": "function",
                        "function": {
                            "name": "return_delivered_order_items",
                            "arguments": (
                                '{"order_id":"#R0000950",'
                                '"item_ids":["item_00326"],'
                                '"payment_method_id":"gift_card_000401"}'
                            ),
                        },
                    }
                ],
            }
        elif self.agent_calls == 5:
            message = {
                "role": "assistant",
                "content": (
                    "The updated return is for the more expensive gold tablet "
                    "from order #R0000950, with the refund going to your gift "
                    "card. Shall I submit this return?"
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": "The gold tablet return is now requested.",
            }
        return _model_completion(
            kwargs,
            call_id=f"task-84-agent-{self.agent_calls}",
            message=message,
        )


def _coordinate() -> Coordinate:
    return Coordinate(
        coordinate_id="b046f333-ce7e-5fcf-a067-dd159dcd64ad",
        logical_task_id="fixture-task",
        profile="ordinary",
        seed=17,
        cohort="canary",
    )


def _task() -> dict[str, Any]:
    return {
        "id": "fixture-task",
        "authenticated_user_id": "retail_user_0001",
        "user_scenario": {"instructions": "Ask for help with an order."},
    }


def _reviewed_task(task_id: str) -> dict[str, Any]:
    rows = load_json(PACKAGE_ROOT / "prepared/tasks.runtime.json")
    return next(row for row in rows if str(row["id"]) == task_id)


def _task21() -> dict[str, Any]:
    return _reviewed_task("21")


def _task21_coordinate() -> Coordinate:
    return _coordinate().model_copy(update={"logical_task_id": "21"})


def _task84() -> dict[str, Any]:
    return _reviewed_task("84")


def _task84_coordinate() -> Coordinate:
    return _coordinate().model_copy(update={"logical_task_id": "84"})


def _runtime() -> RuntimeSnapshot:
    prompts = {
        name: (PACKAGE_ROOT / "prompts" / name).read_text(encoding="utf-8").strip()
        for name in (
            "support-agent.md",
            "ordinary-user.md",
            "concise-user.md",
            "skeptical-user.md",
        )
    }
    return RuntimeSnapshot(
        root=PACKAGE_ROOT,
        preview={"preview_sha256": "0" * 64},
        manifest={},
        tasks=(_task(),),
        policy=(PACKAGE_ROOT / "prepared/policy.md").read_text(encoding="utf-8"),
        user_guidelines=(PACKAGE_ROOT / "prepared/user-guidelines.md").read_text(
            encoding="utf-8"
        ),
        prompts=prompts,
        pricing_lock={},
    )


def _ledger(bundle: Any) -> RunLedger:
    coordinate_id = bundle.coordinate.coordinate_id
    observed = float(bundle.usage.observed_cost_usd or 0.0)
    ledger = RunLedger(
        study_id=bundle.study_id,
        cohort="canary",
        preview_sha256="1" * 64,
        planned_coordinate_ids=[coordinate_id],
        completed_coordinate_ids=[coordinate_id],
        unavailable={},
        observed_cost_usd=observed,
        bundle_digests={coordinate_id: bundle.bundle_sha256},
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    return ledger


def test_stop_before_any_support_model_answer_is_unavailable(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_StopBeforeSupportClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert bundle.terminal_response is False
    assert bundle.unavailable_reason is not None
    assert "support" in bundle.unavailable_reason.lower()


def test_live_agent_turn_roots_contain_only_the_current_user_message(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    telemetry = _RecordingLiveTelemetry()

    bundle = run_coordinate(
        _coordinate(),
        client=_ConfirmedWriteClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
        live_telemetry=telemetry,  # type: ignore[arg-type]
    )

    assert bundle.status == "complete", bundle.unavailable_reason
    assert telemetry.inputs == [
        [{"role": "user", "content": "Please cancel order #R0000001."}],
        [{"role": "user", "content": "Yes, please."}],
    ]
    assert all(turn.finished for turn in telemetry.turns)
    assert telemetry.turns[1].request_ids == [101]
    assert all(
        route == {
            "agent_name": "retail-support-agent",
            "requested_model": "zai-org/GLM-5.3-Flash",
            "requested_temperature": 0.0,
            "provider_name": "wandb",
            "server_address": "api.inference.wandb.ai",
            "server_port": 443,
        }
        for route in telemetry.routes
    )


def test_stop_control_text_does_not_leave_an_unanswered_public_user_turn(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_StopWithVisibleTextClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete", bundle.unavailable_reason
    assert bundle.messages[-1] == {
        "role": "assistant",
        "content": "It arrives tomorrow.",
    }
    assert all("###STOP###" not in str(message) for message in bundle.messages)
    assert all(
        "Thanks, that answers it." not in str(message) for message in bundle.messages
    )


def test_runtime_account_boundary_is_not_added_to_either_model_prompt(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def mcp_factory(**kwargs: Any) -> _FakeMCP:
        captured.update(kwargs)
        return _FakeMCP()

    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        mcp_factory,
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_StopWithVisibleTextClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert captured["authenticated_user_id"] == "retail_user_0001"
    assert all(
        "retail_user_0001" not in str(call.input_messages)
        for call in bundle.model_calls
    )
    assert "retail_user_0001" not in str(bundle.messages)


def test_provider_failure_becomes_a_frozen_unavailable_coordinate(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_UnavailableClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert bundle.terminal_response is False
    assert bundle.unavailable_reason == (
        "RuntimeError: provider transport remained unavailable"
    )
    assert bundle.model_calls == []
    assert bundle.bundle_sha256


def test_cost_uncertainty_stops_before_response_is_used_and_freezes_call(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )

    bundle = run_coordinate(
        _coordinate(),
        client=_AmbiguousResponseClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert len(bundle.model_calls) == 1
    assert bundle.model_calls[0].ambiguous_cost_reserve_usd == 0.25
    assert "cost became uncertain" in str(bundle.unavailable_reason)
    assert bundle.messages == [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]


def test_relaxed_test_project_policy_uses_a_recovered_response(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    base = _runtime()
    runtime = RuntimeSnapshot(
        root=base.root,
        preview={
            **base.preview,
            "run_policy": {
                "frozen_attempt_failure_policy": (
                    "continue_when_bundle_is_frozen"
                ),
                "bounded_retry_cost_ambiguity": (
                    "track_upper_bound_and_continue"
                ),
            },
        },
        manifest=base.manifest,
        tasks=base.tasks,
        policy=base.policy,
        user_guidelines=base.user_guidelines,
        prompts=base.prompts,
        pricing_lock=base.pricing_lock,
    )

    bundle = run_coordinate(
        _coordinate(),
        client=_RecoveredAmbiguousResponseClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=runtime,
    )

    assert bundle.status == "complete"
    assert bundle.terminal_response is True
    assert sum(
        call.ambiguous_cost_reserve_usd for call in bundle.model_calls
    ) == 0.25
    assert bundle.messages[-1] == {
        "role": "assistant",
        "content": "It arrives tomorrow.",
    }


def test_relaxed_policy_still_rejects_missing_usage_ambiguity(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    base = _runtime()
    runtime = RuntimeSnapshot(
        root=base.root,
        preview={
            **base.preview,
            "run_policy": {
                "frozen_attempt_failure_policy": (
                    "continue_when_bundle_is_frozen"
                ),
                "bounded_retry_cost_ambiguity": (
                    "track_upper_bound_and_continue"
                ),
            },
        },
        manifest=base.manifest,
        tasks=base.tasks,
        policy=base.policy,
        user_guidelines=base.user_guidelines,
        prompts=base.prompts,
        pricing_lock=base.pricing_lock,
    )

    bundle = run_coordinate(
        _coordinate(),
        client=_AmbiguousResponseClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=runtime,
    )

    assert bundle.status == "unavailable"
    assert "outside the approved retry policy" in str(bundle.unavailable_reason)


def test_rejected_paid_response_is_frozen_before_coordinate_stops(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )

    bundle = run_coordinate(
        _coordinate(),
        client=_RejectedResponseClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert len(bundle.model_calls) == 1
    assert bundle.model_calls[0].returned_model.endswith("-alias")
    assert bundle.model_calls[0].usage.observed_cost_usd == 0.01
    assert bundle.model_calls[0].response_validation_error


def test_unconfirmed_write_is_not_executed_and_remains_visible(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_UnconfirmedWriteClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    assert bundle.mutation_confirmation_ok is None
    assert _FakeMCP.calls == []
    assert len(bundle.tool_calls) == 1
    assert bundle.tool_calls[0].error_type is None
    assert bundle.tool_calls[0].execution_disposition == "prepared"
    assert bundle.tool_calls[0].result["status"] == "action_prepared"
    assert len(bundle.mutation_confirmation_receipts) == 1
    assert bundle.mutation_confirmation_receipts[0].state == "invalidated"
    assert bundle.mutation_confirmation_receipts[0].invalidation_reason == (
        "proposal_mismatch"
    )


def test_exact_prepared_action_executes_once_after_new_confirmation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_ConfirmedWriteClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    assert bundle.mutation_confirmation_ok is True
    assert _FakeMCP.calls == [
        (
            "cancel_pending_order",
            {"order_id": "#R0000001", "reason": "no longer needed"},
        )
    ]
    assert [record.execution_disposition for record in bundle.tool_calls] == [
        "prepared",
        "executed",
    ]
    assert len(bundle.mutation_confirmation_receipts) == 1
    receipt = bundle.mutation_confirmation_receipts[0]
    assert receipt.state == "consumed"
    assert receipt.staged_tool_call_id == "confirmed-tool-1"
    assert receipt.consumed_tool_call_id == "confirmed-tool-3"
    assert receipt.staged_message_index == 2
    assert receipt.proposal_message_index == 4
    assert receipt.decision_message_index == 5
    assert receipt.consumed_message_index == 6


def test_natural_paypal_return_executes_only_after_later_standalone_assent(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )

    bundle = run_coordinate(
        _coordinate(),
        client=_NaturalReturnConfirmationClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    expected_arguments = {
        "order_id": "#R0000042",
        "item_ids": ["item_00420"],
        "payment_method_id": "paypal_000042",
    }
    assert bundle.status == "complete"
    assert [record.execution_disposition for record in bundle.tool_calls] == [
        "prepared",
        "executed",
    ]
    assert _FakeMCP.calls == [
        ("return_delivered_order_items", expected_arguments)
    ]
    assert bundle.mutation_confirmation_ok is True
    receipt = bundle.mutation_confirmation_receipts[0]
    assert receipt.state == "consumed"
    assert receipt.proposal_message_index < receipt.decision_message_index
    assert receipt.decision_message_index < receipt.consumed_message_index
    proposal = bundle.messages[receipt.proposal_message_index]["content"]
    assert "PayPal" in proposal
    assert "item_00420" not in proposal
    assert "paypal_000042" not in proposal


def test_schema_normalized_arguments_define_the_confirmed_execution(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _NormalizingMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_ConfirmedWriteClient(
            proposal_reason="ordered by mistake"
        ),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    expected = {
        "order_id": "#R0000001",
        "reason": "ordered by mistake",
    }
    assert bundle.status == "complete"
    assert [record.arguments for record in bundle.tool_calls] == [expected, expected]
    assert _NormalizingMCP.calls == [("cancel_pending_order", expected)]


def test_read_only_mixed_visible_text_and_tool_call_is_preserved_and_executed(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_MixedReadContentToolClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    assert _FakeMCP.calls == [
        ("get_order_details", {"order_id": "#R0000001"})
    ]
    mixed = bundle.messages[2]
    assert mixed == bundle.model_calls[1].output_message
    assert mixed["content"] == "I will check that order now."
    assert mixed["tool_calls"][0]["id"] == "mixed-read-tool-1"

    payload = build_public_payload(_ledger(bundle), [bundle])
    first_chat = payload["conversations"][0]["turns"][0]["children"][0]
    assert first_chat["output_messages"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "I will check that order now."},
                {
                    "type": "tool_call",
                    "id": "mixed-read-tool-1",
                    "name": "get_order_details",
                    "arguments": {"order_id": "#R0000001"},
                },
            ],
        }
    ]


def test_pending_mixed_write_cannot_be_abandoned_with_a_stop_token(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_MixedWriteContentToolClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert "scenario-conformance repair" in str(bundle.unavailable_reason)
    assert _FakeMCP.calls == []
    assert [record.execution_disposition for record in bundle.tool_calls] == [
        "prepared"
    ]
    assert bundle.mutation_confirmation_ok is None
    mixed = bundle.messages[2]
    assert mixed == bundle.model_calls[1].output_message
    assert mixed["content"] == "I will prepare that cancellation."
    assert bundle.simulator_conformance_receipts[0].event_id == (
        "pending-confirmation-stop"
    )


def test_runner_passes_frozen_output_limits_to_each_participant(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _coordinate(),
        client=_StopWithVisibleTextClient(),  # type: ignore[arg-type]
        tasks_by_id={"fixture-task": _task()},
        runtime=_runtime(),
    )

    assert {
        call.participant: call.requested_max_tokens for call in bundle.model_calls
    } == {
        "support_agent": SUPPORT_MAX_OUTPUT_TOKENS,
        "user_simulator": USER_MAX_OUTPUT_TOKENS,
    }
    assert {
        call.participant: call.requested_enable_thinking
        for call in bundle.model_calls
    } == {
        "support_agent": None,
        "user_simulator": False,
    }


def test_task21_premature_assent_is_repaired_before_it_reaches_transcript(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    client = _Task21ConfirmationClient(
        repair_text=(
            "Before I confirm, also change the keyboard to the no-backlight, "
            "80% model with linear switches."
        )
    )
    bundle = run_coordinate(
        _task21_coordinate(),
        client=client,  # type: ignore[arg-type]
        tasks_by_id={"21": _task21()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    assert [
        message["content"]
        for message in bundle.messages
        if message.get("role") == "user"
    ] == [
        "I need to change the running shoes in my pending order.",
        (
            "Before I confirm, also change the keyboard to the no-backlight, "
            "80% model with linear switches."
        ),
        "Yes.",
    ]
    rejected = [
        call
        for call in bundle.model_calls
        if call.transcript_disposition == "rejected_simulator_conformance"
    ]
    assert [call.output_message["content"] for call in rejected] == ["Yes, please."]
    assert len(bundle.simulator_conformance_receipts) == 1
    receipt = bundle.simulator_conformance_receipts[0]
    assert receipt.event_id == "task-21-confirmation-1"
    assert receipt.outcome == "repaired"
    assert _FakeMCP.calls == [
        (
            "modify_pending_order_items",
            {
                "order_id": "#R0000001",
                "item_ids": ["item_shoes", "item_keyboard"],
                "new_item_ids": ["variant_shoes", "variant_keyboard"],
            },
        )
    ]

    encoded = str(build_public_payload(_ledger(bundle), [bundle]))
    assert "Yes, please." not in encoded
    assert "task-21-confirmation-1" not in encoded
    assert "Your previous draft did not follow" not in encoded


def test_task21_correction_before_preparation_is_honored_by_final_confirmation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _task21_coordinate(),
        client=_Task21CorrectionBeforePreparationClient(),  # type: ignore[arg-type]
        tasks_by_id={"21": _task21()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    assert bundle.simulator_conformance_receipts == []
    assert [
        message["content"]
        for message in bundle.messages
        if message.get("role") == "user"
    ] == [
        "I need to change the running shoes in my pending order.",
        (
            "Before I confirm, I'd also like to switch the mechanical keyboard "
            "to the no-backlight, 80 percent, linear-switch model."
        ),
        "Yes.",
    ]
    assert _FakeMCP.calls == [
        (
            "modify_pending_order_items",
            {
                "order_id": "#R0000001",
                "item_ids": ["item_shoes", "item_keyboard"],
                "new_item_ids": ["variant_shoes", "variant_keyboard"],
            },
        )
    ]


def test_simulator_repair_evidence_is_bound_to_action_and_visible_message(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _task21_coordinate(),
        client=_Task21ConfirmationClient(  # type: ignore[arg-type]
            repair_text=(
                "Also change the keyboard to the no-backlight, 80% model with "
                "linear switches."
            )
        ),
        tasks_by_id={"21": _task21()},
        runtime=_runtime(),
    )
    assert bundle.status == "complete"

    wrong_action = bundle.model_dump(mode="json")
    wrong_action["simulator_conformance_receipts"][0][
        "staged_action_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="bind one staged action"):
        ConversationBundle.model_validate(wrong_action)

    missing_visible_repair = bundle.model_dump(mode="json")
    missing_visible_repair["simulator_conformance_receipts"][0][
        "accepted_message_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="transcript binding drift"):
        ConversationBundle.model_validate(missing_visible_repair)


def test_task21_second_nonconformant_draft_is_unavailable_without_mutation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    bundle = run_coordinate(
        _task21_coordinate(),
        client=_Task21ConfirmationClient(  # type: ignore[arg-type]
            repair_text="Yes, go ahead."
        ),
        tasks_by_id={"21": _task21()},
        runtime=_runtime(),
    )

    assert bundle.status == "unavailable"
    assert "scenario-conformance repair" in str(bundle.unavailable_reason)
    assert _FakeMCP.calls == []
    assert all(
        message.get("content") not in {"Yes, please.", "Yes, go ahead."}
        for message in bundle.messages
    )
    assert bundle.simulator_conformance_receipts[0].outcome == "unavailable"
    assert len(
        [
            call
            for call in bundle.model_calls
            if call.transcript_disposition == "rejected_simulator_conformance"
        ]
    ) == 2


def test_simulator_prompt_defers_confirmation_conditional_preferences() -> None:
    prompt = _user_system_prompt(_task84(), "skeptical", _runtime())
    preamble = prompt.split("<scenario>", 1)[0]

    assert CONFIRMATION_CONDITION_INSTRUCTION.strip() in preamble
    assert "questions that ask which item, option, or payment" in preamble
    assert "before that condition" in preamble
    assert all(
        private_detail not in preamble.casefold()
        for private_detail in (
            "tablet",
            "gift card",
            "more expensive",
            "parker dale",
        )
    )


@pytest.mark.parametrize(
    ("task_id", "required_initial_intent"),
    (
        ("5", "exchange the water bottle and the desk lamp"),
        ("6", "exchange the water bottle and the desk lamp"),
        ("7", "exchange the water bottle and the desk lamp"),
        ("8", "exchange the water bottle and the desk lamp"),
        ("9", "exchange the water bottle and the desk lamp"),
        ("18", "return the office chair because it came with some broken pieces"),
        ("21", "exchange your shoes to the black, synthetic, size 9"),
        ("24", "You want to cancel the grill."),
        ("71", "you say GC initially."),
        ("72", "you say gift card initially."),
        ("84", "return the less expensive one and refund to credit card"),
        ("91", "return them and refund to credit card"),
        ("92", "return them and refund to credit card"),
        ("98", "exchange your Bicycle for one with a larger frame size"),
    ),
)
def test_initial_prompt_preserves_precondition_and_defers_future_events(
    task_id: str,
    required_initial_intent: str,
) -> None:
    task = _reviewed_task(task_id)
    scenario = _task_instructions(task)
    choreography = SimulatorChoreography(task)

    assert required_initial_intent in scenario
    assert all(clause not in scenario for clause in choreography.source_clauses)
    assert ", but ." not in scenario
    assert ". ." not in scenario


def test_mixed_source_clauses_keep_only_the_preconfirmation_request() -> None:
    source_task24 = _reviewed_task("24")
    clause24 = SimulatorChoreography(source_task24).source_clauses[0]
    task24 = _task_instructions(source_task24)
    assert "You want to cancel the grill." in task24
    assert "you regret and want to keep it" not in task24
    assert clause24 in _task_instructions(
        source_task24,
        confirmation_source_clause=clause24,
    )

    source_task71 = _reviewed_task("71")
    clause71 = SimulatorChoreography(source_task71).source_clauses[0]
    task71 = _task_instructions(source_task71)
    assert "you say GC initially." in task71
    assert "change your mind to PayPal" not in task71
    assert "initially, but" not in task71
    assert clause71 in _task_instructions(
        source_task71,
        confirmation_source_clause=clause71,
    )

    source_task72 = _reviewed_task("72")
    clause72 = SimulatorChoreography(source_task72).source_clauses[0]
    task72 = _task_instructions(source_task72)
    assert "you say gift card initially." in task72
    assert "change your mind to using PayPal" not in task72
    assert "initially, but" not in task72
    assert clause72 in _task_instructions(
        source_task72,
        confirmation_source_clause=clause72,
    )


def test_task84_premature_switch_is_repaired_then_revealed_at_confirmation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.runner.MCPProcessClient",
        lambda **_kwargs: _FakeMCP(),
    )
    client = _Task84LatentConfirmationChangeClient()
    bundle = run_coordinate(
        _task84_coordinate(),
        client=client,  # type: ignore[arg-type]
        tasks_by_id={"84": _task84()},
        runtime=_runtime(),
    )

    assert bundle.status == "complete"
    user_messages = [
        str(message["content"])
        for message in bundle.messages
        if message.get("role") == "user"
    ]
    assert user_messages == [
        (
            "I received two tablets and only need one. I want to return the "
            "cheaper one and have the refund sent to my credit card."
        ),
        (
            "The cheaper tablet, please, with the refund going back to my "
            "credit card."
        ),
        (
            "Actually, return the more expensive tablet instead and put the "
            "refund on my gift card."
        ),
        "Yes.",
    ]
    rejected = [
        call
        for call in bundle.model_calls
        if call.transcript_disposition == "rejected_simulator_conformance"
    ]
    assert [call.output_message["content"] for call in rejected] == [
        (
            "Actually, return the more expensive tablet instead and put the "
            "refund on my gift card."
        )
    ]
    assert len(bundle.simulator_conformance_receipts) == 1
    repair = bundle.simulator_conformance_receipts[0]
    assert repair.schema_version == 3
    assert repair.event_id == "task-84-confirmation-1"
    assert repair.boundary == "premature_confirmation_condition"
    assert repair.staged_tool_call_id is None
    assert repair.staged_action_sha256 is None
    assert repair.trigger_message_index == 2
    assert repair.outcome == "repaired"
    assert [record.execution_disposition for record in bundle.tool_calls] == [
        "prepared",
        "prepared",
        "executed",
    ]
    assert _FakeMCP.calls == [
        (
            "return_delivered_order_items",
            {
                "order_id": "#R0000950",
                "item_ids": ["item_00326"],
                "payment_method_id": "gift_card_000401",
            },
        )
    ]
    assert bundle.mutation_confirmation_ok is True
    conditional_clause = SimulatorChoreography(_task84()).source_clauses[0]
    assert len(client.user_system_prompts) == 6
    assert conditional_clause not in client.user_system_prompts[0]
    assert conditional_clause not in client.user_system_prompts[1]
    assert conditional_clause not in client.user_system_prompts[2]
    assert conditional_clause in client.user_system_prompts[3]
    assert conditional_clause not in client.user_system_prompts[4]
    assert conditional_clause not in client.user_system_prompts[5]
    public = str(build_public_payload(_ledger(bundle), [bundle]))
    assert "task-84-confirmation-1" not in public
    assert "applied this conditional scenario instruction" not in public


@pytest.mark.parametrize(
    "coordinate_id",
    (
        "52e4f796-4dd5-5dd1-b02a-93ac6121c8cf",
        "4d2be00c-8c42-5bc7-ad3e-7f6cecd1fe92",
    ),
)
def test_generation_five_v2_conformance_receipts_keep_sealed_bundle_digest(
    coordinate_id: str,
) -> None:
    path = (
        RUNS_DIR
        / "canary"
        / "ce8ca1f8e03634fbac96afa85af5fec337f0be1bb9620fb6ff030570ab0f40f7"
        / "bundles"
        / f"{coordinate_id}.json"
    )
    raw = load_json(path)
    assert raw["simulator_conformance_receipts"]
    assert {
        receipt["schema_version"]
        for receipt in raw["simulator_conformance_receipts"]
    } == {2}

    bundle = ConversationBundle.model_validate(raw)
    serialized = bundle.model_dump(mode="json")
    assert all(
        "boundary" not in receipt
        and "trigger_message_index" not in receipt
        and "trigger_message_sha256" not in receipt
        for receipt in serialized["simulator_conformance_receipts"]
    )
    assert digest_value(bundle.unsigned_payload()) == bundle.bundle_sha256
