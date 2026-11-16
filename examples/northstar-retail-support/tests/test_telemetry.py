from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode

from retail_support_study_v1.canonical import digest_value
from retail_support_study_v1.confirmation import (
    MutationConfirmationReceipt,
    action_sha256,
    confirmation_intent_id,
    public_preparation_result,
)
from retail_support_study_v1.constants import (
    DESTINATION_PROJECT,
    PACKAGE_ROOT,
    STUDY_ID,
)
from retail_support_study_v1.emitter import (
    AGENT_LENS_OTLP_ENDPOINT,
    AGENT_LENS_RESOURCE_ATTRIBUTES,
    EMISSION_CONVERSATION_BATCH_SIZE,
    _verify_emission_snapshot,
    build_agent_lens_otlp_exporter,
    emit_frozen_cohort,
    emit_frozen_to_agent_lens,
    plan_frozen_emission,
)
from retail_support_study_v1.schemas import (
    ConversationBundle,
    Coordinate,
    ModelCall,
    RunLedger,
    ToolCallRecord,
    Usage,
)
from retail_support_study_v1.telemetry import (
    CHAT_ATTRIBUTE_KEYS,
    MCP_ATTRIBUTE_KEYS,
    ROOT_ATTRIBUTE_KEYS,
    build_public_payload,
    emit_bundle,
    validate_frozen_inputs,
    validate_public_payload,
)

_SNAPSHOT_MANIFEST_SHA = "9" * 64
_RETAINED_UNCERTAIN_LEDGERS = (
    PACKAGE_ROOT
    / "runs/canary/efa7fccb16eb06dcc9780eb7e176352893b4b777ac23c509293fc9a4a8a562f8/ledger.json",
    PACKAGE_ROOT
    / "runs/canary/f3c20c49c5fc4d726c86527624b4cf1030fe5678e9e220bc1b7638cd30bc5084/ledger.json",
    PACKAGE_ROOT
    / "runs/canary/ce8ca1f8e03634fbac96afa85af5fec337f0be1bb9620fb6ff030570ab0f40f7/ledger.json",
)


@pytest.fixture(autouse=True)
def _verified_runtime_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "retail_support_study_v1.emitter._verify_emission_snapshot",
        lambda _ledger: (_SNAPSHOT_MANIFEST_SHA, {}),
    )


def _usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        observed_cost_usd=0.001,
    )


def _support_call(
    call_id: str,
    messages: list[dict[str, Any]],
    output_index: int,
) -> ModelCall:
    return ModelCall(
        call_id=call_id,
        participant="support_agent",
        requested_model="support-model-v1",
        returned_model="support-model-v1.0.3",
        temperature=0.0,
        requested_max_tokens=4_096,
        input_messages=[
            {
                "role": "system",
                "content": "PRIVATE support policy that must not be emitted",
            },
            *messages[:output_index],
        ],
        output_message=messages[output_index],
        usage=_usage(20 + output_index, 4),
    )


def _simulator_call(
    call_id: str, output_message: dict[str, Any], seed: int
) -> ModelCall:
    return ModelCall(
        call_id=call_id,
        participant="user_simulator",
        requested_model="private-simulator-model",
        returned_model="private-simulator-model-build",
        temperature=0.4,
        requested_max_tokens=512,
        seed=seed,
        input_messages=[
            {
                "role": "system",
                "content": "PRIVATE simulator scenario and task truth",
            }
        ],
        output_message=output_message,
        usage=_usage(100, 10),
    )


def _bundle(*, tool_error: bool = False) -> ConversationBundle:
    cancel_arguments = {
        "order_id": "R-100",
        "reason": "ordered by mistake",
    }
    cancel_proposal = (
        "I can cancel order R-100 because it was ordered by mistake. "
        "Would you like me to proceed?"
    )
    first_tool = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "model-tool-1",
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "arguments": json.dumps({"order_id": "R-100"}),
                },
            }
        ],
    }
    prepared_tool = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "model-tool-2-prepared",
                "type": "function",
                "function": {
                    "name": "cancel_pending_order",
                    "arguments": json.dumps(cancel_arguments),
                },
            }
        ],
    }
    second_tool = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "model-tool-2",
                "type": "function",
                "function": {
                    "name": "cancel_pending_order",
                    "arguments": json.dumps(cancel_arguments),
                },
            }
        ],
    }
    second_tool_content = (
        json.dumps(
            {
                "error": "BackendCredentialError",
                "message": "raw private error text",
            }
        )
        if tool_error
        else json.dumps({"status": "cancelled"})
    )
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "What is happening with order R-100?"},
        first_tool,
        {
            "role": "tool",
            "tool_call_id": "model-tool-1",
            "content": json.dumps({"status": "processing"}),
        },
        {
            "role": "assistant",
            "content": "Order R-100 is still processing.",
        },
        {"role": "user", "content": "Please cancel it."},
        prepared_tool,
        {
            "role": "tool",
            "tool_call_id": "model-tool-2-prepared",
            "content": json.dumps(
                public_preparation_result()
            ),
        },
        {
            "role": "assistant",
            "content": cancel_proposal,
        },
        {"role": "user", "content": "Yes, please."},
        second_tool,
        {
            "role": "tool",
            "tool_call_id": "model-tool-2",
            "content": second_tool_content,
        },
        {
            "role": "assistant",
            "content": (
                "I could not cancel the order, so I made no change."
                if tool_error
                else "Order R-100 is now cancelled."
            ),
        },
    ]
    calls = [
        _simulator_call("simulator-1", messages[1], 11),
        _support_call("agent-1", messages, 2),
        _support_call("agent-2", messages, 4),
        _simulator_call("simulator-2", messages[5], 12),
        _support_call("agent-3", messages, 6),
        _support_call("agent-4", messages, 8),
        _simulator_call("simulator-3", messages[9], 13),
        _support_call("agent-5", messages, 10),
        _support_call("agent-6", messages, 12),
    ]
    prepared_record = ToolCallRecord(
        call_id="model-tool-2-prepared",
        name="cancel_pending_order",
        arguments=cancel_arguments,
        result=public_preparation_result(),
        execution_disposition="prepared",
        duration_ms=0.5,
    )
    second_record = ToolCallRecord(
        call_id="model-tool-2",
        name="cancel_pending_order",
        arguments=cancel_arguments,
        result=None if tool_error else {"status": "cancelled"},
        error_type="BackendCredentialError" if tool_error else None,
        error_message="raw private error text" if tool_error else None,
        execution_disposition="executed",
        duration_ms=8.5,
    )
    confirmation_action_digest = action_sha256(
        "cancel_pending_order", cancel_arguments
    )
    confirmation_receipt = MutationConfirmationReceipt(
        conversation_id="conversation-public-001",
        sequence=1,
        intent_id=confirmation_intent_id(
            conversation_id="conversation-public-001",
            sequence=1,
            action_digest=confirmation_action_digest,
        ),
        tool_name="cancel_pending_order",
        action_sha256=confirmation_action_digest,
        staged_tool_call_id="model-tool-2-prepared",
        staged_message_index=6,
        state="consumed",
        proposal_message_index=8,
        proposal_message_sha256=digest_value(cancel_proposal),
        decision_message_index=9,
        decision_message_sha256=digest_value("Yes, please."),
        consumed_tool_call_id="model-tool-2",
        consumed_message_index=10,
    )
    bundle = ConversationBundle(
        study_id=STUDY_ID,
        conversation_id="conversation-public-001",
        coordinate=Coordinate(
            coordinate_id="coordinate-private-001",
            logical_task_id="task-truth-private-001",
            profile="ordinary",
            seed=7,
            cohort="canary",
        ),
        requested_models={
            "support_agent": "support-model-v1",
            "user_simulator": "private-simulator-model",
        },
        returned_models={
            "support_agent": ["support-model-v1.0.3"],
            "user_simulator": ["private-simulator-model-build"],
        },
        status="complete",
        messages=messages,
        model_calls=calls,
        tool_calls=[
            ToolCallRecord(
                call_id="model-tool-1",
                name="get_order_details",
                arguments={"order_id": "R-100"},
                result={"status": "processing"},
                execution_disposition="executed",
                duration_ms=4.25,
            ),
            prepared_record,
            second_record,
        ],
        usage=Usage(
            input_tokens=sum(call.usage.input_tokens for call in calls),
            output_tokens=sum(call.usage.output_tokens for call in calls),
            total_tokens=sum(call.usage.total_tokens for call in calls),
            observed_cost_usd=0.006,
        ),
        mutation_confirmation_receipts=[confirmation_receipt],
        mutation_confirmation_ok=True,
        terminal_response=True,
    )
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
    return bundle


def _ledger(bundle: ConversationBundle) -> RunLedger:
    coordinate_id = bundle.coordinate.coordinate_id
    ledger = RunLedger(
        study_id=STUDY_ID,
        cohort="canary",
        preview_sha256="1" * 64,
        planned_coordinate_ids=[coordinate_id],
        completed_coordinate_ids=[coordinate_id],
        unavailable={},
        observed_cost_usd=0.006,
        bundle_digests={coordinate_id: bundle.bundle_sha256},
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    return ledger


def _second_bundle() -> ConversationBundle:
    bundle = _bundle()
    bundle.coordinate = Coordinate(
        coordinate_id="coordinate-private-002",
        logical_task_id="task-truth-private-002",
        profile="concise",
        seed=8,
        cohort="canary",
    )
    bundle.conversation_id = "conversation-public-002"
    for receipt in bundle.mutation_confirmation_receipts:
        receipt.conversation_id = bundle.conversation_id
        receipt.intent_id = confirmation_intent_id(
            conversation_id=bundle.conversation_id,
            sequence=receipt.sequence,
            action_digest=receipt.action_sha256,
        )
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
    return bundle


def _ledger_for_bundles(bundles: list[ConversationBundle]) -> RunLedger:
    ledger = RunLedger(
        study_id=STUDY_ID,
        cohort="canary",
        preview_sha256="1" * 64,
        planned_coordinate_ids=[bundle.coordinate.coordinate_id for bundle in bundles],
        completed_coordinate_ids=[
            bundle.coordinate.coordinate_id for bundle in bundles
        ],
        unavailable={},
        observed_cost_usd=sum(
            bundle.usage.observed_cost_usd or 0.0 for bundle in bundles
        ),
        bundle_digests={
            bundle.coordinate.coordinate_id: bundle.bundle_sha256 for bundle in bundles
        },
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    return ledger


def _replace_exact_string(value: Any, *, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_exact_string(child, old=old, new=new)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_exact_string(child, old=old, new=new) for child in value
        ]
    return new if value == old else value


def _tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "telemetry-test"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _write_frozen_run(
    tmp_path: Path,
    bundle: ConversationBundle | list[ConversationBundle],
    ledger: RunLedger,
) -> Path:
    run_dir = tmp_path / "run"
    bundle_dir = run_dir / "bundles"
    bundle_dir.mkdir(parents=True)
    bundles = [bundle] if isinstance(bundle, ConversationBundle) else bundle
    for item in bundles:
        (bundle_dir / f"{item.coordinate.coordinate_id}.json").write_text(
            json.dumps(item.model_dump(mode="json")), encoding="utf-8"
        )
    ledger_path = run_dir / "ledger.json"
    ledger_path.write_text(json.dumps(ledger.model_dump(mode="json")), encoding="utf-8")
    return ledger_path


def _partial_terminal_run() -> tuple[ConversationBundle, ConversationBundle, RunLedger]:
    complete = _bundle()
    unavailable = _bundle()
    unavailable.coordinate = Coordinate(
        coordinate_id="coordinate-private-002",
        logical_task_id="task-truth-private-002",
        profile="concise",
        seed=8,
        cohort="canary",
    )
    unavailable.conversation_id = "conversation-unavailable-002"
    for receipt in unavailable.mutation_confirmation_receipts:
        receipt.conversation_id = unavailable.conversation_id
        receipt.intent_id = confirmation_intent_id(
            conversation_id=unavailable.conversation_id,
            sequence=receipt.sequence,
            action_digest=receipt.action_sha256,
        )
    unavailable.status = "unavailable"
    unavailable.unavailable_reason = "ProviderUnavailable: transport exhausted"
    unavailable.terminal_response = False
    unavailable.messages[-1]["content"] = "MUST-NOT-EMIT-UNAVAILABLE-CONTENT"
    unavailable.bundle_sha256 = digest_value(unavailable.unsigned_payload())

    complete_id = complete.coordinate.coordinate_id
    unavailable_id = unavailable.coordinate.coordinate_id
    ledger = RunLedger(
        study_id=STUDY_ID,
        cohort="canary",
        preview_sha256="1" * 64,
        planned_coordinate_ids=[complete_id, unavailable_id],
        completed_coordinate_ids=[complete_id],
        unavailable={unavailable_id: unavailable.unavailable_reason},
        observed_cost_usd=0.012,
        bundle_digests={
            complete_id: complete.bundle_sha256,
            unavailable_id: unavailable.bundle_sha256,
        },
        status="complete",
    )
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    return complete, unavailable, ledger


def test_public_payload_excludes_private_study_and_simulator_state() -> None:
    bundle = _bundle()
    payload = build_public_payload(_ledger(bundle), [bundle])
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["destination_project"] == DESTINATION_PROJECT
    assert payload["payload_sha256"] == digest_value(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    assert "coordinate-private-001" not in encoded
    assert "task-truth-private-001" not in encoded
    assert "private-simulator-model" not in encoded
    assert "PRIVATE simulator" not in encoded
    assert "PRIVATE support policy" not in encoded
    assert '"role": "system"' not in encoded
    receipt = bundle.mutation_confirmation_receipts[0]
    assert receipt.intent_id not in encoded
    assert receipt.action_sha256 not in encoded
    assert "proposal_message_sha256" not in encoded
    assert "decision_message_sha256" not in encoded
    assert "execution_disposition" not in encoded
    assert len(payload["conversations"][0]["turns"]) == 3
    assert [
        child["kind"]
        for turn in payload["conversations"][0]["turns"]
        for child in turn["children"]
    ] == [
        "chat",
        "mcp_tool",
        "chat",
        "chat",
        "chat",
        "chat",
        "mcp_tool",
        "chat",
    ]


def test_prepared_result_cannot_smuggle_private_confirmation_fields() -> None:
    bundle = _bundle()
    value = bundle.model_dump(mode="json")
    prepared = next(
        record
        for record in value["tool_calls"]
        if record["execution_disposition"] == "prepared"
    )
    prepared["result"]["intent_id"] = "private-intent"

    with pytest.raises(ValueError, match="fixed public result"):
        ConversationBundle.model_validate(value)


def test_consumed_receipt_cannot_bind_a_non_assent_decision() -> None:
    bundle = _bundle()
    value = bundle.model_dump(mode="json")
    value["messages"][9]["content"] = "No."
    value["mutation_confirmation_receipts"][0][
        "decision_message_sha256"
    ] = digest_value("No.")

    with pytest.raises(ValueError, match="not unqualified assent"):
        ConversationBundle.model_validate(value)


def test_emits_parentless_turn_roots_and_real_child_spans() -> None:
    bundle = _bundle()
    provider, exporter = _tracer()
    counts = emit_bundle(provider.get_tracer("test"), bundle)
    provider.shutdown()
    spans = exporter.get_finished_spans()

    assert counts.conversations == 1
    assert counts.turns == 3
    assert counts.chat_spans == 6
    assert counts.tool_spans == 2
    assert len(spans) == 11

    roots = [span for span in spans if span.name == "invoke_agent"]
    chats = [span for span in spans if span.name.startswith("chat ")]
    tools = [span for span in spans if span.name.startswith("tools/call ")]
    assert len(roots) == 3
    assert all(span.kind is SpanKind.INTERNAL for span in roots)
    assert all(span.parent is None for span in roots)
    assert all(
        span.attributes["gen_ai.conversation.id"] == "conversation-public-001"
        for span in spans
    )
    assert roots[0].context.trace_id != roots[1].context.trace_id
    root_ids = {root.context.span_id for root in roots}
    assert all(span.kind is SpanKind.CLIENT for span in [*chats, *tools])
    assert all(span.parent.span_id in root_ids for span in [*chats, *tools])

    assert all(set(span.attributes) <= ROOT_ATTRIBUTE_KEYS for span in roots)
    assert all(set(span.attributes) <= CHAT_ATTRIBUTE_KEYS for span in chats)
    assert all(set(span.attributes) <= MCP_ATTRIBUTE_KEYS for span in tools)
    assert all(span.attributes["gen_ai.provider.name"] == "wandb" for span in chats)
    assert all(span.attributes["mcp.method.name"] == "tools/call" for span in tools)
    assert all(span.attributes["network.transport"] == "pipe" for span in tools)
    assert [span.attributes["gen_ai.tool.call.id"] for span in tools] == [
        "model-tool-1",
        "model-tool-2",
    ]
    assert all("jsonrpc.request.id" not in span.attributes for span in tools)

    serialized_attributes = json.dumps(
        [dict(span.attributes) for span in spans], sort_keys=True
    )
    assert "private-simulator-model" not in serialized_attributes
    assert "PRIVATE" not in serialized_attributes
    assert "task-truth" not in serialized_attributes
    receipt = bundle.mutation_confirmation_receipts[0]
    assert receipt.intent_id not in serialized_attributes
    assert receipt.action_sha256 not in serialized_attributes


def test_recovered_tool_error_is_child_error_without_raw_text() -> None:
    bundle = _bundle(tool_error=True)
    provider, exporter = _tracer()
    emit_bundle(provider.get_tracer("test"), bundle)
    provider.shutdown()
    spans = exporter.get_finished_spans()

    failed = next(
        span for span in spans if span.name == "tools/call cancel_pending_order"
    )
    roots = [span for span in spans if span.name == "invoke_agent"]
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.status.description == "BackendCredentialError"
    assert failed.attributes["error.type"] == "BackendCredentialError"
    assert "gen_ai.tool.call.result" not in failed.attributes
    assert all(root.status.status_code is StatusCode.OK for root in roots)
    encoded = json.dumps([dict(span.attributes) for span in spans])
    assert "raw private error text" not in encoded


def test_locally_rejected_tool_attempt_has_no_mcp_execution_span() -> None:
    bundle = _bundle()
    rejected = bundle.tool_calls[0]
    rejected.execution_disposition = "rejected"
    rejected.result = None
    rejected.error_type = "InvalidToolArguments"
    rejected.error_message = "rejected before tools/call"
    bundle.messages[3]["content"] = json.dumps(
        {"error": "invalid_tool_arguments"}
    )
    for call in bundle.model_calls:
        if call.participant != "support_agent":
            continue
        output_index = next(
            index
            for index, message in enumerate(bundle.messages)
            if message == call.output_message
        )
        call.input_messages = [call.input_messages[0], *bundle.messages[:output_index]]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    provider, exporter = _tracer()
    counts = emit_bundle(provider.get_tracer("test"), bundle)
    provider.shutdown()

    tools = [
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("tools/call ")
    ]
    assert counts.tool_spans == 1
    assert [span.name for span in tools] == ["tools/call cancel_pending_order"]


def test_frozen_input_validation_rejects_digest_and_denominator_drift() -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    validate_frozen_inputs(ledger, [bundle])

    bundle.bundle_sha256 = "a" * 64
    with pytest.raises(ValueError, match="bundle digest drift"):
        validate_frozen_inputs(ledger, [bundle])

    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger.completed_coordinate_ids = []
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    with pytest.raises(ValueError, match="completed and unavailable"):
        validate_frozen_inputs(ledger, [bundle])


def test_terminal_partial_run_emits_only_exact_completed_trajectories(
    tmp_path: Path,
) -> None:
    complete, unavailable, ledger = _partial_terminal_run()
    normalized = validate_frozen_inputs(ledger, [complete, unavailable])
    payload = build_public_payload(ledger, normalized)
    encoded = json.dumps(payload, sort_keys=True)

    assert set(normalized) == {
        complete.coordinate.coordinate_id,
        unavailable.coordinate.coordinate_id,
    }
    assert [item["conversation_id"] for item in payload["conversations"]] == [
        complete.conversation_id
    ]
    assert "MUST-NOT-EMIT-UNAVAILABLE-CONTENT" not in encoded
    assert unavailable.conversation_id not in encoded

    ledger_path = _write_frozen_run(tmp_path, [complete, unavailable], ledger)
    receipt = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=InMemorySpanExporter(),
        receipt_path=tmp_path / "partial-emission-receipt.json",
    )
    assert receipt["status"] == "complete"
    assert receipt["conversation_count"] == 1
    assert receipt["span_count"] == 11


def test_terminal_partial_run_requires_exact_frozen_unavailable_bundle(
    tmp_path: Path,
) -> None:
    complete, unavailable, ledger = _partial_terminal_run()
    ledger_path = _write_frozen_run(tmp_path, [complete, unavailable], ledger)
    unavailable_path = (
        ledger_path.parent / "bundles" / f"{unavailable.coordinate.coordinate_id}.json"
    )
    unavailable_path.unlink()

    with pytest.raises(ValueError, match="missing regular frozen bundle"):
        plan_frozen_emission(ledger_path=ledger_path)


def test_terminal_partial_run_rejects_partition_or_reason_drift() -> None:
    complete, unavailable, ledger = _partial_terminal_run()
    unavailable_id = unavailable.coordinate.coordinate_id
    ledger.completed_coordinate_ids.append(unavailable_id)
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    with pytest.raises(ValueError, match="both complete and unavailable"):
        validate_frozen_inputs(ledger, [complete, unavailable])

    complete, unavailable, ledger = _partial_terminal_run()
    ledger.unavailable[unavailable.coordinate.coordinate_id] = "another reason"
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())
    with pytest.raises(ValueError, match="unavailable reason drift"):
        validate_frozen_inputs(ledger, [complete, unavailable])


@pytest.mark.parametrize("status", ["running", "paused", "blocked_budget", "uncertain"])
def test_nonterminal_ledgers_cannot_emit(status: str) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger.status = status
    if status == "uncertain":
        ledger.ambiguous_cost_reserve_usd = 1.0
        ledger.cost_exposure_upper_bound_usd = 1.006
        ledger.cost_uncertainty_reasons = ["unreconciled provider outcome"]
    ledger.ledger_sha256 = digest_value(ledger.unsigned_payload())

    with pytest.raises(ValueError, match="only a complete cohort"):
        validate_frozen_inputs(ledger, [bundle])


@pytest.mark.parametrize("ledger_path", _RETAINED_UNCERTAIN_LEDGERS)
@pytest.mark.parametrize("operation", ["plan", "emit"])
def test_retained_uncertain_ledgers_fail_before_snapshot_or_bundle_loading(
    monkeypatch: pytest.MonkeyPatch,
    ledger_path: Path,
    operation: str,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("later frozen inputs must not be touched")

    monkeypatch.setattr(
        "retail_support_study_v1.emitter._verify_emission_snapshot", forbidden
    )
    monkeypatch.setattr(
        "retail_support_study_v1.emitter._load_frozen_bundles", forbidden
    )

    with pytest.raises(ValueError, match="only a complete cohort"):
        if operation == "plan":
            plan_frozen_emission(ledger_path=ledger_path)
        else:
            emit_frozen_cohort(
                ledger_path=ledger_path,
                approve_ledger_sha="a" * 64,
                approve_payload_sha="b" * 64,
                exporter=InMemorySpanExporter(),
            )


def test_emission_checks_ledger_digest_before_noncomplete_status_or_later_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retained = json.loads(_RETAINED_UNCERTAIN_LEDGERS[1].read_text(encoding="utf-8"))
    retained["ledger_sha256"] = "0" * 64
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(retained), encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("later frozen inputs must not be touched")

    monkeypatch.setattr(
        "retail_support_study_v1.emitter._verify_emission_snapshot", forbidden
    )
    monkeypatch.setattr(
        "retail_support_study_v1.emitter._load_frozen_bundles", forbidden
    )

    with pytest.raises(ValueError, match="emission ledger digest"):
        plan_frozen_emission(ledger_path=ledger_path)


def test_public_payload_digest_binds_visible_conversation_content() -> None:
    first = _bundle()
    first_payload = build_public_payload(_ledger(first), [first])

    second = _bundle()
    second.messages[-1]["content"] = "Order R-100 is cancelled and closed."
    second.model_calls[-1].output_message = second.messages[-1]
    second.bundle_sha256 = digest_value(second.unsigned_payload())
    second_payload = build_public_payload(_ledger(second), [second])

    assert first_payload["payload_sha256"] != second_payload["payload_sha256"]


def test_public_projection_rejects_private_keys_before_export() -> None:
    bundle = _bundle()
    bundle.tool_calls[0].result = {
        "status": "processing",
        "private_label": "must never leave review storage",
    }
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    with pytest.raises(ValueError, match="forbidden public key"):
        build_public_payload(_ledger(bundle), [bundle])


def test_public_projection_allows_canonical_uuid_in_tool_call_id_slots() -> None:
    tool_call_uuid = "12345678-1234-4234-8234-123456789012"
    raw_bundle = _replace_exact_string(
        _bundle().model_dump(mode="python"),
        old="model-tool-1",
        new=tool_call_uuid,
    )
    raw_bundle["bundle_sha256"] = ""
    bundle = ConversationBundle.model_validate(raw_bundle)
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    payload = build_public_payload(_ledger(bundle), [bundle])

    encoded = json.dumps(payload)
    assert tool_call_uuid in encoded


def test_public_projection_rejects_uuid_shaped_card_text_in_content() -> None:
    bundle = _bundle()
    bundle.messages[-1]["content"] = "12345678-1234-4234-8234-123456789012"
    bundle.model_calls[-1].output_message = bundle.messages[-1]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    with pytest.raises(ValueError, match="payment-card-like"):
        build_public_payload(_ledger(bundle), [bundle])


@pytest.mark.parametrize("field", ["arguments", "result"])
def test_public_projection_rejects_uuid_shaped_card_text_in_tool_data(
    field: str,
) -> None:
    bundle = _bundle()
    payload = build_public_payload(_ledger(bundle), [bundle])
    mcp_child = next(
        child
        for turn in payload["conversations"][0]["turns"]
        for child in turn["children"]
        if child["kind"] == "mcp_tool" and field in child
    )
    mcp_child[field]["note"] = "12345678-1234-4234-8234-123456789012"
    payload["payload_sha256"] = digest_value(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )

    with pytest.raises(ValueError, match="payment-card-like"):
        validate_public_payload(payload)


def test_public_projection_still_rejects_real_payment_card_text() -> None:
    bundle = _bundle()
    bundle.messages[-1]["content"] = "The card is 4111 1111 1111 1111."
    bundle.model_calls[-1].output_message = bundle.messages[-1]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    with pytest.raises(ValueError, match="payment-card-like"):
        build_public_payload(_ledger(bundle), [bundle])


@pytest.mark.parametrize(
    "value",
    [
        "file:///private/source.json",
        "/Users/operator/private/source.json",
        "/private/tmp/private-source.json",
        "/tmp/private-source.json",
        "/workspace/private/source.json",
    ],
)
def test_public_projection_rejects_local_paths(value: str) -> None:
    bundle = _bundle()
    bundle.messages[-1]["content"] = value
    bundle.model_calls[-1].output_message = bundle.messages[-1]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    with pytest.raises(ValueError, match="local-only text"):
        build_public_payload(_ledger(bundle), [bundle])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("Contact me at shopper@gmail.com.", "non-fictional email"),
        ("The card is 4111 1111 1111 1111.", "payment-card-like"),
        ("My SSN is 123-45-6789.", "SSN-like"),
        ("Call me at +1 (212) 555-0199.", "phone-number-like"),
        ("Call me at 2125550199.", "phone-number-like"),
    ],
)
def test_public_projection_rejects_model_invented_pii(value: str, message: str) -> None:
    bundle = _bundle()
    bundle.messages[-1]["content"] = value
    bundle.model_calls[-1].output_message = bundle.messages[-1]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    with pytest.raises(ValueError, match=message):
        build_public_payload(_ledger(bundle), [bundle])


@pytest.mark.parametrize(
    "value",
    [
        "Email customer-0001@example.invalid.",
        "Email customer-0001@example.com.",
        "Email customer-0001@example.net.",
        "Email customer-0001@example.org.",
        "The card suffix is 1234.",
        "Order #R0000859 ships to 4320 Lantern Way, Northstar, NY 25994.",
        "Use payment method credit_card_000001.",
    ],
)
def test_public_projection_allows_reviewed_fictional_retail_identity(
    value: str,
) -> None:
    bundle = _bundle()
    bundle.messages[-1]["content"] = value
    bundle.model_calls[-1].output_message = bundle.messages[-1]
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())

    payload = build_public_payload(_ledger(bundle), [bundle])
    assert value in json.dumps(payload)


def test_emitter_requires_exact_approvals_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])
    receipt_path = tmp_path / "emission-receipt.json"
    exporter = InMemorySpanExporter()
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "private_label=must-not-emit,service.name=ambient-service",
    )
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "1")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "4")

    with pytest.raises(ValueError, match="exact frozen ledger"):
        emit_frozen_cohort(
            ledger_path=ledger_path,
            approve_ledger_sha="f" * 64,
            approve_payload_sha=payload["payload_sha256"],
            exporter=exporter,
            receipt_path=receipt_path,
        )
    assert not receipt_path.exists()

    receipt = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=exporter,
        receipt_path=receipt_path,
    )
    first_span_count = len(exporter.get_finished_spans())
    assert receipt["status"] == "complete"
    assert receipt["span_count"] == 11
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert all(
        dict(span.resource.attributes) == AGENT_LENS_RESOURCE_ATTRIBUTES
        for span in exporter.get_finished_spans()
    )
    root = next(
        span for span in exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    assert set(root.attributes) == ROOT_ATTRIBUTE_KEYS
    assert (
        "What is happening with order R-100?"
        in root.attributes["gen_ai.input.messages"]
    )

    repeated = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=exporter,
        receipt_path=receipt_path,
    )
    assert repeated == receipt
    assert len(exporter.get_finished_spans()) == first_span_count


def test_emission_plan_is_read_only_and_binds_exact_public_payload(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])

    plan = plan_frozen_emission(ledger_path=ledger_path)

    assert plan == {
        "schema_version": 1,
        "destination_project": DESTINATION_PROJECT,
        "ledger_sha256": ledger.ledger_sha256,
        "public_payload_sha256": payload["payload_sha256"],
        "runtime_snapshot_manifest_sha256": _SNAPSHOT_MANIFEST_SHA,
        "conversation_count": 1,
        "turn_count": 3,
        "emission_mode": "batched_frozen_bundle_replay",
        "delivery_semantics": "at_least_once",
        "conversation_batch_size": EMISSION_CONVERSATION_BATCH_SIZE,
    }
    assert not (tmp_path / "emission-receipt.json").exists()


def test_emission_snapshot_verifies_sealed_runtime_not_current_projection_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    captured: dict[str, Any] = {}

    class Runtime:
        preview = {
            "study_id": ledger.study_id,
            "cohorts": {
                ledger.cohort: {
                    "coordinates": [
                        {"coordinate_id": coordinate_id}
                        for coordinate_id in ledger.planned_coordinate_ids
                    ]
                }
            },
        }
        manifest = {"manifest_sha256": "9" * 64}

    def verify_snapshot(
        preview_sha256: str,
        *,
        require_current_code: bool = True,
    ) -> Runtime:
        captured.update(
            preview_sha256=preview_sha256,
            require_current_code=require_current_code,
        )
        return Runtime()

    monkeypatch.setattr(
        "retail_support_study_v1.emitter.verify_runtime_snapshot",
        verify_snapshot,
    )

    manifest_sha, preview = _verify_emission_snapshot(ledger)
    assert manifest_sha == "9" * 64
    assert preview == Runtime.preview
    assert captured == {
        "preview_sha256": ledger.preview_sha256,
        "require_current_code": False,
    }


def test_concrete_exporter_has_one_fixed_destination_and_basic_auth() -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> SpanExporter:
        captured.update(kwargs)
        return InMemorySpanExporter()

    exporter = build_agent_lens_otlp_exporter(
        api_key="test-wandb-key",
        environ={
            # Generic W&B routing variables cannot change the fixed export.
            "WANDB_ENTITY": "ignored-entity",
            "WANDB_PROJECT": "ignored-project",
        },
        exporter_factory=factory,
    )

    assert isinstance(exporter, InMemorySpanExporter)
    assert captured["endpoint"] == AGENT_LENS_OTLP_ENDPOINT
    assert captured["timeout"] == 30
    assert captured["compression"] is Compression.Gzip
    assert captured["headers"]["project_id"] == DESTINATION_PROJECT
    scheme, encoded = captured["headers"]["Authorization"].split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode("utf-8") == "api:test-wandb-key"


@pytest.mark.parametrize(
    "name",
    [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TIMEOUT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
    ],
)
def test_concrete_exporter_rejects_otlp_overrides(name: str) -> None:
    with pytest.raises(ValueError, match="rejects OTLP transport overrides"):
        build_agent_lens_otlp_exporter(
            api_key="test-wandb-key",
            environ={name: "https://example.invalid/override"},
        )


def test_concrete_emission_never_persists_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])
    receipt_path = tmp_path / "emission-receipt.json"
    exporter = InMemorySpanExporter()

    def fixed_exporter(**kwargs: Any) -> SpanExporter:
        assert kwargs["api_key"] == "test-wandb-key"
        assert kwargs["environ"] == {"WANDB_API_KEY": "test-wandb-key"}
        return exporter

    monkeypatch.setattr(
        "retail_support_study_v1.emitter.build_agent_lens_otlp_exporter",
        fixed_exporter,
    )
    receipt = emit_frozen_to_agent_lens(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        environ={"WANDB_API_KEY": "test-wandb-key"},
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "complete"
    assert "test-wandb-key" not in receipt_path.read_text(encoding="utf-8")

    def must_not_construct_exporter(**kwargs: Any) -> SpanExporter:
        raise AssertionError("a complete receipt must be a network-free no-op")

    monkeypatch.setattr(
        "retail_support_study_v1.emitter.build_agent_lens_otlp_exporter",
        must_not_construct_exporter,
    )
    repeated = emit_frozen_to_agent_lens(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        environ={},
        receipt_path=receipt_path,
    )
    assert repeated == receipt


def test_concrete_emission_checks_approval_before_credentials(tmp_path: Path) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])

    with pytest.raises(ValueError, match="exact frozen ledger"):
        emit_frozen_to_agent_lens(
            ledger_path=ledger_path,
            approve_ledger_sha="f" * 64,
            approve_payload_sha=payload["payload_sha256"],
            environ={},
            receipt_path=tmp_path / "emission-receipt.json",
        )


@pytest.mark.parametrize("api_key", ["", " key", "key ", "key\nvalue"])
def test_concrete_exporter_rejects_missing_or_multiline_key(api_key: str) -> None:
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        build_agent_lens_otlp_exporter(api_key=api_key, environ={})


class _FailingExporter(SpanExporter):
    def export(self, spans: Any) -> SpanExportResult:
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None


class _RecordingExporter(SpanExporter):
    def __init__(self, *, fail_after_successes: int | None = None) -> None:
        self.batch_sizes: list[int] = []
        self.finished_spans: list[Any] = []
        self.fail_after_successes = fail_after_successes

    def export(self, spans: Any) -> SpanExportResult:
        batch = list(spans)
        self.batch_sizes.append(len(batch))
        if (
            self.fail_after_successes is not None
            and len(self.batch_sizes) > self.fail_after_successes
        ):
            return SpanExportResult.FAILURE
        self.finished_spans.extend(batch)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def test_emitter_uses_batch_processor_instead_of_per_span_export(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])
    exporter = _RecordingExporter()

    receipt = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=exporter,
        receipt_path=tmp_path / "emission-checkpoint.json",
    )

    assert exporter.batch_sizes == [11]
    assert receipt["status"] == "complete"
    assert receipt["delivery_semantics"] == "at_least_once"
    assert receipt["completed_conversation_count"] == 1
    assert receipt["next_conversation_index"] == 1


def test_interrupted_batch_resumes_from_last_completed_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles = [_bundle(), _second_bundle()]
    ledger = _ledger_for_bundles(bundles)
    ledger_path = _write_frozen_run(tmp_path, bundles, ledger)
    payload = build_public_payload(ledger, bundles)
    checkpoint_path = tmp_path / "emission-checkpoint.json"
    monkeypatch.setattr(
        "retail_support_study_v1.emitter.EMISSION_CONVERSATION_BATCH_SIZE",
        1,
    )

    first_exporter = _RecordingExporter(fail_after_successes=1)
    with pytest.raises(RuntimeError, match="did not complete"):
        emit_frozen_cohort(
            ledger_path=ledger_path,
            approve_ledger_sha=ledger.ledger_sha256,
            approve_payload_sha=payload["payload_sha256"],
            exporter=first_exporter,
            receipt_path=checkpoint_path,
        )
    interrupted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed_conversation_count"] == 1
    assert interrupted["next_conversation_index"] == 1
    assert len(first_exporter.finished_spans) == 11

    resumed_exporter = InMemorySpanExporter()
    completed = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=resumed_exporter,
        receipt_path=checkpoint_path,
    )
    resumed_spans = resumed_exporter.get_finished_spans()

    assert completed["status"] == "complete"
    assert completed["completed_conversation_count"] == 2
    assert completed["span_count"] == 22
    assert len(resumed_spans) == 11
    assert {span.attributes["gen_ai.conversation.id"] for span in resumed_spans} == {
        "conversation-public-002"
    }


def test_export_failure_marks_checkpoint_interrupted_and_retryable(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    ledger = _ledger(bundle)
    ledger_path = _write_frozen_run(tmp_path, bundle, ledger)
    payload = build_public_payload(ledger, [bundle])
    receipt_path = tmp_path / "emission-checkpoint.json"

    with pytest.raises(RuntimeError, match="did not complete"):
        emit_frozen_cohort(
            ledger_path=ledger_path,
            approve_ledger_sha=ledger.ledger_sha256,
            approve_payload_sha=payload["payload_sha256"],
            exporter=_FailingExporter(),
            receipt_path=receipt_path,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["next_conversation_index"] == 0

    retry_exporter = InMemorySpanExporter()
    completed = emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=ledger.ledger_sha256,
        approve_payload_sha=payload["payload_sha256"],
        exporter=retry_exporter,
        receipt_path=receipt_path,
    )
    assert completed["status"] == "complete"
    assert len(retry_exporter.get_finished_spans()) == 11
