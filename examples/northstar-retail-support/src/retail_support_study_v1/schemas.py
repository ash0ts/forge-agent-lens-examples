from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from .canonical import digest_value, require_hex_digest
from .confirmation import (
    MutationConfirmationReceipt,
    action_sha256,
    confirmation_intent_id,
    is_unqualified_assent,
    proposal_matches_action,
    public_preparation_result,
)
from .constants import SCHEMA_VERSION, WRITE_TOOLS
from .simulator_output import project_simulator_output


class Coordinate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_id: str
    logical_task_id: str
    profile: Literal["ordinary", "concise", "skeptical"]
    # Kept private with the benchmark coordinate. Agent Lens receives only the
    # projected conversation evidence, never this preregistered replicate key.
    trial_index: int = Field(default=0, ge=0)
    # V10 binds every paid coordinate to one preview-sealed support-model
    # route. Historical coordinates omit the field so their sealed digests
    # remain byte-for-byte verifiable.
    model_route_id: str | None = None
    seed: int
    cohort: Literal["canary", "full"]

    @model_serializer(mode="wrap")
    def serialize_versioned(self, handler: Any) -> dict[str, Any]:
        value = handler(self)
        if "trial_index" not in self.model_fields_set:
            # V1 coordinates predate preregistered trials.  Omitting the new
            # default when those frozen bundles are loaded preserves their
            # sealed content digest; every V2 coordinate supplies this field
            # explicitly and therefore serializes it normally.
            value.pop("trial_index", None)
        if "model_route_id" not in self.model_fields_set:
            value.pop("model_route_id", None)
        return value


class AgentProfile(BaseModel):
    """Preview-sealed public identity and tool boundary for one agent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile_id: str
    agent_name: str
    role: Literal["generalist", "order_management", "returns_resolution"]
    purpose: str
    tool_names: tuple[str, ...]
    instructions_sha256: str
    profile_sha256: str

    @classmethod
    def seal(cls, **values: Any) -> "AgentProfile":
        candidate = cls.model_construct(**values, profile_sha256="")
        return cls(
            **values,
            profile_sha256=digest_value(candidate.model_dump(mode="json")),
        )

    @model_validator(mode="after")
    def validate_seal(self) -> "AgentProfile":
        if not self.profile_id or not self.agent_name or not self.purpose:
            raise ValueError("agent profile identity is incomplete")
        if not self.tool_names or len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("agent profile tools must be nonempty and unique")
        if tuple(sorted(self.tool_names)) != self.tool_names:
            raise ValueError("agent profile tools must use canonical order")
        require_hex_digest(
            self.instructions_sha256,
            label="agent profile instructions_sha256",
        )
        require_hex_digest(self.profile_sha256, label="agent profile")
        if self.profile_sha256 != digest_value(self.unsigned_payload()):
            raise ValueError("agent profile digest mismatch")
        return self

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["profile_sha256"] = ""
        return payload


class ModelRoute(BaseModel):
    """Preview-sealed inference route used by support-agent coordinates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    route_id: str
    provider_name: Literal["wandb", "anthropic"]
    provider_api: Literal["openai_compatible", "anthropic_messages"]
    model: str
    base_url: str
    server_address: str
    credential_env_var: Literal["WANDB_API_KEY", "ANTHROPIC_API_KEY"]
    temperature: float | None
    enable_thinking: bool | None = None
    max_output_tokens: int = Field(gt=0)
    full_coordinate_count: int = Field(ge=0)
    canary_coordinate_count: int = Field(ge=0)
    route_sha256: str

    @classmethod
    def seal(cls, **values: Any) -> "ModelRoute":
        candidate = cls.model_construct(**values, route_sha256="")
        return cls(
            **values,
            route_sha256=digest_value(candidate.model_dump(mode="json")),
        )

    @model_validator(mode="after")
    def validate_seal(self) -> "ModelRoute":
        if not all(
            (
                self.route_id,
                self.model,
                self.base_url,
                self.server_address,
            )
        ):
            raise ValueError("model route identity is incomplete")
        require_hex_digest(self.route_sha256, label="model route")
        if self.route_sha256 != digest_value(self.unsigned_payload()):
            raise ValueError("model route digest mismatch")
        return self

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["route_sha256"] = ""
        return payload


class BehaviorProfile(BaseModel):
    """Preview-sealed latency and rare-fault behavior for the MCP service."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile_id: str
    seed: str
    latency_percentiles_ms: dict[str, dict[str, float]]
    tool_families: dict[str, str]
    sampling_key: tuple[str, ...]
    fault_policy: dict[str, Any]
    profile_sha256: str

    @classmethod
    def seal(cls, **values: Any) -> "BehaviorProfile":
        candidate = cls.model_construct(**values, profile_sha256="")
        return cls(
            **values,
            profile_sha256=digest_value(candidate.model_dump(mode="json")),
        )

    @model_validator(mode="after")
    def validate_seal(self) -> "BehaviorProfile":
        if not self.profile_id or not self.seed:
            raise ValueError("behavior profile identity is incomplete")
        required_percentiles = {"p50", "p95", "p99"}
        for family, values in self.latency_percentiles_ms.items():
            if set(values) != required_percentiles:
                raise ValueError(
                    f"behavior latency family {family!r} has invalid percentiles"
                )
            p50, p95, p99 = (float(values[key]) for key in ("p50", "p95", "p99"))
            if not (0 <= p50 <= p95 <= p99):
                raise ValueError(
                    f"behavior latency family {family!r} is not monotonic"
                )
        if set(self.tool_families.values()) - set(self.latency_percentiles_ms):
            raise ValueError("behavior tool family has no latency distribution")
        require_hex_digest(self.profile_sha256, label="behavior profile")
        if self.profile_sha256 != digest_value(self.unsigned_payload()):
            raise ValueError("behavior profile digest mismatch")
        return self

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["profile_sha256"] = ""
        return payload


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    provider_reported_cost_usd: float | None = Field(default=None, ge=0)
    observed_cost_usd: float | None = None

    @model_validator(mode="after")
    def validate_cached_input(self) -> "Usage":
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed input tokens")
        for label, value in (
            ("provider-reported cost", self.provider_reported_cost_usd),
            ("observed cost", self.observed_cost_usd),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")
        return self


class ModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    participant: Literal["support_agent", "user_simulator"]
    requested_model: str
    returned_model: str
    response_id: str = ""
    system_fingerprint: str | None = None
    finish_reason: str = ""
    transport_attempt_count: int = Field(default=1, ge=1)
    ambiguous_cost_reserve_usd: float = Field(default=0.0, ge=0)
    cost_uncertainty_reasons: list[str] = Field(default_factory=list)
    response_validation_error: str | None = None
    temperature: float | None
    requested_max_tokens: int = Field(gt=0)
    requested_enable_thinking: bool | None = None
    seed: int | None = None
    input_messages: list[dict[str, Any]]
    output_message: dict[str, Any]
    usage: Usage
    transcript_disposition: Literal[
        "accepted", "rejected_simulator_conformance"
    ] = "accepted"

    @model_validator(mode="after")
    def validate_cost_evidence(self) -> "ModelCall":
        if self.ambiguous_cost_reserve_usd and not self.cost_uncertainty_reasons:
            raise ValueError("ambiguous model-call cost requires a reason")
        if not self.ambiguous_cost_reserve_usd and self.cost_uncertainty_reasons:
            raise ValueError("cost uncertainty reasons require an ambiguous reserve")
        if (
            self.transcript_disposition == "rejected_simulator_conformance"
            and self.participant != "user_simulator"
        ):
            raise ValueError("only a simulator draft can be rejected by conformance")
        return self


class SimulatorConformanceReceipt(BaseModel):
    """Private evidence for one bounded simulator correction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 3
    sequence: int = Field(ge=1)
    event_id: str
    boundary: Literal[
        "prepared_action", "premature_confirmation_condition"
    ] = "prepared_action"
    staged_tool_call_id: str | None = None
    staged_action_sha256: str | None = None
    trigger_message_index: int | None = Field(default=None, ge=0)
    trigger_message_sha256: str | None = None
    rejected_call_id: str
    rejected_output_sha256: str
    rejected_message_index: int = Field(ge=0)
    rejected_message_sha256: str
    repair_call_id: str | None = None
    repair_output_sha256: str | None = None
    accepted_message_index: int | None = Field(default=None, ge=0)
    accepted_message_sha256: str | None = None
    outcome: Literal["repaired", "unavailable"]

    @model_serializer(mode="wrap")
    def serialize_versioned(self, handler: Any) -> dict[str, Any]:
        value = handler(self)
        if self.schema_version == 2:
            # V2 receipts predate boundary and trigger evidence. Omitting the
            # model defaults is required to preserve the sealed bundle digest
            # when historical evidence is loaded under the V3-capable model.
            for key in (
                "boundary",
                "trigger_message_index",
                "trigger_message_sha256",
            ):
                value.pop(key, None)
        return value


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error_type: str | None = None
    error_message: str | None = None
    execution_disposition: Literal["prepared", "rejected", "executed"]
    duration_ms: float

    @model_validator(mode="after")
    def validate_execution_disposition(self) -> "ToolCallRecord":
        is_write = self.name in WRITE_TOOLS
        if self.execution_disposition == "prepared":
            if not is_write:
                raise ValueError("read-only tool cannot be prepared for confirmation")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("prepared write cannot be recorded as a tool error")
            if self.result != public_preparation_result():
                raise ValueError("prepared write requires the fixed public result")
        if self.execution_disposition == "rejected" and self.error_type is None:
            raise ValueError("rejected tool call requires an error type")
        return self


class ConversationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    study_id: str
    conversation_id: str
    coordinate: Coordinate
    requested_models: dict[str, str]
    returned_models: dict[str, list[str]]
    status: Literal["complete", "unavailable"]
    unavailable_reason: str | None = None
    messages: list[dict[str, Any]]
    model_calls: list[ModelCall]
    tool_calls: list[ToolCallRecord]
    usage: Usage
    mutation_confirmation_receipts: list[MutationConfirmationReceipt] = Field(
        default_factory=list
    )
    mutation_confirmation_ok: bool | None = None
    simulator_conformance_receipts: list[SimulatorConformanceReceipt] = Field(
        default_factory=list
    )
    terminal_response: bool = False
    bundle_sha256: str = ""

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ConversationBundle":
        if self.status == "complete":
            if self.unavailable_reason is not None:
                raise ValueError("complete bundle cannot have unavailable_reason")
            if not self.terminal_response:
                raise ValueError("complete bundle requires a terminal response")
        elif not self.unavailable_reason:
            raise ValueError("unavailable bundle requires unavailable_reason")
        self._validate_mutation_confirmation_evidence()
        self._validate_simulator_conformance_evidence()
        return self

    def _validate_simulator_conformance_evidence(self) -> None:
        calls = {call.call_id: call for call in self.model_calls}
        if len(calls) != len(self.model_calls) or "" in calls:
            raise ValueError("model call IDs must be non-empty and unique")
        receipts = self.simulator_conformance_receipts
        if [receipt.sequence for receipt in receipts] != list(
            range(1, len(receipts) + 1)
        ):
            raise ValueError("simulator conformance receipt sequence is not canonical")
        referenced_rejections: set[str] = set()
        referenced_repairs: set[str] = set()
        for receipt in receipts:
            if receipt.schema_version not in {2, 3}:
                raise ValueError("simulator conformance receipt schema is unsupported")
            if (
                receipt.schema_version == 2
                and receipt.boundary != "prepared_action"
            ):
                raise ValueError(
                    "legacy simulator conformance receipt has an invalid boundary"
                )
            if not receipt.event_id:
                raise ValueError("simulator conformance event ID is empty")
            require_hex_digest(
                receipt.rejected_output_sha256,
                label="simulator conformance rejected_output_sha256",
            )
            require_hex_digest(
                receipt.rejected_message_sha256,
                label="simulator conformance rejected_message_sha256",
            )
            if receipt.boundary == "prepared_action":
                if (
                    not receipt.staged_tool_call_id
                    or not receipt.staged_action_sha256
                    or receipt.trigger_message_index is not None
                    or receipt.trigger_message_sha256 is not None
                ):
                    raise ValueError(
                        "prepared-action simulator conformance evidence is malformed"
                    )
                require_hex_digest(
                    receipt.staged_action_sha256,
                    label="simulator conformance staged_action_sha256",
                )
                matching_confirmation = [
                    confirmation
                    for confirmation in self.mutation_confirmation_receipts
                    if confirmation.staged_tool_call_id
                    == receipt.staged_tool_call_id
                    and confirmation.action_sha256
                    == receipt.staged_action_sha256
                ]
                if len(matching_confirmation) != 1:
                    raise ValueError(
                        "simulator conformance evidence does not bind one staged action"
                    )
            else:
                if receipt.schema_version != 3:
                    raise ValueError(
                        "premature-disclosure evidence requires receipt schema 3"
                    )
                if (
                    receipt.staged_tool_call_id is not None
                    or receipt.staged_action_sha256 is not None
                    or receipt.trigger_message_index is None
                    or receipt.trigger_message_sha256 is None
                ):
                    raise ValueError(
                        "premature-disclosure simulator evidence is malformed"
                    )
                require_hex_digest(
                    receipt.trigger_message_sha256,
                    label="simulator conformance trigger_message_sha256",
                )
                if receipt.trigger_message_index >= len(self.messages):
                    raise ValueError(
                        "simulator conformance trigger is absent from transcript"
                    )
                trigger = self.messages[receipt.trigger_message_index]
                if (
                    trigger.get("role") != "assistant"
                    or not trigger.get("content")
                    or trigger.get("tool_calls")
                ):
                    raise ValueError(
                        "simulator conformance trigger is not visible assistant text"
                    )
                if digest_value(trigger) != receipt.trigger_message_sha256:
                    raise ValueError("simulator conformance trigger digest drift")
            rejected = calls.get(receipt.rejected_call_id)
            if (
                rejected is None
                or rejected.participant != "user_simulator"
                or rejected.transcript_disposition
                != "rejected_simulator_conformance"
            ):
                raise ValueError("simulator conformance rejection call is invalid")
            if digest_value(rejected.output_message) != receipt.rejected_output_sha256:
                raise ValueError("simulator conformance rejected output drift")
            rejected_text, _ = project_simulator_output(rejected.output_message)
            if (
                digest_value({"role": "user", "content": rejected_text})
                != receipt.rejected_message_sha256
            ):
                raise ValueError(
                    "simulator conformance rejected message projection drift"
                )
            if receipt.rejected_call_id in referenced_rejections:
                raise ValueError(
                    "simulator conformance rejection call was reused"
                )
            referenced_rejections.add(receipt.rejected_call_id)
            if receipt.rejected_message_index < len(self.messages):
                visible_at_rejected_index = self.messages[
                    receipt.rejected_message_index
                ]
                if (
                    digest_value(visible_at_rejected_index)
                    == receipt.rejected_message_sha256
                ):
                    raise ValueError(
                        "rejected simulator draft entered the visible transcript"
                    )

            if (receipt.repair_call_id is None) != (
                receipt.repair_output_sha256 is None
            ):
                raise ValueError("simulator conformance repair evidence is incomplete")
            if receipt.repair_call_id is not None:
                require_hex_digest(
                    str(receipt.repair_output_sha256),
                    label="simulator conformance repair_output_sha256",
                )
                repair = calls.get(receipt.repair_call_id)
                if repair is None or repair.participant != "user_simulator":
                    raise ValueError("simulator conformance repair call is invalid")
                if receipt.repair_call_id in referenced_repairs:
                    raise ValueError("simulator conformance repair call was reused")
                referenced_repairs.add(receipt.repair_call_id)
                if digest_value(repair.output_message) != receipt.repair_output_sha256:
                    raise ValueError("simulator conformance repair output drift")
                if receipt.outcome == "repaired":
                    if repair.transcript_disposition != "accepted":
                        raise ValueError("accepted simulator repair was marked rejected")
                    if (receipt.accepted_message_index is None) != (
                        receipt.accepted_message_sha256 is None
                    ):
                        raise ValueError(
                            "accepted simulator repair transcript evidence is incomplete"
                        )
                    if receipt.accepted_message_index is None:
                        raise ValueError(
                            "accepted simulator repair has no visible message binding"
                        )
                    if receipt.accepted_message_index != receipt.rejected_message_index:
                        raise ValueError(
                            "simulator repair changed the pending transcript position"
                        )
                    require_hex_digest(
                        str(receipt.accepted_message_sha256),
                        label="simulator conformance accepted_message_sha256",
                    )
                    if receipt.accepted_message_index >= len(self.messages):
                        raise ValueError(
                            "accepted simulator repair is absent from the transcript"
                        )
                    if (
                        self.messages[receipt.accepted_message_index].get("role")
                        != "user"
                    ):
                        raise ValueError(
                            "accepted simulator repair is not a customer message"
                        )
                    if (
                        digest_value(self.messages[receipt.accepted_message_index])
                        != receipt.accepted_message_sha256
                    ):
                        raise ValueError(
                            "accepted simulator repair transcript binding drift"
                        )
                    repair_text, repair_stopped = project_simulator_output(
                        repair.output_message
                    )
                    if repair_stopped or not repair_text:
                        raise ValueError(
                            "accepted simulator repair contains terminal control"
                        )
                    if (
                        digest_value({"role": "user", "content": repair_text})
                        != receipt.accepted_message_sha256
                    ):
                        raise ValueError(
                            "accepted simulator repair did not come from model output"
                        )
                else:
                    if (
                        repair.transcript_disposition
                        != "rejected_simulator_conformance"
                    ):
                        raise ValueError("failed simulator repair was marked accepted")
                    referenced_rejections.add(receipt.repair_call_id)
            elif receipt.outcome == "repaired":
                raise ValueError("repaired simulator receipt has no repair call")
            if receipt.outcome == "unavailable":
                if self.status != "unavailable":
                    raise ValueError(
                        "failed simulator conformance requires unavailable bundle"
                    )
                if (
                    receipt.accepted_message_index is not None
                    or receipt.accepted_message_sha256 is not None
                ):
                    raise ValueError(
                        "failed simulator conformance cannot bind an accepted message"
                    )

        rejected_calls = {
            call.call_id
            for call in self.model_calls
            if call.transcript_disposition == "rejected_simulator_conformance"
        }
        if rejected_calls != referenced_rejections:
            raise ValueError("rejected simulator calls and receipts do not align")

    def _validate_mutation_confirmation_evidence(self) -> None:
        records = {record.call_id: record for record in self.tool_calls}
        if len(records) != len(self.tool_calls) or "" in records:
            raise ValueError("tool call IDs must be non-empty and unique")

        receipts = self.mutation_confirmation_receipts
        intent_ids = {receipt.intent_id for receipt in receipts}
        if len(intent_ids) != len(receipts) or "" in intent_ids:
            raise ValueError("confirmation intent IDs must be non-empty and unique")
        if [receipt.sequence for receipt in receipts] != list(
            range(1, len(receipts) + 1)
        ):
            raise ValueError("confirmation receipt sequence is not canonical")
        staged_receipt_ids = [receipt.staged_tool_call_id for receipt in receipts]
        if len(staged_receipt_ids) != len(set(staged_receipt_ids)):
            raise ValueError("confirmation receipts repeat a staged write")
        consumed_receipt_ids = [
            receipt.consumed_tool_call_id
            for receipt in receipts
            if receipt.consumed_tool_call_id is not None
        ]
        if len(consumed_receipt_ids) != len(set(consumed_receipt_ids)):
            raise ValueError("confirmation receipts repeat an executed write")
        for receipt_index, receipt in enumerate(receipts):
            if receipt.schema_version != 1:
                raise ValueError("unsupported confirmation receipt schema")
            if receipt.conversation_id != self.conversation_id:
                raise ValueError("confirmation receipt conversation mismatch")
            require_hex_digest(
                receipt.action_sha256, label="confirmation action_sha256"
            )
            require_hex_digest(receipt.intent_id, label="confirmation intent_id")
            expected_intent_id = confirmation_intent_id(
                conversation_id=receipt.conversation_id,
                sequence=receipt.sequence,
                action_digest=receipt.action_sha256,
            )
            if receipt.intent_id != expected_intent_id:
                raise ValueError("confirmation intent identity drift")
            self._validate_confirmation_state_shape(receipt)
            staged = records.get(receipt.staged_tool_call_id)
            if staged is None or staged.execution_disposition != "prepared":
                raise ValueError("confirmation receipt has no prepared write record")
            if staged.name != receipt.tool_name or action_sha256(
                staged.name, staged.arguments
            ) != receipt.action_sha256:
                raise ValueError("prepared write and confirmation receipt drift")
            self._validate_confirmation_message_evidence(receipt, staged)

            if receipt.state == "consumed":
                if receipt.consumed_tool_call_id is None:
                    raise ValueError("consumed confirmation has no tool call")
                consumed = records.get(receipt.consumed_tool_call_id)
                if consumed is None or consumed.execution_disposition != "executed":
                    raise ValueError("consumed confirmation has no executed write")
                if consumed.name != receipt.tool_name or action_sha256(
                    consumed.name, consumed.arguments
                ) != receipt.action_sha256:
                    raise ValueError("executed write and confirmation receipt drift")
            elif (
                receipt.consumed_tool_call_id is not None
                or receipt.consumed_message_index is not None
            ):
                raise ValueError("unconsumed confirmation names an executed write")
            if receipt.replaced_by_intent_id is not None:
                require_hex_digest(
                    receipt.replaced_by_intent_id,
                    label="confirmation replaced_by_intent_id",
                )
                next_receipt = (
                    receipts[receipt_index + 1]
                    if receipt_index + 1 < len(receipts)
                    else None
                )
                if (
                    next_receipt is None
                    or next_receipt.intent_id != receipt.replaced_by_intent_id
                ):
                    raise ValueError("confirmation replacement identity drift")

        staged_call_ids = {
            receipt.staged_tool_call_id for receipt in receipts
        }
        prepared_call_ids = {
            record.call_id
            for record in self.tool_calls
            if record.execution_disposition == "prepared"
        }
        if staged_call_ids != prepared_call_ids:
            raise ValueError("prepared writes and confirmation receipts do not align")

        consumed_call_ids = {
            receipt.consumed_tool_call_id
            for receipt in receipts
            if receipt.state == "consumed"
        }
        executed_call_ids = {
            record.call_id
            for record in self.tool_calls
            if record.name in WRITE_TOOLS
            and record.execution_disposition == "executed"
        }
        if consumed_call_ids != executed_call_ids:
            raise ValueError("executed writes and consumed confirmations do not align")
        expected_summary: bool | None = True if executed_call_ids else None
        if self.mutation_confirmation_ok is not expected_summary:
            raise ValueError("mutation confirmation summary does not match receipts")

    @staticmethod
    def _validate_confirmation_state_shape(
        receipt: MutationConfirmationReceipt,
    ) -> None:
        has_proposal = receipt.proposal_message_index is not None
        has_decision = receipt.decision_message_index is not None
        has_consumption = (
            receipt.consumed_tool_call_id is not None
            or receipt.consumed_message_index is not None
        )
        if (receipt.proposal_message_sha256 is not None) != has_proposal:
            raise ValueError("confirmation proposal evidence is incomplete")
        if (receipt.decision_message_sha256 is not None) != has_decision:
            raise ValueError("confirmation decision evidence is incomplete")

        if receipt.state == "pending":
            if (
                has_decision
                or has_consumption
                or receipt.invalidation_reason is not None
                or receipt.replaced_by_intent_id is not None
            ):
                raise ValueError("pending confirmation has terminal evidence")
            return
        if receipt.state == "approved":
            if (
                not has_proposal
                or not has_decision
                or has_consumption
                or receipt.invalidation_reason is not None
                or receipt.replaced_by_intent_id is not None
            ):
                raise ValueError("approved confirmation state is malformed")
            return
        if receipt.state == "consumed":
            if (
                not has_proposal
                or not has_decision
                or not has_consumption
                or receipt.invalidation_reason is not None
                or receipt.replaced_by_intent_id is not None
            ):
                raise ValueError("consumed confirmation state is malformed")
            return
        if receipt.state == "superseded":
            if (
                receipt.invalidation_reason != "argument_change"
                or receipt.replaced_by_intent_id is None
                or has_consumption
            ):
                raise ValueError("superseded confirmation state is malformed")
            return
        if receipt.state == "invalidated":
            if (
                receipt.invalidation_reason is None
                or receipt.invalidation_reason == "argument_change"
                or receipt.replaced_by_intent_id is not None
                or has_consumption
            ):
                raise ValueError("invalidated confirmation state is malformed")
            if receipt.invalidation_reason == "missing_proposal" and (
                has_proposal or not has_decision
            ):
                raise ValueError("missing-proposal evidence is malformed")
            if receipt.invalidation_reason == "proposal_mismatch" and (
                has_proposal or has_decision
            ):
                raise ValueError("proposal-mismatch evidence is malformed")
            if receipt.invalidation_reason in {
                "non_assent",
                "post_confirmation_assistant_message",
                "post_confirmation_user_message",
            } and (not has_proposal or not has_decision):
                raise ValueError("post-proposal invalidation evidence is incomplete")

    def _validate_confirmation_message_evidence(
        self,
        receipt: MutationConfirmationReceipt,
        staged: ToolCallRecord,
    ) -> None:
        def message_at(index: int, *, label: str) -> dict[str, Any]:
            if index < 0 or index >= len(self.messages):
                raise ValueError(f"confirmation {label} index is out of range")
            message = self.messages[index]
            if not isinstance(message, dict):  # pragma: no cover - schema type
                raise ValueError(f"confirmation {label} message is malformed")
            return message

        def require_tool_call(index: int, call_id: str, *, label: str) -> None:
            message = message_at(index, label=label)
            if message.get("role") != "assistant":
                raise ValueError(f"confirmation {label} is not an assistant call")
            call_ids = {
                str(raw.get("id") or "")
                for raw in message.get("tool_calls") or []
                if isinstance(raw, dict)
            }
            if call_id not in call_ids:
                raise ValueError(f"confirmation {label} tool call is absent")

        require_tool_call(
            receipt.staged_message_index,
            receipt.staged_tool_call_id,
            label="staged",
        )

        proposal_index = receipt.proposal_message_index
        if proposal_index is None:
            if receipt.proposal_message_sha256 is not None:
                raise ValueError("confirmation proposal digest has no message")
        else:
            if proposal_index <= receipt.staged_message_index:
                raise ValueError("confirmation proposal is out of causal order")
            require_hex_digest(
                str(receipt.proposal_message_sha256 or ""),
                label="confirmation proposal_message_sha256",
            )
            proposal = message_at(proposal_index, label="proposal")
            proposal_text = proposal.get("content")
            if proposal.get("role") != "assistant" or not isinstance(
                proposal_text, str
            ) or not proposal_text.strip():
                raise ValueError("confirmation proposal is not visible assistant text")
            if digest_value(proposal_text) != receipt.proposal_message_sha256:
                raise ValueError("confirmation proposal digest drift")
            if not proposal_matches_action(
                proposal_text,
                tool_name=staged.name,
                arguments=staged.arguments,
            ):
                raise ValueError("confirmation proposal does not match the action")

        decision_index = receipt.decision_message_index
        if decision_index is None:
            if receipt.decision_message_sha256 is not None:
                raise ValueError("confirmation decision digest has no message")
        else:
            if decision_index <= (proposal_index or receipt.staged_message_index):
                raise ValueError("confirmation decision is out of causal order")
            require_hex_digest(
                str(receipt.decision_message_sha256 or ""),
                label="confirmation decision_message_sha256",
            )
            decision = message_at(decision_index, label="decision")
            decision_text = decision.get("content")
            if decision.get("role") != "user" or not isinstance(
                decision_text, str
            ):
                raise ValueError("confirmation decision is not visible user text")
            if digest_value(decision_text) != receipt.decision_message_sha256:
                raise ValueError("confirmation decision digest drift")
            if receipt.state in {"approved", "consumed"} and not (
                is_unqualified_assent(decision_text)
            ):
                raise ValueError("approved confirmation is not unqualified assent")
            if (
                receipt.invalidation_reason == "non_assent"
                and is_unqualified_assent(decision_text)
            ):
                raise ValueError("non-assent invalidation contains assent")

        if receipt.state == "consumed":
            if decision_index is None or receipt.consumed_message_index is None:
                raise ValueError("consumed confirmation lacks causal message evidence")
            if receipt.consumed_message_index <= decision_index:
                raise ValueError("confirmation execution is out of causal order")
            require_tool_call(
                receipt.consumed_message_index,
                str(receipt.consumed_tool_call_id or ""),
                label="consumed",
            )

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["bundle_sha256"] = ""
        return payload


class RunLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    study_id: str
    cohort: Literal["canary", "full"]
    preview_sha256: str
    planned_coordinate_ids: list[str]
    completed_coordinate_ids: list[str]
    unavailable: dict[str, str]
    observed_cost_usd: float = Field(ge=0)
    ambiguous_cost_reserve_usd: float = Field(default=0.0, ge=0)
    cost_exposure_upper_bound_usd: float | None = Field(default=None, ge=0)
    cost_uncertainty_reasons: list[str] = Field(default_factory=list)
    bundle_digests: dict[str, str]
    status: Literal["running", "paused", "complete", "blocked_budget", "uncertain"]
    ledger_sha256: str = ""

    @model_validator(mode="after")
    def validate_cost_exposure(self) -> "RunLedger":
        expected = round(
            self.observed_cost_usd + self.ambiguous_cost_reserve_usd,
            12,
        )
        if self.cost_exposure_upper_bound_usd is None:
            self.cost_exposure_upper_bound_usd = expected
        elif abs(self.cost_exposure_upper_bound_usd - expected) > 1e-12:
            raise ValueError(
                "cost exposure must equal observed cost plus ambiguous reserve"
            )
        if self.ambiguous_cost_reserve_usd and not self.cost_uncertainty_reasons:
            raise ValueError("ambiguous ledger cost requires a reason")
        if not self.ambiguous_cost_reserve_usd and self.cost_uncertainty_reasons:
            raise ValueError("ledger cost uncertainty reasons require a reserve")
        return self

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["ledger_sha256"] = ""
        return payload
