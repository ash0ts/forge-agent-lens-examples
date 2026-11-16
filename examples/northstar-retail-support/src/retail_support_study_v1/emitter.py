from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

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

from .canonical import digest_value, load_json, require_hex_digest, write_json_atomic
from .constants import DESTINATION_PROJECT, RECEIPTS_DIR
from .cost_policy import (
    bundle_has_disallowed_model_evidence,
    continues_frozen_attempt_failures,
)
from .runtime_snapshot import verify_runtime_snapshot
from .schemas import ConversationBundle, RunLedger
from .telemetry import (
    MAX_PUBLIC_ATTRIBUTE_CHARACTERS,
    EmissionCounts,
    build_public_payload,
    count_public_conversations,
    emit_public_conversation,
)

_INSTRUMENTATION_NAME = "retail-support-study"
_INSTRUMENTATION_VERSION = "2"
_SERVICE_NAME = "retail-support-agent"
AGENT_LENS_OTLP_ENDPOINT = "https://trace.wandb.ai/agents/otel/v1/traces"
_EXPORT_TIMEOUT_SECONDS = 30
_FORCE_FLUSH_TIMEOUT_MILLIS = 120_000
_BATCH_QUEUE_SIZE = 8_192
_BATCH_EXPORT_SIZE = 512
_BATCH_SCHEDULE_DELAY_MILLIS = 1_000
EMISSION_CONVERSATION_BATCH_SIZE = 32
AGENT_LENS_RESOURCE_ATTRIBUTES = {
    "service.name": _SERVICE_NAME,
    "service.namespace": "retail-support-study",
    "service.version": _INSTRUMENTATION_VERSION,
    "deployment.environment.name": "test",
}
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_frozen_bundles(
    ledger_path: Path, ledger: RunLedger
) -> dict[str, ConversationBundle]:
    bundle_dir = ledger_path.parent / "bundles"
    result: dict[str, ConversationBundle] = {}
    for coordinate_id in ledger.planned_coordinate_ids:
        if Path(coordinate_id).name != coordinate_id:
            raise ValueError(f"unsafe coordinate path component: {coordinate_id!r}")
        path = bundle_dir / f"{coordinate_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular frozen bundle: {coordinate_id}")
        result[coordinate_id] = ConversationBundle.model_validate(load_json(path))
    return result


def _load_emission_ledger(ledger_path: Path) -> RunLedger:
    """Load the ledger and fail closed before touching other frozen inputs."""

    ledger_path = Path(ledger_path)
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise ValueError("ledger path must be a regular frozen JSON file")
    ledger = RunLedger.model_validate(load_json(ledger_path))
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("emission ledger digest does not match its canonical payload")
    if ledger.status != "complete":
        raise ValueError("only a complete cohort can be emitted")
    return ledger


def _verify_emission_snapshot(
    ledger: RunLedger,
) -> tuple[str, Mapping[str, Any]]:
    """Authenticate execution evidence without conflating projection code.

    The ledger and retained runtime snapshot bind the exact code that executed
    the study. Emission is a later, read-only projection stage: its current
    implementation and visible output are independently bound by the exact
    ``public_payload_sha256`` approval. Requiring today's projection code to
    equal the historical execution tree would prevent a fail-closed projection
    fix from safely exporting already-sealed evidence.
    """

    runtime = verify_runtime_snapshot(
        ledger.preview_sha256,
        require_current_code=False,
    )
    if runtime.preview.get("study_id") != ledger.study_id:
        raise ValueError("runtime snapshot study identity mismatch")
    cohort = runtime.preview.get("cohorts", {}).get(ledger.cohort, {})
    expected = [str(row["coordinate_id"]) for row in cohort.get("coordinates", [])]
    if ledger.planned_coordinate_ids != expected:
        raise ValueError("emission ledger is not bound to the runtime snapshot plan")
    manifest_sha = str(runtime.manifest.get("manifest_sha256") or "")
    if len(manifest_sha) != 64:
        raise ValueError("runtime snapshot manifest has no digest")
    return manifest_sha, runtime.preview


def _validate_emission_cost_policy(
    ledger: RunLedger,
    bundles: Mapping[str, ConversationBundle],
    *,
    preview: Mapping[str, Any],
) -> None:
    """Recheck preview-scoped retry ambiguity before any network export."""

    ambiguous = round(
        sum(
            call.ambiguous_cost_reserve_usd
            for bundle in bundles.values()
            for call in bundle.model_calls
        ),
        12,
    )
    reasons = [
        reason
        for bundle in bundles.values()
        for call in bundle.model_calls
        for reason in call.cost_uncertainty_reasons
    ]
    if abs(ledger.ambiguous_cost_reserve_usd - ambiguous) > 1e-12:
        raise ValueError("emission retry reserve does not reconcile frozen calls")
    if ledger.cost_uncertainty_reasons != reasons:
        raise ValueError("emission retry reasons do not reconcile frozen calls")

    disallowed = [
        coordinate_id
        for coordinate_id, bundle in bundles.items()
        if bundle_has_disallowed_model_evidence(bundle)
    ]
    if disallowed:
        raise ValueError(
            "emission contains model evidence outside the approved retry policy: "
            f"{disallowed[:3]}"
        )
    if ambiguous and not continues_frozen_attempt_failures(preview):
        raise ValueError(
            "emission preview does not permit bounded retry cost ambiguity"
        )


def plan_frozen_emission(*, ledger_path: Path) -> dict[str, Any]:
    """Return the exact public emission identity without constructing an exporter."""

    ledger_path = Path(ledger_path)
    ledger = _load_emission_ledger(ledger_path)
    snapshot_sha, preview = _verify_emission_snapshot(ledger)
    bundles = _load_frozen_bundles(ledger_path, ledger)
    _validate_emission_cost_policy(ledger, bundles, preview=preview)
    payload = build_public_payload(ledger, bundles)
    conversations = payload["conversations"]
    turn_count = sum(len(item["turns"]) for item in conversations)
    return {
        "schema_version": 1,
        "destination_project": DESTINATION_PROJECT,
        "ledger_sha256": ledger.ledger_sha256,
        "public_payload_sha256": payload["payload_sha256"],
        "runtime_snapshot_manifest_sha256": snapshot_sha,
        "conversation_count": len(conversations),
        "turn_count": turn_count,
        "emission_mode": "batched_frozen_bundle_replay",
        "delivery_semantics": "at_least_once",
        "conversation_batch_size": EMISSION_CONVERSATION_BATCH_SIZE,
    }


def _reject_otlp_environment_overrides(environ: Mapping[str, str]) -> None:
    configured = sorted(
        name
        for name in _EXPORT_OVERRIDE_ENVIRONMENT
        if name in environ and environ[name].strip()
    )
    if configured:
        joined = ", ".join(configured)
        raise ValueError(
            f"Agent Lens emission rejects OTLP transport overrides; unset: {joined}"
        )


def build_agent_lens_otlp_exporter(
    *,
    api_key: str,
    environ: Mapping[str, str] | None = None,
    exporter_factory: Callable[..., SpanExporter] = OTLPSpanExporter,
) -> SpanExporter:
    """Build the one reviewed exporter configuration.

    There are deliberately no endpoint, project, or header parameters. Generic
    W&B entity/project variables are ignored; the destination and transport
    headers are fixed here. Known OpenTelemetry exporter overrides fail closed.
    """

    environment = os.environ if environ is None else environ
    _reject_otlp_environment_overrides(environment)
    if not api_key or api_key != api_key.strip() or "\n" in api_key or "\r" in api_key:
        raise ValueError("WANDB_API_KEY must be a non-empty single-line value")
    encoded = base64.b64encode(f"api:{api_key}".encode("utf-8")).decode("ascii")
    return exporter_factory(
        endpoint=AGENT_LENS_OTLP_ENDPOINT,
        headers={
            "Authorization": f"Basic {encoded}",
            "project_id": DESTINATION_PROJECT,
        },
        timeout=_EXPORT_TIMEOUT_SECONDS,
        compression=Compression.Gzip,
    )


def _receipt_identity(
    ledger_sha256: str,
    payload_sha256: str,
    runtime_snapshot_manifest_sha256: str,
    conversation_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "destination_project": DESTINATION_PROJECT,
        "ledger_sha256": ledger_sha256,
        "public_payload_sha256": payload_sha256,
        "runtime_snapshot_manifest_sha256": runtime_snapshot_manifest_sha256,
        "emission_mode": "batched_frozen_bundle_replay",
        "delivery_semantics": "at_least_once",
        "conversation_batch_size": EMISSION_CONVERSATION_BATCH_SIZE,
        "conversation_count": conversation_count,
    }


def _require_exact_approval(
    *,
    ledger_sha256: str,
    payload_sha256: str,
    approve_ledger_sha: str,
    approve_payload_sha: str,
) -> None:
    approved_ledger = require_hex_digest(approve_ledger_sha, label="approve-ledger-sha")
    approved_payload = require_hex_digest(
        approve_payload_sha, label="approve-payload-sha"
    )
    if approved_ledger != ledger_sha256:
        raise ValueError("approval does not match the exact frozen ledger")
    if approved_payload != payload_sha256:
        raise ValueError("approval does not match the exact public telemetry payload")


def _default_receipt_path(ledger: RunLedger) -> Path:
    return RECEIPTS_DIR / (
        f"agent-lens-emission-{ledger.cohort}-{ledger.ledger_sha256[:12]}.json"
    )


def _read_existing_receipt(
    path: Path, *, identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("emission receipt must be a regular file, not a symlink")
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError("emission receipt is malformed")
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise ValueError("existing receipt belongs to another frozen emission")
    status = value.get("status")
    if status not in {"running", "interrupted", "complete"}:
        raise ValueError("emission checkpoint has an unsupported status")
    next_index = value.get("next_conversation_index")
    if (
        type(next_index) is not int
        or next_index < 0
        or next_index > identity["conversation_count"]
        or value.get("completed_conversation_count") != next_index
    ):
        raise ValueError("emission checkpoint has invalid progress")
    for key in ("turn_count", "chat_span_count", "tool_span_count", "span_count"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError("emission checkpoint has invalid span counts")
    if status == "complete" and next_index != identity["conversation_count"]:
        raise ValueError("complete emission checkpoint is missing conversations")
    return value


def _write_exclusive_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


class _CheckedExporter(SpanExporter):
    """Make background batch outcomes observable to the checkpoint loop."""

    def __init__(self, delegate: SpanExporter) -> None:
        self.delegate = delegate
        self.failure: BaseException | str | None = None
        self.exported_span_count = 0

    def export(self, spans: Any) -> SpanExportResult:
        try:
            result = self.delegate.export(spans)
        except BaseException as exc:
            self.failure = exc
            raise
        if result is not SpanExportResult.SUCCESS:
            self.failure = f"export returned {result!r}"
        else:
            self.exported_span_count += len(spans)
        return result

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        method = getattr(self.delegate, "force_flush", None)
        if method is None:
            return True
        try:
            completed = bool(method(timeout_millis=timeout_millis))
        except TypeError:
            completed = bool(method(timeout_millis))
        except BaseException as exc:
            self.failure = exc
            raise
        if not completed:
            self.failure = "exporter force_flush timed out"
        return completed

    def shutdown(self) -> None:
        self.delegate.shutdown()


def _add_counts(left: EmissionCounts, right: EmissionCounts) -> EmissionCounts:
    return EmissionCounts(
        conversations=left.conversations + right.conversations,
        turns=left.turns + right.turns,
        chat_spans=left.chat_spans + right.chat_spans,
        tool_spans=left.tool_spans + right.tool_spans,
    )


def _progress_fields(
    *, next_conversation_index: int, counts: EmissionCounts
) -> dict[str, int]:
    return {
        "next_conversation_index": next_conversation_index,
        "completed_conversation_count": counts.conversations,
        "turn_count": counts.turns,
        "chat_span_count": counts.chat_spans,
        "tool_span_count": counts.tool_spans,
        "span_count": counts.total_spans,
    }


def _require_checkpoint_matches_prefix(
    checkpoint: Mapping[str, Any],
    *,
    counts: EmissionCounts,
) -> None:
    expected = _progress_fields(
        next_conversation_index=counts.conversations,
        counts=counts,
    )
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("emission checkpoint does not match the public payload prefix")


def emit_frozen_cohort(
    *,
    ledger_path: Path,
    approve_ledger_sha: str,
    approve_payload_sha: str,
    exporter: SpanExporter,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Emit one exact reviewed cohort, once, through an injected exporter.

    The caller owns exporter configuration and credentials. This function sees
    neither an endpoint nor a token; it only exports the canonical public plan.
    """

    ledger_path = Path(ledger_path)
    ledger = _load_emission_ledger(ledger_path)
    snapshot_sha, preview = _verify_emission_snapshot(ledger)
    bundles = _load_frozen_bundles(ledger_path, ledger)
    _validate_emission_cost_policy(ledger, bundles, preview=preview)
    payload = build_public_payload(ledger, bundles)

    _require_exact_approval(
        ledger_sha256=ledger.ledger_sha256,
        payload_sha256=payload["payload_sha256"],
        approve_ledger_sha=approve_ledger_sha,
        approve_payload_sha=approve_payload_sha,
    )

    if receipt_path is None:
        receipt_path = _default_receipt_path(ledger)
    receipt_path = Path(receipt_path)
    identity = _receipt_identity(
        ledger.ledger_sha256,
        payload["payload_sha256"],
        snapshot_sha,
        len(payload["conversations"]),
    )
    existing = _read_existing_receipt(receipt_path, identity=identity)
    conversations = payload["conversations"]
    if existing is not None and existing["status"] == "complete":
        _require_checkpoint_matches_prefix(
            existing,
            counts=count_public_conversations(conversations),
        )
        return existing

    if existing is None:
        progress_counts = EmissionCounts(0, 0, 0, 0)
        checkpoint = {
            **identity,
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            **_progress_fields(
                next_conversation_index=0,
                counts=progress_counts,
            ),
        }
        _write_exclusive_receipt(receipt_path, checkpoint)
    else:
        next_index = existing["next_conversation_index"]
        progress_counts = count_public_conversations(conversations[:next_index])
        _require_checkpoint_matches_prefix(existing, counts=progress_counts)
        checkpoint = existing

    next_index = checkpoint["next_conversation_index"]
    if next_index == len(conversations):
        exporter.shutdown()
        complete = {
            **checkpoint,
            "status": "complete",
            "updated_at": _now(),
            "completed_at": _now(),
        }
        write_json_atomic(receipt_path, complete)
        return complete

    checked_exporter = _CheckedExporter(exporter)
    provider: TracerProvider | None = None
    try:
        # Do not use Resource.create(): it runs environment detectors and can
        # attach unreviewed OTEL_RESOURCE_ATTRIBUTES outside the public payload.
        provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource(attributes=dict(AGENT_LENS_RESOURCE_ATTRIBUTES)),
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
        provider.add_span_processor(
            BatchSpanProcessor(
                checked_exporter,
                max_queue_size=_BATCH_QUEUE_SIZE,
                schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MILLIS,
                max_export_batch_size=_BATCH_EXPORT_SIZE,
                export_timeout_millis=_FORCE_FLUSH_TIMEOUT_MILLIS,
            )
        )
        tracer = provider.get_tracer(
            _INSTRUMENTATION_NAME,
            _INSTRUMENTATION_VERSION,
        )

        while next_index < len(conversations):
            stop_index = min(
                next_index + EMISSION_CONVERSATION_BATCH_SIZE,
                len(conversations),
            )
            conversation_batch = conversations[next_index:stop_index]
            batch_counts = count_public_conversations(conversation_batch)
            exported_before = checked_exporter.exported_span_count
            emitted = sum(
                emit_public_conversation(tracer, conversation)
                for conversation in conversation_batch
            )
            if emitted != batch_counts.total_spans:  # pragma: no cover - defensive
                raise AssertionError("telemetry span count drift")
            if not provider.force_flush(_FORCE_FLUSH_TIMEOUT_MILLIS):
                raise RuntimeError(
                    "telemetry provider did not flush before its deadline"
                )
            if checked_exporter.failure is not None:
                raise RuntimeError("telemetry export did not complete")
            exported = checked_exporter.exported_span_count - exported_before
            if exported != batch_counts.total_spans:
                raise RuntimeError(
                    "telemetry processor did not export the complete conversation batch"
                )

            progress_counts = _add_counts(progress_counts, batch_counts)
            next_index = stop_index
            checkpoint = {
                **identity,
                "status": "running",
                "started_at": checkpoint["started_at"],
                "updated_at": _now(),
                **_progress_fields(
                    next_conversation_index=next_index,
                    counts=progress_counts,
                ),
            }
            write_json_atomic(receipt_path, checkpoint)
    except BaseException as exc:
        if provider is not None:
            try:
                provider.shutdown()
            except BaseException:
                pass
        interrupted = {
            **checkpoint,
            "status": "interrupted",
            "updated_at": _now(),
            "failed_at": _now(),
            "error_type": type(exc).__name__,
        }
        write_json_atomic(receipt_path, interrupted)
        raise

    provider.shutdown()
    provider = None
    complete = {
        **checkpoint,
        "status": "complete",
        "updated_at": _now(),
        "completed_at": _now(),
    }
    write_json_atomic(receipt_path, complete)
    return complete


def emit_frozen_to_agent_lens(
    *,
    ledger_path: Path,
    approve_ledger_sha: str,
    approve_payload_sha: str,
    environ: Mapping[str, str] | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Emit an approved frozen cohort to the fixed Agent Lens destination."""

    plan = plan_frozen_emission(ledger_path=ledger_path)
    _require_exact_approval(
        ledger_sha256=plan["ledger_sha256"],
        payload_sha256=plan["public_payload_sha256"],
        approve_ledger_sha=approve_ledger_sha,
        approve_payload_sha=approve_payload_sha,
    )
    ledger = _load_emission_ledger(Path(ledger_path))
    resolved_receipt_path = (
        _default_receipt_path(ledger) if receipt_path is None else Path(receipt_path)
    )
    identity = _receipt_identity(
        plan["ledger_sha256"],
        plan["public_payload_sha256"],
        plan["runtime_snapshot_manifest_sha256"],
        plan["conversation_count"],
    )
    existing = _read_existing_receipt(resolved_receipt_path, identity=identity)
    if existing is not None and existing["status"] == "complete":
        return existing

    environment = os.environ if environ is None else environ
    api_key = environment.get("WANDB_API_KEY", "")
    exporter = build_agent_lens_otlp_exporter(
        api_key=api_key,
        environ=environment,
    )
    return emit_frozen_cohort(
        ledger_path=ledger_path,
        approve_ledger_sha=approve_ledger_sha,
        approve_payload_sha=approve_payload_sha,
        exporter=exporter,
        receipt_path=resolved_receipt_path,
    )
