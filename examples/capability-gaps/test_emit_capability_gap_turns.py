from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "capability_gap_emitter", ROOT / "emit_capability_gap_turns.py"
)
assert SPEC and SPEC.loader
emitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = emitter
SPEC.loader.exec_module(emitter)


def _loaded():
    return emitter.load_manifest(ROOT / "corpus_manifest.json")


def _payload():
    manifest, digest = _loaded()
    return (
        manifest,
        digest,
        emitter.build_payload(manifest, digest, f"capability-gap-v1-{digest[:12]}"),
    )


def test_frozen_manifest_has_exact_distribution() -> None:
    manifest, _ = _loaded()
    assert manifest["planned_turns"] == 180
    assert {item["id"]: item["count"] for item in manifest["families"]} == {
        "helpful_read_only_restart_limit": 36,
        "bare_restart_limit": 24,
        "supported_diagnostic": 30,
        "false_gap_available_action": 24,
        "permission_bound_restart": 30,
        "deliberate_safety_boundary": 18,
        "unrelated_supported_control": 18,
    }
    scenarios = emitter.expand_scenarios(manifest)
    assert len(scenarios) == 180
    assert sum(len(item.tool_calls) for item in scenarios) == 378


def test_payload_identity_is_deterministic_and_unique() -> None:
    manifest, digest, first = _payload()
    second = emitter.build_payload(manifest, digest, f"capability-gap-v1-{digest[:12]}")
    assert emitter.canonical_json(first) == emitter.canonical_json(second)
    conversation_ids = [turn["conversation_id"] for turn in first["turns"]]
    trace_ids = [turn["spans"][0]["trace_id"] for turn in first["turns"]]
    span_ids = [span["span_id"] for turn in first["turns"] for span in turn["spans"]]
    assert len(conversation_ids) == len(set(conversation_ids)) == 180
    assert len(trace_ids) == len(set(trace_ids)) == 180
    assert len(span_ids) == len(set(span_ids)) == 558


def test_every_span_has_required_genai_attributes() -> None:
    _, _, payload = _payload()
    for turn in payload["turns"]:
        root = turn["spans"][0]
        root_attrs = root["attributes"]
        assert root_attrs["gen_ai.operation.name"] == "invoke_agent"
        assert root_attrs["gen_ai.agent.name"]
        assert root_attrs["gen_ai.agent.version"]
        assert root_attrs["gen_ai.request.model"]
        assert root_attrs["gen_ai.conversation.id"] == turn["conversation_id"]
        assert json.loads(root_attrs["gen_ai.input.messages"])[0]["role"] == "user"
        assert (
            json.loads(root_attrs["gen_ai.output.messages"])[0]["role"] == "assistant"
        )
        for tool in turn["spans"][1:]:
            attrs = tool["attributes"]
            assert attrs["gen_ai.operation.name"] == "execute_tool"
            assert attrs["gen_ai.conversation.id"] == turn["conversation_id"]
            assert attrs["gen_ai.tool.name"]
            assert attrs["gen_ai.tool.call.id"]
            assert isinstance(json.loads(attrs["gen_ai.tool.call.arguments"]), dict)
            assert isinstance(json.loads(attrs["gen_ai.tool.call.result"]), dict)


def test_private_review_expectations_never_enter_payload_or_receipt() -> None:
    manifest, digest, payload = _payload()
    receipt = emitter._receipt(
        payload, digest, f"capability-gap-v1-{digest[:12]}", "dry_run"
    )
    public_text = (
        emitter.canonical_json(payload) + emitter.canonical_json(receipt)
    ).decode()
    for family in manifest["families"]:
        assert family["id"] not in public_text
        assert family["review_expectation"] not in public_text
    forbidden = (
        "WANDB_API_KEY",
        "authorization",
        "bearer ",
        "api_key",
        "customer",
        "/Users/",
        "/private/",
        "localhost",
        "internal",
        "expected_review",
    )
    lowered = public_text.lower()
    for value in forbidden:
        assert value.lower() not in lowered


def test_dry_run_never_exports_and_receipt_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"

    def forbidden_export(_payload):
        raise AssertionError("dry run attempted a network export")

    monkeypatch.setattr(emitter, "export_payload", forbidden_export)
    assert emitter.main(["--dry-run", "--receipt", str(receipt)]) == 0
    first = receipt.read_bytes()
    assert emitter.main(["--dry-run", "--receipt", str(receipt)]) == 0
    assert receipt.read_bytes() == first
    value = json.loads(first)
    assert value["status"] == "dry_run"
    assert value["planned_turns"] == 180
    assert value["planned_spans"] == 558
    assert value["exported_turns"] == value["exported_spans"] == 0


def test_export_requires_exact_destination_and_explicit_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "FORGE_TRACE_PROJECT",
        "WANDB_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="FORGE_TRACE_PROJECT"):
        emitter.main(["--receipt", str(tmp_path / "receipt.json")])
    monkeypatch.setenv("FORGE_TRACE_PROJECT", emitter.EXPECTED_PROJECT)
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        emitter.main(["--receipt", str(tmp_path / "receipt.json")])


def test_reviewed_dry_run_can_advance_once_to_exported_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    assert emitter.main(["--dry-run", "--receipt", str(receipt)]) == 0
    observed: list[int] = []

    def offline_export(payload, api_key):
        observed.append(len(payload["turns"]))
        assert api_key == "placeholder-test-key"

    monkeypatch.setattr(emitter, "export_payload", offline_export)
    monkeypatch.setenv("FORGE_TRACE_PROJECT", emitter.EXPECTED_PROJECT)
    monkeypatch.setenv("WANDB_API_KEY", "placeholder-test-key")
    assert emitter.main(["--receipt", str(receipt)]) == 0
    assert observed == [180]
    exported = json.loads(receipt.read_text())
    assert exported["status"] == "exported"
    assert exported["exported_turns"] == 180
    assert exported["exported_spans"] == 558
    assert "placeholder-test-key" not in receipt.read_text()

    def unexpected_second_export(_payload, _api_key):
        raise AssertionError("an exported receipt did not suppress a duplicate export")

    monkeypatch.setattr(emitter, "export_payload", unexpected_second_export)
    monkeypatch.delenv("FORGE_TRACE_PROJECT")
    monkeypatch.delenv("WANDB_API_KEY")
    assert emitter.main(["--receipt", str(receipt)]) == 0
    assert json.loads(receipt.read_text()) == exported


def test_manifest_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "corpus_manifest.json"
    changed.write_text((ROOT / "corpus_manifest.json").read_text() + "\n")
    (tmp_path / "corpus_manifest.sha256").write_text(
        (ROOT / "corpus_manifest.sha256").read_text()
    )
    with pytest.raises(ValueError, match="manifest digest drift"):
        emitter.load_manifest(changed)


def test_v2_manifest_exceeds_insights_distinct_wording_floor() -> None:
    manifest, _ = emitter.load_manifest(ROOT / "corpus_manifest_v2.json")
    assert manifest["planned_turns"] == 240
    scenarios = emitter.expand_scenarios(manifest)
    assert len(scenarios) == 240
    limit_answers = {
        s.final_answer
        for s in scenarios[:100]
    }
    assert len(limit_answers) >= 20
