from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from retail_support_study_v1 import cli, runner, verify
from retail_support_study_v1.canonical import digest_value, load_json
from retail_support_study_v1.constants import STUDY_ID, SUPPORT_MODEL
from retail_support_study_v1.pricing import CostBudget
from retail_support_study_v1.schemas import (
    ConversationBundle,
    Coordinate,
    ModelCall,
    Usage,
)


class _BudgetOnlyClient:
    def __init__(self, *, cost_cap_usd: float, **_kwargs: Any) -> None:
        self.budget = CostBudget(cost_cap_usd)


def _coordinates() -> list[dict[str, Any]]:
    return [
        Coordinate(
            coordinate_id=f"coordinate-{index}",
            logical_task_id=f"task-{index}",
            profile="ordinary",
            seed=100 + index,
            cohort="canary",
        ).model_dump(mode="json")
        for index in range(2)
    ]


def _preview(
    *,
    execution_generation: int = 1,
    continue_frozen_attempt_failures: bool = False,
) -> dict[str, Any]:
    preview = {
        "preview_sha256": "a" * 64,
        "cost_caps_usd": {"canary": 1.0, "full": 50.0},
        "cohorts": {"canary": {"coordinates": _coordinates()}},
    }
    if continue_frozen_attempt_failures:
        preview["run_policy"] = {
            "frozen_attempt_failure_policy": "continue_when_bundle_is_frozen",
            "bounded_retry_cost_ambiguity": (
                "track_upper_bound_and_continue"
            ),
        }
    if execution_generation > 1:
        preview["canary_execution"] = {
            "execution_generation": execution_generation,
        }
    return preview


def _configure_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "tasks.runtime.json").write_text(
        json.dumps(
            [
                {
                    "id": f"task-{index}",
                    "user_scenario": {"instructions": "Ask about an order."},
                }
                for index in range(2)
            ]
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    monkeypatch.setattr(runner, "RUNS_DIR", run_root)
    tasks = tuple(json.loads((prepared / "tasks.runtime.json").read_text()))
    runtime = SimpleNamespace(
        preview=_preview(),
        manifest={"manifest_sha256": "f" * 64},
        tasks=tasks,
        pricing_lock={"models": {}},
    )
    monkeypatch.setattr(runner, "verify_runtime_snapshot", lambda _sha: runtime)
    monkeypatch.setattr(runner, "WandbInferenceClient", _BudgetOnlyClient)
    monkeypatch.setattr(verify, "RUNS_DIR", run_root)
    monkeypatch.setattr(verify, "verify_preview", _preview)
    monkeypatch.setattr(verify, "verify_runtime_snapshot", lambda _sha: runtime)
    return run_root


def _configure_relaxed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    run_root = _configure_run(tmp_path, monkeypatch)
    relaxed_preview = _preview(continue_frozen_attempt_failures=True)
    runtime = runner.verify_runtime_snapshot("a" * 64)
    relaxed_runtime = SimpleNamespace(
        preview=relaxed_preview,
        manifest=runtime.manifest,
        tasks=runtime.tasks,
        pricing_lock=runtime.pricing_lock,
    )
    monkeypatch.setattr(
        runner,
        "verify_runtime_snapshot",
        lambda _sha: relaxed_runtime,
    )
    monkeypatch.setattr(verify, "verify_preview", lambda: relaxed_preview)
    monkeypatch.setattr(
        verify,
        "verify_runtime_snapshot",
        lambda _sha: relaxed_runtime,
    )
    return run_root


def _configure_replacement_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution_generation: int = 2,
) -> Path:
    run_root = _configure_run(tmp_path, monkeypatch)
    replacement_preview = _preview(execution_generation=execution_generation)
    runtime = runner.verify_runtime_snapshot("a" * 64)
    replacement_runtime = SimpleNamespace(
        preview=replacement_preview,
        manifest=runtime.manifest,
        tasks=runtime.tasks,
        pricing_lock=runtime.pricing_lock,
    )
    monkeypatch.setattr(
        runner,
        "verify_runtime_snapshot",
        lambda _sha: replacement_runtime,
    )
    monkeypatch.setattr(verify, "verify_preview", lambda: replacement_preview)
    monkeypatch.setattr(
        verify,
        "verify_runtime_snapshot",
        lambda _sha: replacement_runtime,
    )
    return run_root


def _bundle(
    coordinate: Coordinate,
    *,
    observed_cost: float = 0.1,
    ambiguous_cost: float = 0.0,
    response_validation_error: str | None = None,
) -> ConversationBundle:
    reasons = (
        [
            "retryable transport outcome was ambiguous: attempt=1, "
            "error=InternalServerError"
        ]
        if ambiguous_cost
        else []
    )
    call = ModelCall(
        call_id=f"call-{coordinate.coordinate_id}",
        participant="support_agent",
        requested_model=SUPPORT_MODEL,
        returned_model=SUPPORT_MODEL,
        response_id=f"response-{coordinate.coordinate_id}",
        finish_reason="stop",
        ambiguous_cost_reserve_usd=ambiguous_cost,
        cost_uncertainty_reasons=reasons,
        response_validation_error=response_validation_error,
        temperature=0.0,
        requested_max_tokens=4_096,
        input_messages=[{"role": "user", "content": "Where is my order?"}],
        output_message={"role": "assistant", "content": "It is in transit."},
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            observed_cost_usd=observed_cost,
        ),
    )
    is_uncertain = bool(ambiguous_cost or response_validation_error)
    bundle = ConversationBundle(
        study_id=STUDY_ID,
        conversation_id=f"conversation-{coordinate.coordinate_id}",
        coordinate=coordinate,
        requested_models={"support_agent": SUPPORT_MODEL},
        returned_models={"support_agent": [SUPPORT_MODEL]},
        status="unavailable" if is_uncertain else "complete",
        unavailable_reason=("model evidence is uncertain" if is_uncertain else None),
        messages=[
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "It is in transit."},
        ],
        model_calls=[call],
        tool_calls=[],
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            observed_cost_usd=observed_cost,
        ),
        mutation_confirmation_ok=None,
        terminal_response=not is_uncertain,
    )
    bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
    return bundle


@pytest.mark.parametrize("entry_kind", ["file", "directory", "symlink"])
def test_replacement_canary_rejects_any_imported_artifact_without_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    run_root = _configure_replacement_run(tmp_path, monkeypatch)
    cohort_dir = run_root / "canary" / ("a" * 64)
    cohort_dir.mkdir(parents=True)
    imported = cohort_dir / "bundles"
    if entry_kind == "file":
        imported.write_text("copied prior bundle", encoding="utf-8")
    elif entry_kind == "directory":
        imported.mkdir()
        (imported / "coordinate-0.json").write_text("{}", encoding="utf-8")
    else:
        target = tmp_path / "prior-bundles"
        target.mkdir()
        imported.symlink_to(target, target_is_directory=True)

    executed: list[str] = []
    monkeypatch.setattr(
        runner,
        "run_coordinate",
        lambda coordinate, **_kwargs: executed.append(coordinate.coordinate_id),
    )

    with pytest.raises(ValueError, match="requires a fresh run directory"):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert executed == []


@pytest.mark.parametrize("execution_generation", [2, 3, 4, 5, 6])
def test_replacement_canary_allows_empty_directory_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_generation: int,
) -> None:
    run_root = _configure_replacement_run(
        tmp_path,
        monkeypatch,
        execution_generation=execution_generation,
    )
    cohort_dir = run_root / "canary" / ("a" * 64)
    cohort_dir.mkdir(parents=True)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)

    first = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )
    assert first.status == "paused"
    assert executed == ["coordinate-0"]

    resumed = runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert resumed.status == "complete"
    assert executed == ["coordinate-0", "coordinate-1"]


@pytest.mark.parametrize("execution_generation", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("missing", ["bundle-and-attempt", "attempt"])
def test_replacement_resume_never_regenerates_missing_paid_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    execution_generation: int,
) -> None:
    run_root = _configure_replacement_run(
        tmp_path,
        monkeypatch,
        execution_generation=execution_generation,
    )
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )
    assert executed == ["coordinate-0"]

    cohort_dir = run_root / "canary" / ("a" * 64)
    (cohort_dir / "attempts" / "coordinate-0.json").unlink()
    if missing == "bundle-and-attempt":
        (cohort_dir / "bundles" / "coordinate-0.json").unlink()

    def forbid_client(**_kwargs: Any) -> _BudgetOnlyClient:
        raise AssertionError("resume reached client construction")

    monkeypatch.setattr(runner, "WandbInferenceClient", forbid_client)
    message = (
        "ledger bundle digests do not match"
        if missing == "bundle-and-attempt"
        else "attempt files do not match"
    )
    with pytest.raises(ValueError, match=message):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert executed == ["coordinate-0"]


def test_replacement_resume_rejects_ledger_outcome_drift_before_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_replacement_run(tmp_path, monkeypatch)

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )
    ledger_path = run_root / "canary" / ("a" * 64) / "ledger.json"
    ledger = load_json(ledger_path)
    ledger["completed_coordinate_ids"] = []
    ledger["ledger_sha256"] = ""
    ledger["ledger_sha256"] = digest_value(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    def forbid_client(**_kwargs: Any) -> _BudgetOnlyClient:
        raise AssertionError("resume reached client construction")

    monkeypatch.setattr(runner, "WandbInferenceClient", forbid_client)
    with pytest.raises(ValueError, match="completed coordinates do not match"):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("study_id", "other-study", "run ledger identity mismatch"),
        ("preview_sha256", "b" * 64, "run ledger preview mismatch"),
        (
            "planned_coordinate_ids",
            ["coordinate-1", "coordinate-0"],
            "run ledger planned coordinates mismatch",
        ),
    ],
)
def test_replacement_resume_validates_ledger_identity_before_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    run_root = _configure_replacement_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )
    assert executed == ["coordinate-0"]

    ledger_path = run_root / "canary" / ("a" * 64) / "ledger.json"
    ledger = load_json(ledger_path)
    ledger[field] = replacement
    ledger["ledger_sha256"] = ""
    ledger["ledger_sha256"] = digest_value(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert executed == ["coordinate-0"]


def test_replacement_resume_rejects_ledger_digest_drift_before_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_replacement_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )

    ledger_path = run_root / "canary" / ("a" * 64) / "ledger.json"
    ledger = load_json(ledger_path)
    ledger["status"] = "running"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="run ledger content digest mismatch"):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert executed == ["coordinate-0"]


def test_replacement_resume_requires_a_regular_ledger_before_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_replacement_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )

    ledger_path = run_root / "canary" / ("a" * 64) / "ledger.json"
    ledger_path.unlink()
    ledger_path.mkdir()

    with pytest.raises(ValueError, match="run ledger must be a regular file"):
        runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert executed == ["coordinate-0"]


def test_paused_checkpoint_resumes_with_exact_reconstructed_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)

    first = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        max_new_coordinates=1,
    )
    assert first.status == "paused"
    assert first.observed_cost_usd == 0.1
    assert executed == ["coordinate-0"]

    resumed = runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert resumed.status == "complete"
    assert resumed.completed_coordinate_ids == ["coordinate-0", "coordinate-1"]
    assert resumed.observed_cost_usd == 0.2
    assert resumed.ambiguous_cost_reserve_usd == 0
    assert executed == ["coordinate-0", "coordinate-1"]

    verified = verify.verify_cohort("canary")
    assert verified["status"] == "complete"
    assert verified["observed_cost_usd"] == 0.2


def test_ambiguous_bundle_stops_cohort_and_resume_makes_no_new_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        client.budget.record_ambiguous(
            0.2,
            reason=(
                "retryable transport outcome was ambiguous: attempt=1, "
                "error=InternalServerError"
            ),
        )
        return _bundle(coordinate, ambiguous_cost=0.2)

    monkeypatch.setattr(runner, "run_coordinate", execute)

    first = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        concurrency=1,
    )
    assert first.status == "uncertain"
    assert first.observed_cost_usd == 0.1
    assert first.ambiguous_cost_reserve_usd == 0.2
    assert first.cost_exposure_upper_bound_usd == 0.3
    assert executed == ["coordinate-0"]

    resumed = runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert resumed.status == "uncertain"
    assert resumed.observed_cost_usd == 0.1
    assert resumed.ambiguous_cost_reserve_usd == 0.2
    assert executed == ["coordinate-0"]

    verified = verify.verify_cohort("canary")
    assert verified["status"] == "uncertain"
    assert verified["cost_exposure_upper_bound_usd"] == 0.3


def test_relaxed_test_project_policy_continues_after_frozen_cost_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_relaxed_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        client.budget.record_ambiguous(
            0.02,
            reason=(
                "retryable transport outcome was ambiguous: attempt=1, "
                "error=InternalServerError"
            ),
        )
        bundle = _bundle(coordinate, ambiguous_cost=0.02)
        bundle.model_calls[0].returned_model = ""
        bundle.model_calls[0].finish_reason = "transport_error"
        bundle.model_calls[0].response_validation_error = (
            "Forge Inference request unavailable after 3 attempts: "
            "InternalServerError"
        )
        bundle.bundle_sha256 = digest_value(bundle.unsigned_payload())
        return bundle

    monkeypatch.setattr(runner, "run_coordinate", execute)

    ledger = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        concurrency=1,
    )

    assert ledger.status == "complete"
    assert ledger.completed_coordinate_ids == []
    assert list(ledger.unavailable) == ["coordinate-0", "coordinate-1"]
    assert ledger.observed_cost_usd == 0.2
    assert ledger.ambiguous_cost_reserve_usd == 0.04
    assert ledger.cost_exposure_upper_bound_usd == 0.24
    assert executed == ["coordinate-0", "coordinate-1"]
    assert verify.verify_cohort("canary")["status"] == "complete"


def test_relaxed_test_project_policy_continues_after_frozen_length_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_relaxed_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        client.budget.record(0.1)
        if coordinate.coordinate_id == "coordinate-0":
            return _bundle(
                coordinate,
                response_validation_error=(
                    "Forge Inference returned an unusable finish reason: "
                    "finish_reason='length'"
                ),
            )
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)

    ledger = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        concurrency=1,
    )

    assert ledger.status == "complete"
    assert ledger.completed_coordinate_ids == ["coordinate-1"]
    assert list(ledger.unavailable) == ["coordinate-0"]
    assert executed == ["coordinate-0", "coordinate-1"]
    assert verify.verify_cohort("canary")["status"] == "complete"


def test_parallel_run_is_bounded_isolated_ordered_and_cost_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_run(tmp_path, monkeypatch)
    coordinates = [
        Coordinate(
            coordinate_id=f"coordinate-{index}",
            logical_task_id=f"task-{index}",
            profile="ordinary",
            trial_index=index,
            seed=100 + index,
            cohort="canary",
        )
        for index in range(6)
    ]
    preview = {
        "preview_sha256": "a" * 64,
        "cost_caps_usd": {"canary": 1.0},
        "cohorts": {
            "canary": {
                "coordinates": [
                    coordinate.model_dump(mode="json")
                    for coordinate in coordinates
                ]
            }
        },
    }
    runtime = SimpleNamespace(
        preview=preview,
        manifest={"manifest_sha256": "f" * 64},
        tasks=tuple(
            {
                "id": f"task-{index}",
                "user_scenario": {"instructions": "Ask about an order."},
            }
            for index in range(6)
        ),
        pricing_lock={"models": {}},
    )
    monkeypatch.setattr(runner, "verify_runtime_snapshot", lambda _sha: runtime)

    created_clients: list[_BudgetOnlyClient] = []

    def client_factory(**kwargs: Any) -> _BudgetOnlyClient:
        client = _BudgetOnlyClient(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setattr(runner, "WandbInferenceClient", client_factory)

    main_thread_id = threading.get_ident()
    worker_thread_ids: set[int] = set()
    state_lock = threading.Lock()
    first_wave_barrier = threading.Barrier(3)
    active = 0
    maximum_active = 0
    used_client_ids: list[int] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        nonlocal active, maximum_active
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        index = int(coordinate.coordinate_id.rsplit("-", 1)[1])
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            worker_thread_ids.add(threading.get_ident())
            used_client_ids.append(id(client))
        try:
            if index < 3:
                first_wave_barrier.wait(timeout=2)
            observed_cost = round((index + 1) / 100, 12)
            client.budget.record(observed_cost)
            return _bundle(coordinate, observed_cost=observed_cost)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(runner, "run_coordinate", execute)

    original_write_json_atomic = runner.write_json_atomic
    original_write_exclusive = runner._write_exclusive

    def write_json_atomic_from_coordinator(path: Path, value: Any) -> None:
        assert threading.get_ident() == main_thread_id
        original_write_json_atomic(path, value)

    def write_exclusive_from_coordinator(path: Path, value: Any) -> None:
        assert threading.get_ident() == main_thread_id
        original_write_exclusive(path, value)

    monkeypatch.setattr(runner, "write_json_atomic", write_json_atomic_from_coordinator)
    monkeypatch.setattr(runner, "_write_exclusive", write_exclusive_from_coordinator)

    ledger = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        concurrency=3,
    )

    assert ledger.status == "complete"
    assert ledger.completed_coordinate_ids == [
        coordinate.coordinate_id for coordinate in coordinates
    ]
    assert list(ledger.bundle_digests) == ledger.completed_coordinate_ids
    assert ledger.observed_cost_usd == 0.21
    assert ledger.ambiguous_cost_reserve_usd == 0
    assert maximum_active == 3
    assert main_thread_id not in worker_thread_ids
    assert len(created_clients) == len(coordinates)
    assert len(set(used_client_ids)) == len(coordinates)
    assert (run_root / "canary" / ("a" * 64) / "ledger.json").is_file()


def test_parallel_worker_failure_reserves_cap_and_resume_never_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_run(tmp_path, monkeypatch)
    executed: list[str] = []

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        executed.append(coordinate.coordinate_id)
        if coordinate.coordinate_id == "coordinate-0":
            raise RuntimeError("worker boundary failed")
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)

    first = runner.run_cohort(
        "canary",
        approve_preview_sha="a" * 64,
        concurrency=2,
    )
    assert first.status == "uncertain"
    assert first.completed_coordinate_ids == ["coordinate-1"]
    assert "prior paid attempt has no frozen bundle" in first.unavailable[
        "coordinate-0"
    ]
    assert first.observed_cost_usd == 0.1
    assert first.ambiguous_cost_reserve_usd == 0.9
    assert first.cost_exposure_upper_bound_usd == 1.0
    assert set(executed) == {"coordinate-0", "coordinate-1"}
    assert verify.verify_cohort("canary")["status"] == "uncertain"

    resumed = runner.run_cohort("canary", approve_preview_sha="a" * 64)
    assert resumed.status == "uncertain"
    assert resumed.completed_coordinate_ids == ["coordinate-1"]
    assert resumed.cost_exposure_upper_bound_usd == 1.0
    assert set(executed) == {"coordinate-0", "coordinate-1"}
    assert verify.verify_cohort("canary")["status"] == "uncertain"


def test_run_cli_forwards_configurable_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_cohort(cohort: str, **kwargs: Any) -> Any:
        captured.update({"cohort": cohort, **kwargs})
        return SimpleNamespace(model_dump=lambda: {"status": "complete"})

    monkeypatch.setattr(runner, "run_cohort", fake_run_cohort)

    assert (
        cli.main(
            [
                "run",
                "--cohort",
                "canary",
                "--approve-preview-sha",
                "a" * 64,
                "--concurrency",
                "3",
            ]
        )
        == 0
    )
    assert captured == {
        "cohort": "canary",
        "approve_preview_sha": "a" * 64,
        "max_new_coordinates": None,
        "concurrency": 3,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


def test_verify_normalizes_harmless_bundle_cost_float_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_run(tmp_path, monkeypatch)

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort("canary", approve_preview_sha="a" * 64)
    cohort_dir = run_root / "canary" / ("a" * 64)
    bundle_path = cohort_dir / "bundles" / "coordinate-0.json"
    attempt_path = cohort_dir / "attempts" / "coordinate-0.json"
    ledger_path = cohort_dir / "ledger.json"

    def set_bundle_cost(value: float) -> None:
        bundle = load_json(bundle_path)
        bundle["usage"]["observed_cost_usd"] = value
        bundle["bundle_sha256"] = ""
        bundle["bundle_sha256"] = digest_value(bundle)
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

        attempt = load_json(attempt_path)
        attempt["bundle_sha256"] = bundle["bundle_sha256"]
        attempt["attempt_sha256"] = ""
        attempt["attempt_sha256"] = digest_value(attempt)
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

        ledger = load_json(ledger_path)
        ledger["bundle_digests"]["coordinate-0"] = bundle["bundle_sha256"]
        ledger["ledger_sha256"] = ""
        ledger["ledger_sha256"] = digest_value(ledger)
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    set_bundle_cost(0.10000000000000002)
    assert verify.verify_cohort("canary")["status"] == "complete"

    set_bundle_cost(0.1000000001)
    with pytest.raises(ValueError, match="bundle observed cost mismatch"):
        verify.verify_cohort("canary")


def test_verify_rejects_ledger_digest_and_attempt_cost_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _configure_run(tmp_path, monkeypatch)

    def execute(
        coordinate: Coordinate,
        *,
        client: _BudgetOnlyClient,
        tasks_by_id: dict[str, dict[str, Any]],
        runtime: Any,
    ) -> ConversationBundle:
        assert coordinate.logical_task_id in tasks_by_id
        assert runtime.manifest["manifest_sha256"] == "f" * 64
        client.budget.record(0.1)
        return _bundle(coordinate)

    monkeypatch.setattr(runner, "run_coordinate", execute)
    runner.run_cohort("canary", approve_preview_sha="a" * 64)
    cohort_dir = run_root / "canary" / ("a" * 64)
    ledger_path = cohort_dir / "ledger.json"
    original_ledger = load_json(ledger_path)
    ledger = dict(original_ledger)
    ledger["completed_coordinate_ids"] = []
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="ledger content digest"):
        verify.verify_cohort("canary")

    ledger_path.write_text(json.dumps(original_ledger), encoding="utf-8")
    runner.run_cohort("canary", approve_preview_sha="a" * 64)
    attempt_path = cohort_dir / "attempts" / "coordinate-0.json"
    attempt = load_json(attempt_path)
    attempt["observed_cost_usd"] = 0.9
    attempt["attempt_sha256"] = ""
    attempt["attempt_sha256"] = digest_value(attempt)
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

    with pytest.raises(ValueError, match="attempt/bundle observed_cost"):
        verify.verify_cohort("canary")
