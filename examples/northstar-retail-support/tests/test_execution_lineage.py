from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from retail_support_study_v1 import verify
from retail_support_study_v1.canonical import digest_value
from retail_support_study_v1.constants import STUDY_ID


def _preview(generation: int) -> dict[str, Any]:
    supersedes: dict[str, Any] = {
        "ambiguous_cost_reserve_usd": 0.02,
        "completed_coordinate_count": 15,
        "cost_exposure_upper_bound_usd": 0.2,
        "ledger_file_sha256": "a" * 64,
        "ledger_sha256": "b" * 64,
        "observed_cost_usd": 0.18,
        "planned_coordinate_count": 36,
        "preview_sha256": "c" * 64,
        "runtime_manifest_file_sha256": "d" * 64,
        "runtime_manifest_sha256": "e" * 64,
        "status": "uncertain",
        "unavailable": {
            "coordinate-15": (
                "InferenceCallError: request unavailable after 3 attempts: "
                "InternalServerError"
            )
        },
        "unavailable_coordinate_count": 1,
        "unavailable_coordinate_ids": ["coordinate-15"],
    }
    if generation == 4:
        supersedes.update(
            {
                "ambiguous_cost_reserve_usd": 0.0,
                "completed_coordinate_count": 5,
                "cost_exposure_upper_bound_usd": 0.08665227,
                "observed_cost_usd": 0.08665227,
                "status": "paused",
                "unavailable": {
                    "coordinate-5": (
                        "RuntimeError: simulator failed the bounded "
                        "scenario-conformance repair"
                    )
                },
                "unavailable_coordinate_count": 1,
                "unavailable_coordinate_ids": ["coordinate-5"],
            }
        )
    elif generation == 5:
        supersedes.update(
            {
                "ambiguous_cost_reserve_usd": 0.00285705,
                "completed_coordinate_count": 1,
                "cost_exposure_upper_bound_usd": 0.03007249,
                "observed_cost_usd": 0.02721544,
                "unavailable": {
                    "coordinate-1": (
                        "InferenceCallError: request unavailable after 3 "
                        "attempts: InternalServerError"
                    )
                },
                "unavailable_coordinate_count": 1,
                "unavailable_coordinate_ids": ["coordinate-1"],
            }
        )
    elif generation == 6:
        supersedes.update(
            {
                "ambiguous_cost_reserve_usd": 0.0,
                "completed_coordinate_count": 29,
                "cost_exposure_upper_bound_usd": 0.32820088,
                "observed_cost_usd": 0.32820088,
                "status": "paused",
                "unavailable": {
                    "coordinate-29": (
                        "RuntimeError: simulator failed the bounded "
                        "scenario-conformance repair"
                    )
                },
                "unavailable_coordinate_count": 1,
                "unavailable_coordinate_ids": ["coordinate-29"],
            }
        )
    if generation > 2:
        supersedes.update(
            {
                "execution_generation": generation - 1,
                "execution_lock_file_sha256": "2" * 64,
                "execution_lock_sha256": "1" * 64,
                "cost_uncertainty_reasons": {
                    3: [
                        "retryable transport outcome was ambiguous: "
                        "attempt=1, error=APITimeoutError"
                    ],
                    4: [],
                    5: [
                        "retryable transport outcome was ambiguous: "
                        "attempt=1, error=InternalServerError",
                        "retryable transport outcome was ambiguous: "
                        "attempt=2, error=InternalServerError",
                        "retryable transport outcome was ambiguous: "
                        "attempt=3, error=InternalServerError",
                    ],
                    6: [],
                }[generation],
            }
        )
    modes = {
        2: "full_rerun_after_uncertain",
        3: "full_rerun_after_uncertain",
        4: "full_rerun_after_nonqualifying_partial",
        5: "full_rerun_after_uncertain",
        6: "full_rerun_after_nonqualifying_partial",
    }
    reasons = {
        2: "provider_internal_server_error_after_bounded_retries",
        3: "provider_timeout_created_ambiguous_cost_reserve",
        4: "simulator_contract_false_negative_after_visible_correction",
        5: "provider_internal_server_error_after_bounded_retries",
        6: "simulator_conditional_clause_revealed_before_confirmation_boundary",
    }
    execution = {
        "cohort": "canary",
        "execution_generation": generation,
        "lock_sha256": "",
        "mode": modes[generation],
        "reason": reasons[generation],
        "replacement_policy": {
            "planned_coordinate_count": 36,
            "reuse_prior_attempts": False,
            "reuse_prior_bundles": False,
            "scope": "rerun_all_36_canary_coordinates",
        },
        "schema_version": 1,
        "study_id": STUDY_ID,
        "supersedes": supersedes,
    }
    execution["lock_sha256"] = digest_value(execution)
    return {
        "preview_sha256": "f" * 64,
        "canary_execution": execution,
    }


def _redigest(preview: dict[str, Any]) -> None:
    execution = preview["canary_execution"]
    execution["lock_sha256"] = ""
    execution["lock_sha256"] = digest_value(execution)


def test_generation_three_lineage_summarizes_exact_predecessor() -> None:
    summary = verify._verify_canary_execution_lineage(_preview(3))

    assert summary == {
        "execution_generation": 3,
        "is_replacement": True,
        "lock_sha256": _preview(3)["canary_execution"]["lock_sha256"],
        "mode": "full_rerun_after_uncertain",
        "reason": "provider_timeout_created_ambiguous_cost_reserve",
        "replacement_scope": "rerun_all_36_canary_coordinates",
        "planned_coordinates": 36,
        "reuses_prior_attempts": False,
        "reuses_prior_bundles": False,
        "superseded_execution_generation": 2,
        "superseded_execution_lock_sha256": "1" * 64,
        "superseded_execution_lock_file_sha256": "2" * 64,
        "superseded_preview_sha256": "c" * 64,
        "superseded_runtime_manifest_sha256": "e" * 64,
        "superseded_ledger_sha256": "b" * 64,
        "superseded_status": "uncertain",
        "superseded_completed_coordinates": 15,
        "superseded_unavailable_coordinates": 1,
        "superseded_terminal_coordinates": 16,
        "superseded_pending_coordinates": 20,
        "superseded_observed_cost_usd": 0.18,
        "superseded_ambiguous_cost_reserve_usd": 0.02,
        "superseded_cost_exposure_upper_bound_usd": 0.2,
        "superseded_cost_uncertainty_reasons": [
            "retryable transport outcome was ambiguous: "
            "attempt=1, error=APITimeoutError"
        ],
    }


def test_generation_four_lineage_summarizes_nonqualifying_predecessor() -> None:
    summary = verify._verify_canary_execution_lineage(_preview(4))

    assert summary["execution_generation"] == 4
    assert summary["mode"] == "full_rerun_after_nonqualifying_partial"
    assert summary["reason"] == (
        "simulator_contract_false_negative_after_visible_correction"
    )
    assert summary["superseded_execution_generation"] == 3
    assert summary["superseded_status"] == "paused"
    assert summary["superseded_completed_coordinates"] == 5
    assert summary["superseded_unavailable_coordinates"] == 1
    assert summary["superseded_pending_coordinates"] == 30
    assert summary["superseded_observed_cost_usd"] == 0.08665227
    assert summary["superseded_ambiguous_cost_reserve_usd"] == 0.0
    assert summary["superseded_cost_uncertainty_reasons"] == []


def test_generation_five_reuses_reviewed_internal_server_recovery_contract() -> None:
    summary = verify._verify_canary_execution_lineage(_preview(5))

    assert summary["execution_generation"] == 5
    assert summary["mode"] == "full_rerun_after_uncertain"
    assert summary["reason"] == (
        "provider_internal_server_error_after_bounded_retries"
    )
    assert summary["superseded_execution_generation"] == 4
    assert summary["superseded_status"] == "uncertain"
    assert summary["superseded_completed_coordinates"] == 1
    assert summary["superseded_unavailable_coordinates"] == 1
    assert summary["superseded_pending_coordinates"] == 34
    assert summary["superseded_observed_cost_usd"] == 0.02721544
    assert summary["superseded_ambiguous_cost_reserve_usd"] == 0.00285705
    assert summary["superseded_cost_exposure_upper_bound_usd"] == 0.03007249
    assert len(summary["superseded_cost_uncertainty_reasons"]) == 3


def test_generation_six_binds_exact_conditional_disclosure_failure() -> None:
    summary = verify._verify_canary_execution_lineage(_preview(6))

    assert summary["execution_generation"] == 6
    assert summary["mode"] == "full_rerun_after_nonqualifying_partial"
    assert summary["reason"] == (
        "simulator_conditional_clause_revealed_before_confirmation_boundary"
    )
    assert summary["superseded_execution_generation"] == 5
    assert summary["superseded_status"] == "paused"
    assert summary["superseded_completed_coordinates"] == 29
    assert summary["superseded_unavailable_coordinates"] == 1
    assert summary["superseded_pending_coordinates"] == 6
    assert summary["superseded_observed_cost_usd"] == 0.32820088
    assert summary["superseded_ambiguous_cost_reserve_usd"] == 0.0
    assert summary["superseded_cost_exposure_upper_bound_usd"] == 0.32820088
    assert summary["superseded_cost_uncertainty_reasons"] == []


def test_generation_two_sealed_schema_remains_verifiable() -> None:
    summary = verify._verify_canary_execution_lineage(_preview(2))

    assert summary["execution_generation"] == 2
    assert summary["superseded_execution_generation"] == 1
    assert summary["superseded_execution_lock_sha256"] is None


def test_generation_one_preview_remains_verifiable_only_as_legacy() -> None:
    preview = {"preview_sha256": "f" * 64}

    assert verify._verify_canary_execution_lineage(
        preview,
        allow_legacy=True,
    ) == {
        "execution_generation": 1,
        "is_replacement": False,
    }
    with pytest.raises(ValueError, match="no valid canary execution lineage"):
        verify._verify_canary_execution_lineage(preview)


@pytest.mark.parametrize(
    ("generation", "wrong_reviewed_reason"),
    [
        (2, "provider_timeout_created_ambiguous_cost_reserve"),
        (3, "provider_internal_server_error_after_bounded_retries"),
        (4, "provider_internal_server_error_after_bounded_retries"),
        (5, "simulator_contract_false_negative_after_visible_correction"),
    ],
)
def test_sealed_historical_generation_rejects_a_different_reviewed_reason(
    generation: int,
    wrong_reviewed_reason: str,
) -> None:
    preview = deepcopy(_preview(generation))
    preview["canary_execution"]["reason"] = wrong_reviewed_reason
    _redigest(preview)

    with pytest.raises(
        ValueError,
        match="sealed historical canary recovery reason mismatch",
    ):
        verify._verify_canary_execution_lineage(preview)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution["supersedes"].__setitem__(
                "execution_generation", 1
            ),
            "generations are not monotonic",
        ),
        (
            lambda execution: execution["replacement_policy"].__setitem__(
                "reuse_prior_bundles", True
            ),
            "replacement policy mismatch",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_exposure_upper_bound_usd", 0.19
            ),
            "cost evidence does not reconcile",
        ),
        (
            lambda execution: execution.__setitem__(
                "reason", "some_unreviewed_recovery_reason"
            ),
            "unsupported canary recovery reason",
        ),
        (
            lambda execution: execution["supersedes"].pop(
                "execution_lock_file_sha256"
            ),
            "execution-lock file digest is required",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "execution_lock_file_sha256", "A" * 64
            ),
            "must use lowercase SHA-256",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_uncertainty_reasons", []
            ),
            "cost-uncertainty reasons are incomplete",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_uncertainty_reasons",
                ["retryable transport outcome was ambiguous"],
            ),
            "no retained APITimeoutError evidence",
        ),
    ],
)
def test_replacement_lineage_rejects_unsafe_or_incoherent_recovery(
    mutation: Any,
    message: str,
) -> None:
    preview = deepcopy(_preview(3))
    mutation(preview["canary_execution"])
    _redigest(preview)

    with pytest.raises(ValueError, match=message):
        verify._verify_canary_execution_lineage(preview)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution["supersedes"].__setitem__(
                "status", "uncertain"
            ),
            "predecessor status does not match recovery",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "completed_coordinate_count", 6
            ),
            "must retain 5 complete and 1 unavailable",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "ambiguous_cost_reserve_usd", 0.01
            ),
            "cost evidence does not reconcile",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_uncertainty_reasons", ["ambiguous"]
            ),
            "cannot retain cost uncertainty",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "unavailable", {"coordinate-5": "some other failure"}
            ),
            "no exact false-negative evidence",
        ),
    ],
)
def test_generation_four_rejects_incoherent_nonqualifying_recovery(
    mutation: Any,
    message: str,
) -> None:
    preview = deepcopy(_preview(4))
    mutation(preview["canary_execution"])
    _redigest(preview)

    with pytest.raises(ValueError, match=message):
        verify._verify_canary_execution_lineage(preview)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_uncertainty_reasons",
                ["retryable transport outcome was ambiguous"],
            ),
            "no retained InternalServerError uncertainty evidence",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "unavailable",
                {"coordinate-1": "some other failure"},
            ),
            "no matching unavailable evidence",
        ),
    ],
)
def test_generation_five_rejects_unsubstantiated_internal_server_recovery(
    mutation: Any,
    message: str,
) -> None:
    preview = deepcopy(_preview(5))
    mutation(preview["canary_execution"])
    _redigest(preview)

    with pytest.raises(ValueError, match=message):
        verify._verify_canary_execution_lineage(preview)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution["supersedes"].__setitem__(
                "completed_coordinate_count", 28
            ),
            "must retain 29 complete, 1 unavailable, and 6 pending",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "cost_uncertainty_reasons", ["ambiguous"]
            ),
            "cannot retain cost uncertainty",
        ),
        (
            lambda execution: execution["supersedes"].__setitem__(
                "unavailable", {"coordinate-29": "some other failure"}
            ),
            "no exact simulator evidence",
        ),
    ],
)
def test_generation_six_rejects_incoherent_conditional_disclosure_recovery(
    mutation: Any,
    message: str,
) -> None:
    preview = deepcopy(_preview(6))
    mutation(preview["canary_execution"])
    _redigest(preview)

    with pytest.raises(ValueError, match=message):
        verify._verify_canary_execution_lineage(preview)
