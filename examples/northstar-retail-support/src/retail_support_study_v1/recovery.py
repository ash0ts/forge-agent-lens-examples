from __future__ import annotations

CANARY_RECOVERY_CONTRACTS: dict[str, dict[str, str]] = {
    "provider_internal_server_error_after_bounded_retries": {
        "mode": "full_rerun_after_uncertain",
        "predecessor_status": "uncertain",
    },
    "provider_timeout_created_ambiguous_cost_reserve": {
        "mode": "full_rerun_after_uncertain",
        "predecessor_status": "uncertain",
    },
    "simulator_contract_false_negative_after_visible_correction": {
        "mode": "full_rerun_after_nonqualifying_partial",
        "predecessor_status": "paused",
    },
    "simulator_conditional_clause_revealed_before_confirmation_boundary": {
        "mode": "full_rerun_after_nonqualifying_partial",
        "predecessor_status": "paused",
    },
}

SEALED_CANARY_RECOVERY_REASONS = {
    2: "provider_internal_server_error_after_bounded_retries",
    3: "provider_timeout_created_ambiguous_cost_reserve",
    4: "simulator_contract_false_negative_after_visible_correction",
    5: "provider_internal_server_error_after_bounded_retries",
}


def reviewed_canary_recovery_contract(
    *,
    generation: int,
    reason: object,
) -> dict[str, str]:
    """Resolve a reviewed contract without permitting historical relabeling."""

    if not isinstance(reason, str):
        raise ValueError("unsupported canary recovery reason")
    contract = CANARY_RECOVERY_CONTRACTS.get(reason)
    if contract is None:
        raise ValueError("unsupported canary recovery reason")
    sealed_reason = SEALED_CANARY_RECOVERY_REASONS.get(generation)
    if sealed_reason is not None and reason != sealed_reason:
        raise ValueError("sealed historical canary recovery reason mismatch")
    return contract
