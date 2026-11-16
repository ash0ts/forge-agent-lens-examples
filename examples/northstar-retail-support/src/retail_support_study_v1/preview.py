from __future__ import annotations

import hashlib
import sys
import uuid
from typing import Any

from .canonical import (
    digest_file,
    digest_value,
    load_json,
    require_hex_digest,
    write_json_atomic,
)
from .constants import (
    CANARY_COORDINATE_COUNT,
    CANARY_COST_CAP_USD,
    CANARY_EXECUTION_LOCK_PATH,
    CANARY_SELECTION_POLICY,
    CANARY_TASK_COUNT,
    CANARY_TASK_GROUPS,
    CANARY_TASK_IDS,
    CANARY_TRIAL_ASSIGNMENT_POLICY,
    FULL_COORDINATE_COUNT,
    FULL_COST_CAP_USD,
    FULL_COST_PROJECTION_MARGIN,
    INFERENCE_PROJECT,
    INTERNAL_SERVER_RETRY_DELAYS_SECONDS,
    LEGACY_CANARY_COORDINATE_COUNT,
    LIVE_AGENT_LENS_PROJECTS,
    LIVE_INSTRUMENTATION_VERSION,
    MAX_AGENT_TOOL_CALLS,
    MAX_CONVERSATION_STEPS,
    OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS,
    PACKAGE_ROOT,
    PREPARED_DIR,
    PREVIEW_PATH,
    PRIVATE_DIR,
    PROFILES,
    RUNS_DIR,
    SCHEMA_VERSION,
    STUDY_ID,
    SUPPORT_MAX_OUTPUT_TOKENS,
    TASK_COUNT,
    TRANSPORT_RETRIES,
    TRIAL_INDICES,
    USER_CONFORMANCE_RETRIES,
    USER_ENABLE_THINKING,
    USER_MAX_OUTPUT_TOKENS,
    USER_MODEL,
    USER_REPAIR_SEED_OFFSET,
    USER_TEMPERATURE,
    WANDB_INFERENCE_BASE_URL,
)
from .mcp_protocol import MCP_PROTOCOL_VERSION, MCPProcessClient
from .recovery import reviewed_canary_recovery_contract
from .remap import verify_prepared_source
from .schemas import Coordinate, RunLedger
from .simulator_contracts import (
    SIMULATOR_CONTRACT_VERSION,
    confirmation_contract_inventory,
)
from .traffic_config import (
    AGENT_ROUTER_POLICY,
    MODEL_PROBE_POLICY,
    allocate_model_route_ids,
    build_agent_profiles,
    build_behavior_profile,
    build_model_routes,
)

_NAMESPACE = uuid.UUID("5aa9524a-07ad-5536-8584-3cf8ba68321d")

_SIMULATOR_FALSE_NEGATIVE_REASON = (
    "RuntimeError: simulator failed the bounded scenario-conformance repair"
)
_SIMULATOR_CONDITIONAL_DISCLOSURE_REASON = (
    "simulator_conditional_clause_revealed_before_confirmation_boundary"
)


def _canary_execution_lock() -> dict[str, Any]:
    """Validate a replacement generation against its retained predecessor."""

    if (
        CANARY_EXECUTION_LOCK_PATH.is_symlink()
        or not CANARY_EXECUTION_LOCK_PATH.is_file()
    ):
        raise ValueError("canary execution lock must be a regular file")
    lock = load_json(CANARY_EXECUTION_LOCK_PATH)
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("canary execution lock schema mismatch")
    claimed = require_hex_digest(
        str(lock.get("lock_sha256") or ""), label="canary execution lock"
    )
    unsigned = {**lock, "lock_sha256": ""}
    if claimed != digest_value(unsigned):
        raise ValueError("canary execution lock digest mismatch")
    if lock.get("study_id") != STUDY_ID or lock.get("cohort") != "canary":
        raise ValueError("canary execution lock identity mismatch")
    generation = lock.get("execution_generation")
    if not isinstance(generation, int) or generation < 2:
        raise ValueError("canary execution generation must be at least 2")
    reason = lock.get("reason")
    recovery_contract = reviewed_canary_recovery_contract(
        generation=generation,
        reason=reason,
    )
    if lock.get("mode") != recovery_contract["mode"]:
        raise ValueError("canary execution recovery mode mismatch")
    if lock.get("replacement_policy") != {
        "planned_coordinate_count": LEGACY_CANARY_COORDINATE_COUNT,
        "reuse_prior_attempts": False,
        "reuse_prior_bundles": False,
        "scope": "rerun_all_36_canary_coordinates",
    }:
        raise ValueError("canary replacement policy mismatch")

    supersedes = lock.get("supersedes")
    if (
        not isinstance(supersedes, dict)
        or supersedes.get("status") != recovery_contract["predecessor_status"]
    ):
        raise ValueError(
            "canary execution lock predecessor status does not match recovery"
        )
    digest_fields = (
        "preview_sha256",
        "ledger_sha256",
        "ledger_file_sha256",
        "runtime_manifest_sha256",
        "runtime_manifest_file_sha256",
    )
    if generation > 2:
        digest_fields += (
            "execution_lock_sha256",
            "execution_lock_file_sha256",
        )
    for field in digest_fields:
        require_hex_digest(
            str(supersedes.get(field) or ""),
            label=f"superseded {field}",
        )

    preview_sha = str(supersedes["preview_sha256"])
    ledger_path = RUNS_DIR / "canary" / preview_sha / "ledger.json"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("superseded canary ledger must be a regular file")
    if digest_file(ledger_path) != supersedes["ledger_file_sha256"]:
        raise ValueError("superseded canary ledger file digest mismatch")
    ledger = RunLedger.model_validate(load_json(ledger_path))
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("superseded canary ledger content digest mismatch")
    if ledger.ledger_sha256 != supersedes["ledger_sha256"]:
        raise ValueError("superseded canary ledger digest mismatch")
    if (
        ledger.study_id != STUDY_ID
        or ledger.cohort != "canary"
        or ledger.preview_sha256 != preview_sha
    ):
        raise ValueError("superseded canary ledger identity mismatch")
    if ledger.status != recovery_contract["predecessor_status"]:
        raise ValueError("superseded canary status does not match recovery")

    expected_unavailable = supersedes.get("unavailable")
    if not isinstance(expected_unavailable, dict) or ledger.unavailable != (
        expected_unavailable
    ):
        raise ValueError("superseded canary unavailable evidence mismatch")
    expected_counts = {
        "planned_coordinate_count": len(ledger.planned_coordinate_ids),
        "completed_coordinate_count": len(ledger.completed_coordinate_ids),
        "unavailable_coordinate_count": len(ledger.unavailable),
    }
    for field, actual in expected_counts.items():
        if supersedes.get(field) != actual:
            raise ValueError(f"superseded canary {field} mismatch")
    if supersedes.get("unavailable_coordinate_ids") != list(ledger.unavailable):
        raise ValueError("superseded canary unavailable-coordinate order mismatch")
    if supersedes.get("cost_uncertainty_reasons") != (
        ledger.cost_uncertainty_reasons
    ):
        raise ValueError("superseded canary cost-uncertainty evidence mismatch")
    for field, actual in (
        ("observed_cost_usd", ledger.observed_cost_usd),
        ("ambiguous_cost_reserve_usd", ledger.ambiguous_cost_reserve_usd),
        ("cost_exposure_upper_bound_usd", ledger.cost_exposure_upper_bound_usd),
    ):
        expected = supersedes.get(field)
        if not isinstance(expected, (int, float)) or abs(float(expected) - actual) > 1e-12:
            raise ValueError(f"superseded canary {field} mismatch")

    # Import lazily because runtime_snapshot imports constants shared by preview.
    from .runtime_snapshot import verify_runtime_snapshot

    prior_runtime = verify_runtime_snapshot(
        preview_sha,
        require_current_code=False,
    )
    if prior_runtime.manifest.get("manifest_sha256") != supersedes[
        "runtime_manifest_sha256"
    ]:
        raise ValueError("superseded runtime manifest digest mismatch")
    manifest_path = prior_runtime.root / "manifest.json"
    if digest_file(manifest_path) != supersedes["runtime_manifest_file_sha256"]:
        raise ValueError("superseded runtime manifest file digest mismatch")
    prior_execution = prior_runtime.preview.get("canary_execution")
    if isinstance(prior_execution, dict):
        prior_generation = prior_execution.get("execution_generation")
        prior_lock_sha = prior_execution.get("lock_sha256")
        if not isinstance(prior_generation, int) or prior_generation < 2:
            raise ValueError("superseded canary execution generation is invalid")
        require_hex_digest(
            str(prior_lock_sha or ""),
            label="superseded canary execution lock",
        )
        prior_lock_path = prior_runtime.root / "canary-execution.lock.json"
        prior_lock = load_json(prior_lock_path)
        if not isinstance(prior_lock, dict):
            raise ValueError("superseded canary execution lock schema mismatch")
        if prior_lock.get("lock_sha256") != digest_value(
            {**prior_lock, "lock_sha256": ""}
        ):
            raise ValueError("superseded canary execution lock self-digest mismatch")
        if prior_lock != prior_execution:
            raise ValueError("superseded runtime execution lock content mismatch")
        if supersedes.get("execution_lock_sha256") != prior_lock_sha:
            raise ValueError("superseded canary execution lock digest mismatch")
        if digest_file(prior_lock_path) != supersedes.get(
            "execution_lock_file_sha256"
        ):
            raise ValueError("superseded canary execution lock file mismatch")
    else:
        prior_generation = 1
        if any(
            field in supersedes
            for field in (
                "execution_lock_sha256",
                "execution_lock_file_sha256",
            )
        ):
            raise ValueError("legacy canary cannot claim an execution lock")
    if supersedes.get("execution_generation") != prior_generation:
        raise ValueError("superseded canary generation mismatch")
    if generation != prior_generation + 1:
        raise ValueError("replacement canary generation is not monotonic")

    if reason == "provider_timeout_created_ambiguous_cost_reserve" and not any(
        "APITimeoutError" in item for item in ledger.cost_uncertainty_reasons
    ):
        raise ValueError("timeout recovery reason has no timeout evidence")
    if reason == "provider_internal_server_error_after_bounded_retries" and not any(
        "InternalServerError" in item for item in ledger.unavailable.values()
    ):
        raise ValueError("internal-server recovery reason has no matching evidence")
    if reason == "simulator_contract_false_negative_after_visible_correction":
        if len(ledger.completed_coordinate_ids) != 5 or len(ledger.unavailable) != 1:
            raise ValueError(
                "simulator-contract recovery predecessor must retain 5 complete "
                "and 1 unavailable coordinate"
            )
        if (
            ledger.ambiguous_cost_reserve_usd != 0
            or ledger.cost_uncertainty_reasons
        ):
            raise ValueError(
                "simulator-contract recovery cannot retain cost uncertainty"
            )
        if set(ledger.unavailable.values()) != {_SIMULATOR_FALSE_NEGATIVE_REASON}:
            raise ValueError(
                "simulator-contract recovery has no exact false-negative evidence"
            )
    if reason == _SIMULATOR_CONDITIONAL_DISCLOSURE_REASON:
        if (
            len(ledger.completed_coordinate_ids) != 29
            or len(ledger.unavailable) != 1
            or len(ledger.planned_coordinate_ids)
            - len(ledger.completed_coordinate_ids)
            - len(ledger.unavailable)
            != 6
        ):
            raise ValueError(
                "conditional-disclosure recovery predecessor must retain 29 "
                "complete, 1 unavailable, and 6 pending coordinates"
            )
        if (
            ledger.ambiguous_cost_reserve_usd != 0
            or ledger.cost_uncertainty_reasons
        ):
            raise ValueError(
                "conditional-disclosure recovery cannot retain cost uncertainty"
            )
        if set(ledger.unavailable.values()) != {_SIMULATOR_FALSE_NEGATIVE_REASON}:
            raise ValueError(
                "conditional-disclosure recovery has no exact simulator evidence"
            )
    prior_coordinates = [
        str(row["coordinate_id"])
        for row in prior_runtime.preview["cohorts"]["canary"]["coordinates"]
    ]
    if prior_coordinates != ledger.planned_coordinate_ids:
        raise ValueError("superseded runtime and ledger coordinate order mismatch")
    return lock


def _validate_replacement_contract(
    preview: dict[str, Any],
    canary_execution: dict[str, Any],
) -> None:
    """Prove which non-source inputs a replacement intentionally preserves."""

    from .runtime_snapshot import verify_runtime_snapshot

    prior = verify_runtime_snapshot(
        str(canary_execution["supersedes"]["preview_sha256"]),
        require_current_code=False,
    )
    prior_preview = prior.preview
    input_keys = (
        "database_sha256",
        "policy_sha256",
        "user_guidelines_sha256",
        "runtime_tasks_sha256",
        "task_splits_sha256",
        "task_truth_sha256",
        "prompt_files",
    )
    for key in input_keys:
        if preview["inputs"].get(key) != prior_preview["inputs"].get(key):
            raise ValueError(f"replacement canary changed preserved input: {key}")
    for key in ("source_lock_sha256", "mcp", "cost_caps_usd"):
        if preview.get(key) != prior_preview.get(key):
            raise ValueError(f"replacement canary changed preserved contract: {key}")
    current_run_policy = preview.get("run_policy")
    prior_run_policy = prior_preview.get("run_policy")
    if canary_execution["reason"] == (
        "provider_internal_server_error_after_bounded_retries"
    ):
        if not isinstance(prior_run_policy, dict):
            raise ValueError("superseded canary run policy is malformed")
        expected_run_policy = {
            **prior_run_policy,
            "transport_retry_delays_seconds": {
                "http_500_internal_server_error": list(
                    INTERNAL_SERVER_RETRY_DELAYS_SECONDS
                ),
                "other_retryable_transport": list(
                    OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS
                ),
            },
        }
        if current_run_policy != expected_run_policy:
            raise ValueError(
                "internal-server recovery changed more than the reviewed "
                "transport retry delays"
            )
    elif current_run_policy != prior_run_policy:
        raise ValueError("replacement canary changed preserved contract: run_policy")
    current_models = preview.get("models")
    prior_models = prior_preview.get("models")
    simulator_contract_versions = {
        "simulator_contract_false_negative_after_visible_correction": (
            "confirmation-choreography-v6",
            "confirmation-choreography-v7",
        ),
        _SIMULATOR_CONDITIONAL_DISCLOSURE_REASON: (
            "confirmation-choreography-v7",
            "confirmation-choreography-v8",
        ),
    }
    simulator_contract_boundary = simulator_contract_versions.get(
        str(canary_execution["reason"])
    )
    if simulator_contract_boundary is not None:
        if not isinstance(current_models, dict) or not isinstance(prior_models, dict):
            raise ValueError("replacement canary model contract is malformed")
        prior_user = prior_models.get("user_simulator")
        if not isinstance(prior_user, dict):
            raise ValueError("superseded user-simulator contract is malformed")
        prior_conformance = prior_user.get("scenario_conformance")
        if (
            not isinstance(prior_conformance, dict)
            or prior_conformance.get("contract_version")
            != simulator_contract_boundary[0]
            or SIMULATOR_CONTRACT_VERSION != simulator_contract_boundary[1]
        ):
            raise ValueError("simulator-contract recovery version boundary mismatch")
        expected_models = {
            **prior_models,
            "user_simulator": {
                **prior_user,
                "scenario_conformance": {
                    **prior_conformance,
                    "contract_version": SIMULATOR_CONTRACT_VERSION,
                },
            },
        }
        if current_models != expected_models:
            raise ValueError(
                "replacement canary changed more than the reviewed simulator "
                "contract version"
            )
    elif current_models != prior_models:
        raise ValueError("replacement canary changed preserved contract: models")
    if preview["cohorts"] != prior_preview["cohorts"]:
        raise ValueError("replacement canary changed the frozen coordinate matrix")


def _coordinate_key(
    task_id: str,
    profile: str,
    cohort: str,
    trial_index: int,
) -> str:
    return f"{STUDY_ID}:{cohort}:{task_id}:{profile}:{trial_index}"


def _seed(
    task_id: str,
    profile: str,
    cohort: str,
    trial_index: int,
    model_route_id: str,
) -> int:
    material = (
        f"{_coordinate_key(task_id, profile, cohort, trial_index)}:"
        f"{model_route_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _coordinate(
    task_id: str,
    profile: str,
    cohort: str,
    trial_index: int,
    model_route_id: str,
) -> Coordinate:
    identity = (
        f"{_coordinate_key(task_id, profile, cohort, trial_index)}:"
        f"{model_route_id}"
    )
    coordinate_id = str(uuid.uuid5(_NAMESPACE, identity))
    return Coordinate(
        coordinate_id=coordinate_id,
        logical_task_id=task_id,
        profile=profile,
        trial_index=trial_index,
        model_route_id=model_route_id,
        seed=_seed(task_id, profile, cohort, trial_index, model_route_id),
        cohort=cohort,
    )


def coordinate_matrix(cohort: str, task_ids: list[str]) -> list[Coordinate]:
    """Return the exact preregistered order for one cohort.

    Full runs are trial-major so each contiguous block is one balanced
    114-task x 3-profile replicate. The fixed 60-coordinate canary keeps one
    profile coordinate per selected task while round-robining those coordinates
    evenly across the same four trial indices. Both matrices then receive an
    exact, digest-stable model-route allocation without changing their order.
    """

    if cohort == "full":
        planned = [
            (task_id, profile, trial_index)
            for trial_index in TRIAL_INDICES
            for task_id in task_ids
            for profile in PROFILES
        ]
    elif cohort == "canary":
        planned = [
            (task_id, profile, trial_index)
            for scheduled_trial in TRIAL_INDICES
            for task_id, profile, trial_index in (
                (
                    task_id,
                    profile,
                    (
                        task_position * len(PROFILES) + profile_position
                    )
                    % len(TRIAL_INDICES),
                )
                for task_position, task_id in enumerate(task_ids)
                for profile_position, profile in enumerate(PROFILES)
            )
            if trial_index == scheduled_trial
        ]
    else:
        raise ValueError("cohort must be canary or full")
    route_ids = allocate_model_route_ids(
        cohort=cohort,
        coordinate_keys=[
            _coordinate_key(task_id, profile, cohort, trial_index)
            for task_id, profile, trial_index in planned
        ],
    )
    return [
        _coordinate(task_id, profile, cohort, trial_index, route_id)
        for (task_id, profile, trial_index), route_id in zip(
            planned,
            route_ids,
            strict=True,
        )
    ]


def _mcp_manifest() -> dict[str, Any]:
    with MCPProcessClient(
        db_path=PREPARED_DIR / "db.json",
        policy_path=PREPARED_DIR / "policy.md",
    ) as mcp:
        tools = mcp.list_openai_tools()
    names = [tool["function"]["name"] for tool in tools]
    return {
        "transport": "stdio",
        "protocol": f"mcp-jsonrpc-{MCP_PROTOCOL_VERSION}",
        "account_scope": "authenticated-session-user",
        "tool_names": names,
        "tool_count": len(names),
        "tools": tools,
        "digest": digest_value(tools),
    }


def _runtime_manifest() -> dict[str, Any]:
    source_root = PACKAGE_ROOT / "src" / "retail_support_study_v1"
    source_files = {
        path.relative_to(PACKAGE_ROOT).as_posix(): digest_file(path)
        for path in sorted(source_root.glob("*.py"))
    }
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pyproject_sha256": digest_file(PACKAGE_ROOT / "pyproject.toml"),
        "uv_lock_sha256": digest_file(PACKAGE_ROOT / "uv.lock"),
        "source_files": source_files,
        "source_digest": digest_value(source_files),
    }


def _canary_selection_metadata(
    *,
    truth: list[dict[str, Any]],
    task_splits: dict[str, list[str]],
    manifest_tool_names: list[str],
) -> dict[str, Any]:
    truth_by_id = {str(row["id"]): row for row in truth}
    selected_truth = [truth_by_id[task_id] for task_id in CANARY_TASK_IDS]
    selected_actions = [
        action
        for row in selected_truth
        for action in row["evaluation_criteria"]["actions"]
    ]
    selected_reference_tools = {str(action["name"]) for action in selected_actions}
    all_reference_tools = {
        str(action["name"])
        for row in truth
        for action in row["evaluation_criteria"]["actions"]
    }
    manifest_tools = set(manifest_tool_names)
    unknown_reference_tools = all_reference_tools - manifest_tools
    if unknown_reference_tools:
        raise ValueError(
            "reference actions use tools absent from the MCP manifest: "
            f"{sorted(unknown_reference_tools)}"
        )
    missing_reference_tools = all_reference_tools - selected_reference_tools
    if missing_reference_tools:
        raise ValueError(
            "canary selection misses pinned reference tools: "
            f"{sorted(missing_reference_tools)}"
        )

    split_sets = {
        split: {str(task_id) for task_id in task_splits[split]}
        for split in ("train", "test")
    }
    invalid_split_membership = [
        task_id
        for task_id in CANARY_TASK_IDS
        if sum(task_id in task_ids for task_ids in split_sets.values()) != 1
    ]
    if invalid_split_membership:
        raise ValueError(
            "canary task IDs must belong to exactly one source split: "
            f"{invalid_split_membership}"
        )
    source_split_counts = {
        split: sum(task_id in task_ids for task_id in CANARY_TASK_IDS)
        for split, task_ids in split_sets.items()
    }

    return {
        "id": CANARY_SELECTION_POLICY,
        "task_group_counts": {
            group: len(task_ids) for group, task_ids in CANARY_TASK_GROUPS.items()
        },
        "source_split_counts": source_split_counts,
        "reference_action_count": len(selected_actions),
        "reference_tool_count": len(selected_reference_tools),
        "manifest_tool_count": len(manifest_tools),
        "manifest_tools_without_reference_actions": len(
            manifest_tools - all_reference_tools
        ),
        "covers_all_reference_tools": True,
        "required_strata": [
            "human_handoff",
            "lookup_read_only",
            "mutation_confirmation",
            "no_action_communication",
            "simple_mutation_baseline",
        ],
    }


def build_preview() -> dict[str, Any]:
    if len(CANARY_TASK_IDS) != CANARY_TASK_COUNT:
        raise ValueError("canary task count mismatch")
    if len(set(CANARY_TASK_IDS)) != CANARY_TASK_COUNT:
        raise ValueError("canary task IDs are not unique")

    source_lock = verify_prepared_source()
    tasks = load_json(PREPARED_DIR / "tasks.runtime.json")
    truth = load_json(PRIVATE_DIR / "task-truth.json")
    task_splits = load_json(PREPARED_DIR / "task_splits.json")
    if len(tasks) != TASK_COUNT or len(truth) != TASK_COUNT:
        raise ValueError("prepared task count is not 114")
    task_ids = [str(task["id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs are not unique")

    available_task_ids = {str(task["id"]) for task in truth}
    missing_canary = set(CANARY_TASK_IDS) - available_task_ids
    if missing_canary:
        raise ValueError(f"canary task IDs missing from pinned source: {missing_canary}")
    canary_task_ids = list(CANARY_TASK_IDS)
    canary = [
        coordinate.model_dump(mode="json")
        for coordinate in coordinate_matrix("canary", canary_task_ids)
    ]
    full = [
        coordinate.model_dump(mode="json")
        for coordinate in coordinate_matrix("full", task_ids)
    ]
    if len(canary) != CANARY_COORDINATE_COUNT:
        raise ValueError("canary coordinate count mismatch")
    if len(full) != FULL_COORDINATE_COUNT:
        raise ValueError("full coordinate count mismatch")

    package_root = PACKAGE_ROOT
    mcp_manifest = _mcp_manifest()
    selection_metadata = _canary_selection_metadata(
        truth=truth,
        task_splits=task_splits,
        manifest_tool_names=mcp_manifest["tool_names"],
    )
    prompt_files = {
        name: digest_file(package_root / "prompts" / name)
        for name in (
            "support-agent.md",
            "ordinary-user.md",
            "concise-user.md",
            "skeptical-user.md",
        )
    }
    agent_profiles = build_agent_profiles(
        instructions_sha256=prompt_files["support-agent.md"],
        manifest_tool_names=mcp_manifest["tool_names"],
    )
    model_routes = build_model_routes()
    behavior_profile = build_behavior_profile()
    route_ids = {route.route_id for route in model_routes}
    if {
        str(row["model_route_id"])
        for row in canary + full
    } != route_ids:
        raise ValueError("coordinate matrix does not use every sealed model route")

    def route_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            route.route_id: sum(
                row["model_route_id"] == route.route_id for row in rows
            )
            for route in model_routes
        }

    telemetry_schema = load_json(package_root / "telemetry-schema.json")
    preview = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "inputs": {
            "database_sha256": digest_file(PREPARED_DIR / "db.json"),
            "policy_sha256": digest_file(PREPARED_DIR / "policy.md"),
            "user_guidelines_sha256": digest_file(
                PREPARED_DIR / "user-guidelines.md"
            ),
            "runtime_tasks_sha256": digest_file(
                PREPARED_DIR / "tasks.runtime.json"
            ),
            "task_splits_sha256": digest_file(PREPARED_DIR / "task_splits.json"),
            "task_truth_sha256": digest_file(PRIVATE_DIR / "task-truth.json"),
            "prompt_files": prompt_files,
        },
        "models": {
            "support_agent": {
                "selection": "coordinate_model_route_id",
                "route_ids": [route.route_id for route in model_routes],
                "temperature": "sealed_per_model_route",
                "max_output_tokens": SUPPORT_MAX_OUTPUT_TOKENS,
            },
            "user_simulator": {
                "provider": "wandb_inference",
                "model": USER_MODEL,
                "temperature": USER_TEMPERATURE,
                "max_output_tokens": USER_MAX_OUTPUT_TOKENS,
                "enable_thinking": USER_ENABLE_THINKING,
                "seed_policy": "coordinate_sha256_u32",
                "scenario_conformance": {
                    "contract_version": SIMULATOR_CONTRACT_VERSION,
                    "confirmation_inventory": confirmation_contract_inventory(),
                    "max_retries": USER_CONFORMANCE_RETRIES,
                    "repair_seed_offset": USER_REPAIR_SEED_OFFSET,
                    "rejected_drafts": "private_costed_not_emitted",
                    "unresolved": "unavailable",
                },
            },
            "base_url": WANDB_INFERENCE_BASE_URL,
            "usage_project": INFERENCE_PROJECT,
            "pricing_lock_sha256": digest_file(
                package_root / "model-pricing.lock.json"
            ),
        },
        "traffic": {
            "agent_profiles": [
                profile.model_dump(mode="json") for profile in agent_profiles
            ],
            "agent_router": dict(AGENT_ROUTER_POLICY),
            "model_routes": [
                route.model_dump(mode="json") for route in model_routes
            ],
            "model_probe_policy": dict(MODEL_PROBE_POLICY),
            "behavior_profile": behavior_profile.model_dump(mode="json"),
        },
        "mcp": mcp_manifest,
        "runtime": _runtime_manifest(),
        "run_policy": {
            "transport_retries": TRANSPORT_RETRIES,
            "transport_retry_delays_seconds": {
                "http_500_internal_server_error": list(
                    INTERNAL_SERVER_RETRY_DELAYS_SECONDS
                ),
                "other_retryable_transport": list(
                    OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS
                ),
            },
            "max_conversation_steps": MAX_CONVERSATION_STEPS,
            "max_agent_tool_calls": MAX_AGENT_TOOL_CALLS,
            "reset_store_after_every_conversation": True,
            "unresolved_attempts": "unavailable",
            "frozen_attempt_failure_policy": (
                "continue_when_bundle_is_frozen"
            ),
            "bounded_retry_cost_ambiguity": (
                "track_upper_bound_and_continue"
            ),
            "full_run_requires_exact_preview_approval": True,
            "full_run_approval_mode": "direct_operator_exact_preview_sha",
            "canary_cost_cap_usd": CANARY_COST_CAP_USD,
            "full_cost_projection_margin": FULL_COST_PROJECTION_MARGIN,
            "full_cost_hard_ceiling_usd": FULL_COST_CAP_USD,
        },
        "telemetry": {
            "mode": "live_application_instrumentation",
            "destinations": dict(LIVE_AGENT_LENS_PROJECTS),
            "agent_names": [profile.agent_name for profile in agent_profiles],
            "instrumentation_version": LIVE_INSTRUMENTATION_VERSION,
            "schema": telemetry_schema,
            "schema_sha256": digest_value(telemetry_schema),
            "schema_file_sha256": digest_file(
                package_root / "telemetry-schema.json"
            ),
            "emit_from_frozen_bundles_only": False,
            "support_agent_inference_spans": "live",
            "mcp_tool_spans": "live_executions_only",
            "simulator_spans": "excluded",
            "run_requires_live_agent_lens": True,
        },
        "cost_caps_usd": {
            "canary": CANARY_COST_CAP_USD,
            "full": FULL_COST_CAP_USD,
        },
        "cohorts": {
            "canary": {
                "logical_tasks": len(canary_task_ids),
                "profiles": list(PROFILES),
                "trial_indices": list(TRIAL_INDICES),
                "trial_assignment_policy": CANARY_TRIAL_ASSIGNMENT_POLICY,
                "trial_coordinate_counts": [
                    sum(row["trial_index"] == trial for row in canary)
                    for trial in TRIAL_INDICES
                ],
                "ordering": "trial-major-task-profile",
                "selection_policy": selection_metadata,
                "model_route_counts": route_counts(canary),
                "coordinates": canary,
            },
            "full": {
                "logical_tasks": len(task_ids),
                "profiles": list(PROFILES),
                "trial_indices": list(TRIAL_INDICES),
                "trial_coordinate_counts": [
                    len(task_ids) * len(PROFILES) for _ in TRIAL_INDICES
                ],
                "ordering": "trial-major-task-profile",
                "model_route_counts": route_counts(full),
                "coordinates": full,
            },
        },
        "private_boundaries": {
            "never_emit": [
                "task IDs",
                "expected actions",
                "deterministic scores",
                "simulator system prompts",
                "review labels",
                "benchmark source paths or URLs",
                "credentials",
            ]
        },
    }
    preview["preview_sha256"] = digest_value(preview)
    return preview


def write_preview() -> dict[str, Any]:
    preview = build_preview()
    # The digest is approved only after every runtime input has been copied into
    # a write-once snapshot. Paid execution never reads the mutable authoring tree.
    from .runtime_snapshot import materialize_runtime_snapshot

    materialize_runtime_snapshot(preview)
    write_json_atomic(PREVIEW_PATH, preview)
    return preview


def verify_preview() -> dict[str, Any]:
    saved = load_json(PREVIEW_PATH)
    rebuilt = build_preview()
    if saved != rebuilt:
        raise ValueError("preview drift; regenerate and obtain a new approval")
    from .runtime_snapshot import verify_runtime_snapshot

    verify_runtime_snapshot(saved["preview_sha256"])
    return saved
