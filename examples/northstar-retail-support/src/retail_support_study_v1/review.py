from __future__ import annotations

from typing import Any

from .canonical import digest_value, load_json, write_json_atomic
from .constants import PRIVATE_DIR, RUNS_DIR
from .preview import verify_preview
from .schemas import ConversationBundle, RunLedger
from .verify import verify_cohort


def build_canary_review_template() -> dict[str, Any]:
    preview = verify_preview()
    preview_sha = preview["preview_sha256"]
    cohort_dir = RUNS_DIR / "canary" / preview_sha
    ledger = load_json(cohort_dir / "ledger.json")
    rows = []
    for coordinate_id in ledger["planned_coordinate_ids"]:
        path = cohort_dir / "bundles" / f"{coordinate_id}.json"
        if not path.exists():
            rows.append(
                {
                    "coordinate_id": coordinate_id,
                    "run_status": "missing",
                    "natural_progressive_dialogue": None,
                    "no_hidden_truth_leakage": None,
                    "tool_calling_valid": None,
                    "mutation_confirmation_valid": None,
                    "simulator_conformance_repair_count": None,
                    "simulator_conformance_valid": None,
                    "terminal_response_valid": None,
                    "review_notes": "",
                }
            )
            continue
        bundle = ConversationBundle.model_validate(load_json(path))
        repaired = sum(
            receipt.outcome == "repaired"
            for receipt in bundle.simulator_conformance_receipts
        )
        rows.append(
            {
                "coordinate_id": coordinate_id,
                "run_status": bundle.status,
                "natural_progressive_dialogue": None,
                "no_hidden_truth_leakage": None,
                "tool_calling_valid": None,
                "mutation_confirmation_valid": None,
                "simulator_conformance_repair_count": repaired,
                "simulator_conformance_valid": None,
                "terminal_response_valid": None,
                "review_notes": "",
            }
        )
    template = {
        "schema_version": 2,
        "preview_sha256": preview_sha,
        "canary_ledger_sha256": ledger["ledger_sha256"],
        "review_status": "pending_human_review",
        "reviewed_by": "",
        "reviewed_at": None,
        "rows": rows,
        "review_sha256": "",
    }
    unsigned = {**template, "review_sha256": ""}
    template["review_sha256"] = digest_value(unsigned)
    path = PRIVATE_DIR / f"canary-review-{preview_sha}.json"
    write_json_atomic(path, template)
    return template


def verify_canary_review() -> dict[str, Any]:
    preview = verify_preview()
    cohort_dir = RUNS_DIR / "canary" / preview["preview_sha256"]
    ledger = RunLedger.model_validate(load_json(cohort_dir / "ledger.json"))
    if ledger.ledger_sha256 != digest_value(ledger.unsigned_payload()):
        raise ValueError("canary ledger content digest mismatch")
    if ledger.preview_sha256 != preview["preview_sha256"]:
        raise ValueError("canary ledger preview mismatch")
    planned_ids = [
        row["coordinate_id"] for row in preview["cohorts"]["canary"]["coordinates"]
    ]
    if ledger.planned_coordinate_ids != planned_ids:
        raise ValueError("canary ledger planned coordinates mismatch")
    if set(ledger.completed_coordinate_ids) != set(planned_ids):
        raise ValueError("every canary coordinate must complete before approval")
    if ledger.unavailable:
        raise ValueError("unavailable canary coordinates cannot be approved")
    verified = verify_cohort("canary")
    if verified["status"] != "complete":
        raise ValueError("canary ledger must verify as complete before approval")
    if verified["ledger_sha256"] != ledger.ledger_sha256:
        raise ValueError("verified canary ledger does not match the reviewed ledger")

    path = PRIVATE_DIR / f"canary-review-{preview['preview_sha256']}.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("canary review must be a regular file")
    review = load_json(path)
    unsigned = {**review, "review_sha256": ""}
    if review.get("review_sha256") != digest_value(unsigned):
        raise ValueError("canary review digest mismatch")
    if review.get("review_status") != "approved":
        raise ValueError("canary review is not approved")
    if review.get("canary_ledger_sha256") != ledger.ledger_sha256:
        raise ValueError("canary review is for a different ledger")
    if not str(review.get("reviewed_by") or "").strip():
        raise ValueError("canary review requires a human reviewer identity")
    if not str(review.get("reviewed_at") or "").strip():
        raise ValueError("canary review requires a review timestamp")
    rows = review.get("rows") or []
    row_ids = [str(row.get("coordinate_id") or "") for row in rows]
    if len(row_ids) != len(planned_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("canary review must contain each coordinate exactly once")
    if set(row_ids) != set(planned_ids):
        raise ValueError("canary review coordinate set does not match the ledger")
    required = {
        "natural_progressive_dialogue",
        "no_hidden_truth_leakage",
        "tool_calling_valid",
        "mutation_confirmation_valid",
        "simulator_conformance_valid",
        "terminal_response_valid",
    }
    failures = []
    for row in rows:
        coordinate_id = str(row.get("coordinate_id") or "")
        if row.get("run_status") != "complete":
            failures.append(f"{coordinate_id}:run_status")
        bundle = ConversationBundle.model_validate(
            load_json(cohort_dir / "bundles" / f"{coordinate_id}.json")
        )
        if bundle.status != row.get("run_status"):
            failures.append(f"{coordinate_id}:bundle_status")
        repaired = sum(
            receipt.outcome == "repaired"
            for receipt in bundle.simulator_conformance_receipts
        )
        if row.get("simulator_conformance_repair_count") != repaired:
            failures.append(f"{coordinate_id}:simulator_conformance_repair_count")
        for field in required:
            if row.get(field) is not True:
                failures.append(f"{coordinate_id}:{field}")
    if failures:
        raise ValueError(f"canary review has unresolved gates: {failures[:10]}")
    return review
