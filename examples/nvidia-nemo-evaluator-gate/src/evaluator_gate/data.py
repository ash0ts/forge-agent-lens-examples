"""Frozen inputs: tickets, prompts and policy. Everything is hashed before any model call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LABELS = ("inference", "training", "observability", "billing")
SPLITS = ("development", "heldout")
VARIANTS = ("baseline", "candidate")

DEFAULT_POLICY = {
    "candidate_heldout_accuracy_min": 0.85,
    "group_accuracy_min": 0.75,
    "max_paired_regressions": 0,
    "min_heldout_delta": 0.0,
    "require_complete": True,
    "require_strict_label": True,
    "required_heldout_cases": 48,
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def load_tickets(path: Path | None = None) -> dict:
    data = load_json(path or DATA_DIR / "tickets.json")
    validate_tickets(data)
    return data


def load_prompts(path: Path | None = None) -> dict[str, str]:
    prompts = load_json(path or DATA_DIR / "prompts.json")
    for name in VARIANTS + ("control",):
        text = prompts.get(name)
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError(f"prompts.json needs a non-empty '{name}' prompt under 8000 characters")
    return prompts


def load_policy(path: Path | None = None) -> dict:
    policy = {**DEFAULT_POLICY, **load_json(path or DATA_DIR / "policy.json")}
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError(f"Unknown policy keys: {sorted(set(policy) - set(DEFAULT_POLICY))}")
    for key in ("candidate_heldout_accuracy_min", "group_accuracy_min"):
        if not isinstance(policy[key], (int, float)) or isinstance(policy[key], bool) or not 0 <= policy[key] <= 1:
            raise ValueError(f"{key} must be a number in [0, 1]")
    for key in ("require_complete", "require_strict_label"):
        if type(policy[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("required_heldout_cases", "max_paired_regressions"):
        if type(policy[key]) is not int or policy[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not -1 <= policy["min_heldout_delta"] <= 1:
        raise ValueError("min_heldout_delta must be in [-1, 1]")
    return policy


def validate_tickets(data: dict) -> list[dict]:
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("tickets.json must contain a non-empty 'cases' list")
    if tuple(data.get("labels", ())) != LABELS:
        raise ValueError(f"tickets.json labels must be {LABELS}")
    seen = set()
    for case in cases:
        for key in ("case_id", "prompt", "expected", "group", "split"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"case missing '{key}': {case}")
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id {case['case_id']}")
        seen.add(case["case_id"])
        if case["expected"] not in LABELS or case["group"] != case["expected"] or case["split"] not in SPLITS:
            raise ValueError(f"invalid label or split in {case['case_id']}")
    return cases


def control_cases(cases: list[dict], per_label: int = 2) -> list[dict]:
    """Two held-out tickets per label for the deliberately bad prompt."""
    return [row for label in LABELS
            for row in [c for c in cases if c["split"] == "heldout" and c["group"] == label][:per_label]]
