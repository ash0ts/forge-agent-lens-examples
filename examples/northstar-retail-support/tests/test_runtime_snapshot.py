from __future__ import annotations

import copy
from pathlib import Path

import pytest

from retail_support_study_v1.canonical import digest_value
from retail_support_study_v1.mcp_protocol import MCPProcessClient
from retail_support_study_v1.preview import build_preview
from retail_support_study_v1.runtime_snapshot import (
    materialize_runtime_snapshot,
    verify_runtime_snapshot,
)


def test_runtime_snapshot_is_complete_and_runs_its_own_mcp_package(
    tmp_path: Path,
) -> None:
    preview = build_preview()
    snapshot = materialize_runtime_snapshot(preview, snapshots_root=tmp_path)

    assert snapshot.preview_sha256 == preview["preview_sha256"]
    assert len(snapshot.tasks) == 114
    assert snapshot.db_path.is_file()
    assert snapshot.pricing_lock["models"]
    assert snapshot.manifest["manifest_sha256"]
    assert snapshot.manifest["sealed_read_only"] is True
    assert snapshot.manifest["files"]["prepared/user-guidelines.md"] == (
        preview["inputs"]["user_guidelines_sha256"]
    )
    assert snapshot.manifest["files"]["telemetry-schema.json"] == (
        preview["telemetry"]["schema_file_sha256"]
    )
    assert "canary-execution.lock.json" not in snapshot.manifest["files"]

    with MCPProcessClient(
        db_path=snapshot.db_path,
        policy_path=snapshot.policy_path,
        package_source_dir=snapshot.package_source_dir,
    ) as mcp:
        assert len(mcp.list_tools()) == 16

    with pytest.raises(PermissionError):
        (snapshot.package_source_dir / "observer-cache.pyc").write_bytes(b"cache")


def test_runtime_snapshot_fails_closed_on_file_drift(tmp_path: Path) -> None:
    preview = build_preview()
    snapshot = materialize_runtime_snapshot(preview, snapshots_root=tmp_path)
    snapshot.policy_path.chmod(0o600)
    snapshot.policy_path.write_text("changed after approval", encoding="utf-8")
    snapshot.policy_path.chmod(0o400)

    with pytest.raises(ValueError, match="runtime snapshot file drift"):
        verify_runtime_snapshot(
            preview["preview_sha256"],
            snapshots_root=tmp_path,
            require_current_code=False,
        )


def test_runtime_snapshot_rejects_traversal_in_source_manifest(
    tmp_path: Path,
) -> None:
    preview = copy.deepcopy(build_preview())
    preview["runtime"]["source_files"][
        "src/retail_support_study_v1/../../escape.py"
    ] = "0" * 64
    unsigned = dict(preview)
    unsigned.pop("preview_sha256")
    preview["preview_sha256"] = digest_value(unsigned)

    with pytest.raises(ValueError, match="unsafe source path"):
        materialize_runtime_snapshot(preview, snapshots_root=tmp_path)


def test_v2_runtime_snapshot_does_not_copy_the_frozen_v1_recovery_lock(
    tmp_path: Path,
) -> None:
    preview = build_preview()
    snapshot = materialize_runtime_snapshot(preview, snapshots_root=tmp_path)

    assert "canary_execution" not in preview
    assert "canary_execution_lock_sha256" not in preview["inputs"]
    assert "canary-execution.lock.json" not in snapshot.manifest["files"]
