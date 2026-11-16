from __future__ import annotations

import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest_file, digest_value, load_json, write_json_atomic
from .constants import PACKAGE_ROOT, RUNTIME_SNAPSHOTS_DIR

_PROMPT_NAMES = (
    "support-agent.md",
    "ordinary-user.md",
    "concise-user.md",
    "skeptical-user.md",
)
_FIXED_INPUTS = (
    "prepared/db.json",
    "prepared/policy.md",
    "prepared/tasks.runtime.json",
    "prepared/user-guidelines.md",
    "private/task-truth.json",
    "model-pricing.lock.json",
    "source.lock.json",
    "telemetry-schema.json",
    "pyproject.toml",
    "uv.lock",
)


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Verified immutable inputs for one exact preview identity."""

    root: Path
    preview: Mapping[str, Any]
    manifest: Mapping[str, Any]
    tasks: tuple[dict[str, Any], ...]
    policy: str
    user_guidelines: str
    prompts: Mapping[str, str]
    pricing_lock: Mapping[str, Any]

    @property
    def preview_sha256(self) -> str:
        return str(self.preview["preview_sha256"])

    @property
    def db_path(self) -> Path:
        return self.root / "prepared" / "db.json"

    @property
    def runtime_tasks_path(self) -> Path:
        return self.root / "prepared" / "tasks.runtime.json"

    @property
    def truth_path(self) -> Path:
        return self.root / "private" / "task-truth.json"

    @property
    def policy_path(self) -> Path:
        return self.root / "prepared" / "policy.md"

    @property
    def source_lock_path(self) -> Path:
        return self.root / "source.lock.json"

    @property
    def package_source_dir(self) -> Path:
        return self.root / "src"


def _validated_preview_digest(preview: Mapping[str, Any]) -> str:
    claimed = str(preview.get("preview_sha256") or "")
    unsigned = dict(preview)
    unsigned.pop("preview_sha256", None)
    if len(claimed) != 64 or claimed != digest_value(unsigned):
        raise ValueError("preview digest does not match its canonical payload")
    return claimed


def _source_file_map(preview: Mapping[str, Any]) -> dict[str, Path]:
    result = {
        relative: PACKAGE_ROOT / relative
        for relative in _FIXED_INPUTS
    }
    if preview.get("inputs", {}).get("canary_execution_lock_sha256"):
        result["canary-execution.lock.json"] = (
            PACKAGE_ROOT / "canary-execution.lock.json"
        )
    for name in _PROMPT_NAMES:
        result[f"prompts/{name}"] = PACKAGE_ROOT / "prompts" / name
    runtime_files = preview.get("runtime", {}).get("source_files", {})
    if not isinstance(runtime_files, Mapping) or not runtime_files:
        raise ValueError("preview has no runtime source-file manifest")
    for relative in runtime_files:
        if not isinstance(relative, str) or not relative.startswith(
            "src/retail_support_study_v1/"
        ):
            raise ValueError("preview runtime contains an unsafe source path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("preview runtime contains an unsafe source path")
        result[relative] = PACKAGE_ROOT / relative
    return result


def _write_snapshot_file(path: Path, source: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"snapshot input must be a regular file: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    shutil.copyfile(source, path)
    path.chmod(0o600)


def _set_snapshot_tree_mode(root: Path, *, read_only: bool) -> None:
    """Seal retained bytes against observer-created caches and later edits."""

    file_mode = 0o400 if read_only else 0o600
    directory_mode = 0o500 if read_only else 0o700
    entries = list(root.rglob("*"))
    for path in entries:
        if path.is_file() and not path.is_symlink():
            path.chmod(file_mode)
    for path in sorted(
        (path for path in entries if path.is_dir() and not path.is_symlink()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(directory_mode)
    root.chmod(directory_mode)


def materialize_runtime_snapshot(
    preview: Mapping[str, Any],
    *,
    snapshots_root: Path = RUNTIME_SNAPSHOTS_DIR,
) -> RuntimeSnapshot:
    """Create a write-once snapshot for one exact reviewed preview."""

    preview_sha = _validated_preview_digest(preview)
    snapshots_root = Path(snapshots_root)
    destination = snapshots_root / preview_sha
    if destination.exists() or destination.is_symlink():
        snapshot = verify_runtime_snapshot(
            preview_sha,
            snapshots_root=snapshots_root,
            require_current_code=False,
        )
        if dict(snapshot.preview) != dict(preview):
            raise ValueError("existing runtime snapshot belongs to another preview")
        return snapshot

    snapshots_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.chmod(0o700)
    temporary = snapshots_root / f".{preview_sha}.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("runtime snapshot temporary path already exists")
    temporary.mkdir(mode=0o700)
    try:
        files: dict[str, str] = {}
        preview_path = temporary / "preview.json"
        write_json_atomic(preview_path, dict(preview), mode=0o600)
        files["preview.json"] = digest_file(preview_path)
        for relative, source in sorted(_source_file_map(preview).items()):
            target = temporary / relative
            _write_snapshot_file(target, source)
            files[relative] = digest_file(target)

        manifest: dict[str, Any] = {
            "schema_version": 2,
            "preview_sha256": preview_sha,
            "sealed_read_only": True,
            "files": files,
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = digest_value(
            {**manifest, "manifest_sha256": ""}
        )
        write_json_atomic(temporary / "manifest.json", manifest, mode=0o600)
        _set_snapshot_tree_mode(temporary, read_only=True)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            _set_snapshot_tree_mode(temporary, read_only=False)
            shutil.rmtree(temporary)
        raise
    return verify_runtime_snapshot(
        preview_sha,
        snapshots_root=snapshots_root,
        require_current_code=False,
    )


def _require_snapshot_digest(
    *,
    root: Path,
    files: Mapping[str, Any],
    relative: str,
) -> str:
    expected = files.get(relative)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"runtime snapshot manifest is missing {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"runtime snapshot file is not regular: {relative}")
    if digest_file(path) != expected:
        raise ValueError(f"runtime snapshot file drift: {relative}")
    return expected


def verify_runtime_snapshot(
    preview_sha256: str,
    *,
    snapshots_root: Path = RUNTIME_SNAPSHOTS_DIR,
    require_current_code: bool = True,
) -> RuntimeSnapshot:
    """Verify and load the exact retained runtime for a preview digest."""

    if len(preview_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in preview_sha256
    ):
        raise ValueError("runtime snapshot preview SHA must be lowercase SHA-256")
    root = Path(snapshots_root) / preview_sha256
    if root.is_symlink() or not root.is_dir():
        raise ValueError("exact runtime snapshot is missing")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("runtime snapshot manifest must be a regular file")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("runtime snapshot manifest is malformed")
    claimed_manifest = manifest.get("manifest_sha256")
    if claimed_manifest != digest_value({**manifest, "manifest_sha256": ""}):
        raise ValueError("runtime snapshot manifest digest mismatch")
    if manifest.get("preview_sha256") != preview_sha256:
        raise ValueError("runtime snapshot preview identity mismatch")
    if manifest.get("schema_version") == 2:
        if manifest.get("sealed_read_only") is not True:
            raise ValueError("runtime snapshot is not marked read-only")
        if stat.S_IMODE(root.stat().st_mode) != 0o500:
            raise ValueError("runtime snapshot root permissions drift")
        for path in root.rglob("*"):
            expected_mode = 0o500 if path.is_dir() else 0o400
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise ValueError(
                    "runtime snapshot permissions drift: "
                    f"{path.relative_to(root).as_posix()}"
                )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime snapshot has no file manifest")

    snapshot_entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in snapshot_entries):
        raise ValueError("runtime snapshot cannot contain symlinks")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in snapshot_entries
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    }
    if actual_files != set(files):
        raise ValueError("runtime snapshot file set drift")
    for relative in sorted(files):
        _require_snapshot_digest(root=root, files=files, relative=relative)

    preview = load_json(root / "preview.json")
    if not isinstance(preview, dict):
        raise ValueError("runtime snapshot preview is malformed")
    if _validated_preview_digest(preview) != preview_sha256:
        raise ValueError("runtime snapshot contains another preview")

    input_files = {
        "prepared/db.json": preview["inputs"]["database_sha256"],
        "prepared/policy.md": preview["inputs"]["policy_sha256"],
        "prepared/user-guidelines.md": preview["inputs"][
            "user_guidelines_sha256"
        ],
        "prepared/tasks.runtime.json": preview["inputs"]["runtime_tasks_sha256"],
        "private/task-truth.json": preview["inputs"]["task_truth_sha256"],
        "model-pricing.lock.json": preview["models"]["pricing_lock_sha256"],
        "telemetry-schema.json": preview["telemetry"]["schema_file_sha256"],
        "pyproject.toml": preview["runtime"]["pyproject_sha256"],
        "uv.lock": preview["runtime"]["uv_lock_sha256"],
    }
    canary_execution_lock_sha256 = preview.get("inputs", {}).get(
        "canary_execution_lock_sha256"
    )
    if canary_execution_lock_sha256:
        input_files["canary-execution.lock.json"] = (
            canary_execution_lock_sha256
        )
    input_files.update(
        {
            f"prompts/{name}": digest
            for name, digest in preview["inputs"]["prompt_files"].items()
        }
    )
    input_files.update(preview["runtime"]["source_files"])
    for relative, expected in input_files.items():
        if files.get(relative) != expected:
            raise ValueError(f"runtime snapshot is not bound to preview: {relative}")

    if canary_execution_lock_sha256:
        execution_lock = load_json(root / "canary-execution.lock.json")
        if execution_lock != preview.get("canary_execution"):
            raise ValueError(
                "runtime snapshot canary execution lock differs from preview"
            )

    source_lock = load_json(root / "source.lock.json")
    unsigned_lock = dict(source_lock)
    claimed_lock = unsigned_lock.pop("source_lock_sha256", None)
    if claimed_lock != digest_value(unsigned_lock):
        raise ValueError("runtime snapshot source lock digest mismatch")
    if claimed_lock != preview["source_lock_sha256"]:
        raise ValueError("runtime snapshot source lock is not bound to preview")

    if require_current_code:
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_python != preview["runtime"]["python"]:
            raise ValueError("current Python version differs from runtime snapshot")
        for relative, expected in (
            ("pyproject.toml", preview["runtime"]["pyproject_sha256"]),
            ("uv.lock", preview["runtime"]["uv_lock_sha256"]),
        ):
            current = PACKAGE_ROOT / relative
            if current.is_symlink() or not current.is_file():
                raise ValueError(f"current runtime lock is missing: {relative}")
            if digest_file(current) != expected:
                raise ValueError(f"current runtime lock differs from snapshot: {relative}")
        for relative, expected in preview["runtime"]["source_files"].items():
            current = PACKAGE_ROOT / relative
            if current.is_symlink() or not current.is_file():
                raise ValueError(f"current runtime source is missing: {relative}")
            if digest_file(current) != expected:
                raise ValueError(f"current runtime code differs from snapshot: {relative}")

    raw_tasks = load_json(root / "prepared" / "tasks.runtime.json")
    if not isinstance(raw_tasks, list):
        raise ValueError("runtime snapshot tasks are malformed")
    prompts = {
        name: (root / "prompts" / name).read_text(encoding="utf-8").strip()
        for name in _PROMPT_NAMES
    }
    pricing_lock = load_json(root / "model-pricing.lock.json")
    if not isinstance(pricing_lock, dict):
        raise ValueError("runtime snapshot pricing lock is malformed")
    return RuntimeSnapshot(
        root=root,
        preview=preview,
        manifest=manifest,
        tasks=tuple(dict(task) for task in raw_tasks),
        policy=(root / "prepared" / "policy.md").read_text(encoding="utf-8").strip(),
        user_guidelines=(root / "prepared" / "user-guidelines.md").read_text(
            encoding="utf-8"
        ).strip(),
        prompts=prompts,
        pricing_lock=pricing_lock,
    )
