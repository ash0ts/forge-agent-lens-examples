"""Publish upgraded NVIDIA NeMo Evaluator & NeMo Relay Forge Reports with native PanelGrids.

Uploads all referenced article PNGs to a public W&B asset run in `wandb/nvidia-forge-examples`
(verified via unauthenticated HTTP readback), updates both live Forge Reports in place with
`wr.TableOfContents()` + live interactive `wr.PanelGrid` blocks, and records the 3-way sync
baseline (`graphql-preserve-panels`) so `sync_forge_report.py` protects the native panels.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
import wandb
import wandb_workspaces.reports.v2 as wr

HOME = Path.home()
SKILLS_SCRIPTS = Path(os.environ.get("CREATE_ARTICLE_SCRIPTS", HOME / ".codex/skills/create-article/scripts"))
sys.path.insert(0, str(SKILLS_SCRIPTS))
from _article_tools import DEFAULT_LIBRARY, article_dir, read_json
from sync_forge_report import IMAGE_BLOCK_RE, LINKED_IMAGE_RE, get_wandb_api_key

ENTITY = "wandb"
PROJECT = "nvidia-forge-examples"
L = wr.Layout

ARTICLES = {
    "nvidia-nemo-evaluator-gate": {
        "report_id": "VmlldzoxODMzMTAxMw==",
        "description": (
            "How we built a nine-check CI release gate around NVIDIA NeMo Evaluator and "
            "NVIDIA Nemotron 3.5 Lightning on CoreWeave Forge, including a negative control "
            "and a 143x thinking-token catch."
        ),
    },
    "nvidia-relay-agent-lens": {
        "report_id": "VmlldzoxODMzMTAxNA==",
        "description": (
            "Direct GenAI OpenTelemetry export from NVIDIA NeMo Relay into CoreWeave Agent Lens, "
            "tested on a three-task SQLite agent with an injected database timeout."
        ),
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upload_public_assets() -> dict[str, dict[str, str]]:
    receipt_path = Path(
        os.environ.get(
            "REPORT_ASSET_RECEIPT",
            HOME / "Documents/Writing/handoffs/nvidia-npn-marketing-desk-2026-11-20/public-report-assets.json",
        )
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {"files": {}}

    if "run_id" not in receipt:
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            job_type="article-publication",
            name="nvidia-forge-articles-assets",
            settings=wandb.Settings(x_disable_meta=True, disable_git=True, disable_code=True),
            config={"kind": "article-publication-assets", "new_model_runs": 0},
        )
        receipt["run_id"] = run.id
        run.finish()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{receipt['run_id']}")
    anon = requests.Session()
    anon.trust_env = False

    for slug in ARTICLES:
        proj = article_dir(DEFAULT_LIBRARY, slug)
        md_text = (proj / "article.md").read_text(encoding="utf-8")
        cleaned = LINKED_IMAGE_RE.sub(r"![\1](\2)", md_text)
        for m in IMAGE_BLOCK_RE.finditer(cleaned):
            rel = m.group("path").strip()
            local_file = (proj / rel).resolve()
            key = f"{slug}/{rel}"
            digest = sha256_bytes(local_file.read_bytes())
            existing = receipt["files"].get(key)
            if existing and existing.get("sha256") == digest and existing.get("anonymous_verified"):
                continue
            remote = run.upload_file(str(local_file), root=str(DEFAULT_LIBRARY))
            url = remote.url
            assert not urlparse(url).query, f"Refusing expiring signed URL: {url}"
            resp = anon.get(url, timeout=60)
            resp.raise_for_status()
            assert sha256_bytes(resp.content) == digest, f"SHA mismatch on {key}"
            receipt["files"][key] = {
                "url": url,
                "sha256": digest,
                "bytes": len(resp.content),
                "anonymous_verified": True,
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            print(f"Uploaded & anonymous-verified: {key} -> {url}", flush=True)

    return receipt["files"]


def build_evaluator_panels() -> dict[str, wr.PanelGrid]:
    gate_primary_runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Primary live gate run (enable_thinking=False)",
        filters='Name in ["release-gate-nemotron-3.5-lightning-no-thinking"]',
    )
    gate_all_runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="All three release-gate runs (no-thinking, thinking-on, Oct 29 replay)",
        filters=(
            'Name in ["release-gate-nemotron-3.5-lightning-no-thinking", '
            '"release-gate-nemotron-3.5-lightning-thinking-on", '
            '"release-gate-nemotron-3.5-lightning-2026-10-29-replay"]'
        ),
    )

    summary_grid = wr.PanelGrid(
        runsets=[gate_primary_runset],
        panels=[
            wr.ScalarChart(title="gate/ci_exit_code (0 = proceed)", metric="gate/ci_exit_code", layout=L(x=0, y=0, w=6, h=5)),
            wr.ScalarChart(title="gate/heldout_delta (+6/48)", metric="gate/heldout_delta", layout=L(x=6, y=0, w=6, h=5)),
            wr.ScalarChart(title="gate/paired_regressions (budget: 0)", metric="gate/paired_regressions", layout=L(x=12, y=0, w=6, h=5)),
            wr.ScalarChart(title="gate/paired_improvements", metric="gate/paired_improvements", layout=L(x=18, y=0, w=6, h=5)),
            wr.WeavePanelSummaryTable(table_name="release_gate/policy_checks", layout=L(x=0, y=5, w=10, h=10)),
            wr.WeavePanelSummaryTable(table_name="release_gate/paired_changes", layout=L(x=10, y=5, w=14, h=10)),
        ],
    )

    latency_grid = wr.PanelGrid(
        runsets=[gate_all_runset],
        panels=[
            wr.BarPlot(
                title="Median held-out latency (ms)",
                metrics=["baseline/heldout/latency_ms_p50", "candidate/heldout/latency_ms_p50"],
                legend_template="${run:displayName}",
                layout=L(x=0, y=0, w=12, h=9),
            ),
            wr.BarPlot(
                title="Held-out output tokens across 48 tickets",
                metrics=["baseline/heldout/completion_tokens_total", "candidate/heldout/completion_tokens_total"],
                legend_template="${run:displayName}",
                layout=L(x=12, y=0, w=12, h=9),
            ),
            wr.BarPlot(
                title="Accuracy: negative control vs baseline vs candidate",
                metrics=["control/accuracy", "baseline/heldout/accuracy", "candidate/heldout/accuracy"],
                range_x=(0, 1),
                legend_template="${run:displayName}",
                layout=L(x=0, y=9, w=12, h=9),
            ),
            wr.WeavePanelSummaryTable(table_name="release_gate/category_accuracy", layout=L(x=12, y=9, w=12, h=9)),
        ],
    )
    return {"summary": summary_grid, "latency": latency_grid}


def build_relay_panels() -> dict[str, wr.PanelGrid]:
    relay_runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="NeMo Relay SQLite agent companion run",
        filters='JobType == "relay-agent-lens"',
    )
    spans_grid = wr.PanelGrid(
        runsets=[relay_runset],
        panels=[
            wr.ScalarChart(title="agent/spans_exported", metric="agent/spans_exported", layout=L(x=0, y=0, w=6, h=5)),
            wr.ScalarChart(title="agent/error_spans", metric="agent/error_spans", layout=L(x=6, y=0, w=6, h=5)),
            wr.ScalarChart(title="agent/model_calls", metric="agent/model_calls", layout=L(x=12, y=0, w=6, h=5)),
            wr.ScalarChart(title="agent/output_tokens (incl. reasoning)", metric="agent/output_tokens", layout=L(x=18, y=0, w=6, h=5)),
            wr.WeavePanelSummaryTable(table_name="agent/span_ledger", layout=L(x=0, y=5, w=24, h=12)),
        ],
    )
    return {"spans": spans_grid}


def parse_blocks_with_panels(
    slug: str,
    md_text: str,
    asset_map: dict[str, dict[str, str]],
    panel_map: dict[str, wr.PanelGrid],
) -> tuple[str, list]:
    cleaned = LINKED_IMAGE_RE.sub(r"![\1](\2)", md_text)
    lines = cleaned.splitlines()
    assert lines and lines[0].startswith("# "), f"{slug} missing H1"
    title = lines[0][2:].strip()

    blocks: list = [wr.TableOfContents()]
    buf: list[str] = []

    def flush_prose() -> None:
        chunk = "\n".join(buf).strip()
        buf.clear()
        if not chunk:
            return
        cursor = 0
        for m in IMAGE_BLOCK_RE.finditer(chunk):
            before = chunk[cursor : m.start()].strip()
            if before:
                blocks.append(wr.MarkdownBlock(text=before))
            rel = m.group("path").strip()
            key = f"{slug}/{rel}"
            cap = (m.group("caption") or "").strip().strip("*").strip() or m.group("alt").strip()
            blocks.append(wr.Image(url=asset_map[key]["url"], caption=cap))
            cursor = m.end()
        tail = chunk[cursor:].strip()
        if tail:
            blocks.append(wr.MarkdownBlock(text=tail))

    for line in lines[1:]:
        if line.startswith("## "):
            flush_prose()
            blocks.append(wr.H1(text=line[3:].strip()))
        elif line.strip().startswith("> **Interactive Forge Workspace view:**"):
            flush_prose()
            anchor_m = re.search(r"#([a-z0-9_-]+)\)", line)
            anchor = anchor_m.group(1) if anchor_m else next(iter(panel_map))
            blocks.append(panel_map[anchor])
        else:
            buf.append(line)
    flush_prose()
    return title, blocks


def main() -> None:
    os.environ["WANDB_API_KEY"] = get_wandb_api_key()
    asset_map = upload_public_assets()
    all_panels = {
        "nvidia-nemo-evaluator-gate": build_evaluator_panels(),
        "nvidia-relay-agent-lens": build_relay_panels(),
    }

    for slug, cfg in ARTICLES.items():
        proj = article_dir(DEFAULT_LIBRARY, slug)
        md_text = (proj / "article.md").read_text(encoding="utf-8")
        panel_map = all_panels[slug]
        title, blocks = parse_blocks_with_panels(slug, md_text, asset_map, panel_map)

        safe_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/reports/sync--{cfg['report_id']}"
        rep = wr.Report.from_url(safe_url)
        rep.title = title
        rep.description = cfg["description"]
        rep.width = "fluid"
        rep.blocks = blocks
        rep.save(draft=False)

        # Save panel preservation marker & serialized PanelGrid specs
        (proj / "articles_custom_publisher").write_text(
            "Published via examples/nvidia-nemo-evaluator-gate/scripts/publish_nvidia_reports.py with native PanelGrids.\n",
            encoding="utf-8",
        )
        panel_dump = {k: v._to_model().model_dump(mode="json") for k, v in panel_map.items()}
        (proj / "report-panels.json").write_text(json.dumps(panel_dump, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(SKILLS_SCRIPTS / "sync_forge_report.py"),
                "record-baseline",
                "--slug",
                slug,
                "--mode",
                "graphql-preserve-panels",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(SKILLS_SCRIPTS / "sync_forge_report.py"),
                "status",
                "--slug",
                slug,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
