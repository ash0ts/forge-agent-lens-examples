"""Create or update the two public saved Forge Workspace views in wandb/nvidia-forge-examples.

Usage (requires wandb-workspaces):
    python scripts/build_workspace_views.py kzxnkpkxl5n kuzuutugvcv
"""
from __future__ import annotations

import base64
import sys

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
import wandb_workspaces.workspaces.internal as internal
from wandb_workspaces.reports.v2.internal import UnknownPanel

E, P = "wandb", "nvidia-forge-examples"
L = wr.Layout
LS = ws.SectionLayoutSettings

LIVE_NO_THINK = "e0ce3715"
LIVE_THINK = "imy7ehrq"
OCT_REPLAY = "gv5j7kfg"
RELAY_RUN = "mv1niark"
CATS = ["billing", "inference", "observability", "training"]


def run_gid(run_id: str) -> str:
    return base64.b64encode(f"Run:v1:{run_id}:{P}:{E}".encode()).decode()


def eval_table_panel(pid: str, eval_name: str, reference_run: str, compared: list[str], x, y, w, h, hidden=()):
    return {
        "__id__": pid,
        "viewType": "Evaluation Tables",
        "isAuto": False,
        "config": {
            "evalName": eval_name,
            "evalTableType": "eval-table",
            "referenceRunId": run_gid(reference_run),
            "comparedEvals": [{"runId": run_gid(r), "step": 0} for r in compared],
            "aggregatedScoresCollapsed": False,
            "rowSize": "medium",
            "pageSize": 10,
            "columnVisibility": {k: False for k in hidden},
            "columnWidths": {},
        },
        "layout": {"x": x, "y": y, "w": w, "h": h},
    }


def table(name: str, x: int, y: int, w: int = 24, h: int = 8):
    return wr.WeavePanelSummaryTable(table_name=name, layout=L(x=x, y=y, w=w, h=h))


def md(text: str, x: int, y: int, w: int, h: int):
    return wr.MarkdownPanel(markdown=text, layout=L(x=x, y=y, w=w, h=h))


HEIGHTS = {
    "2. Held-out answers side by side (EvalTable)": 900,
    "3. Policy checks and paired ticket fixes": 440,
    "4. Latency and token cost when thinking is left on": 380,
    "5. Per-category accuracy against the 0.75 floor": 420,
    "6. Negative control: the gate has to reject this": 360,
    "7. Every response": 720,
    "1. Three tasks, one injected timeout (EvalTable)": 520,
    "3. Span ledger: offset, duration, status, tokens": 720,
}


def save_with_raw(workspace: ws.Workspace, raw_by_section: dict[str, list[dict]], existing_url_name: str | None):
    view = workspace._to_model()
    if existing_url_name:
        view.name = internal._url_query_str_to_internal_name(existing_url_name)
        old = internal.View.from_name(E, P, view.name)
        view.id = old.id
    for sec in view.spec.section.panel_bank_config.sections:
        if sec.name in HEIGHTS:
            sec.flow_config.box_height = HEIGHTS[sec.name]
        for raw in raw_by_section.get(sec.name, []):
            sec.panels.append(UnknownPanel(**raw))
    resp = internal.upsert_view2(view)
    name = resp["upsertView"]["view"]["name"]
    print(f"https://forge.coreweave.com/wandb/{E}/{P}?nw={internal._internal_name_to_url_query_str(name)}")


gate = ws.Workspace(
    entity=E,
    project=P,
    name="Release gate: NeMo Evaluator on Forge",
    runset_settings=ws.RunsetSettings(
        filters=(
            'Name in ["release-gate-nemotron-3.5-lightning-no-thinking", '
            '"release-gate-nemotron-3.5-lightning-thinking-on", '
            '"release-gate-nemotron-3.5-lightning-2026-10-29-replay"]'
        )
    ),
    sections=[
        ws.Section(
            name="1. The decision CI reads from the run summary",
            is_open=True,
            layout_settings=LS(columns=3, rows=2),
            panels=[
                md(
                    "**Three runs of the same 64 tickets and policy (`data/policy.json`, NeMo Evaluator `exact_match`).**\n\n"
                    "- `...-no-thinking` (2026-11-18 live, `enable_thinking=False`): baseline **42/48**, candidate **48/48**, ~185 ms, 132 output tokens.\n"
                    "- `...-2026-10-29-replay`: rescored Oct 29 capture, same **42/48 → 48/48**.\n"
                    "- `...-thinking-on` (2026-11-18 live, `enable_thinking=True`): baseline **48/48** at **3,039 ms** and **18,830** output tokens.",
                    0, 0, 8, 6,
                ),
                wr.ScalarChart(title="gate/ci_exit_code (0 = proceed)", metric="gate/ci_exit_code", layout=L(x=8, y=0, w=8, h=6)),
                wr.ScalarChart(title="gate/heldout_delta (candidate minus baseline)", metric="gate/heldout_delta", layout=L(x=16, y=0, w=8, h=6)),
                wr.ScalarChart(title="gate/paired_regressions (budget: 0)", metric="gate/paired_regressions", layout=L(x=0, y=6, w=8, h=6)),
                wr.ScalarChart(title="gate/paired_improvements (candidate fixes)", metric="gate/paired_improvements", layout=L(x=8, y=6, w=8, h=6)),
                wr.ScalarChart(title="control/correct (of 8; must fail floor)", metric="control/correct", layout=L(x=16, y=6, w=8, h=6)),
            ],
        ),
        ws.Section(
            name="2. Held-out answers side by side (EvalTable)",
            is_open=True,
            layout_settings=LS(columns=1, rows=1),
            panels=[],
        ),
        ws.Section(
            name="3. Policy checks and paired ticket fixes",
            is_open=True,
            layout_settings=LS(columns=2, rows=1),
            panels=[
                table("release_gate/policy_checks", 0, 0, 10, 10),
                table("release_gate/paired_changes", 10, 0, 14, 10),
            ],
        ),
        ws.Section(
            name="4. Latency and token cost when thinking is left on",
            is_open=True,
            layout_settings=LS(columns=3, rows=1),
            panels=[
                wr.BarPlot(
                    title="Median held-out latency (ms)",
                    metrics=["baseline/heldout/latency_ms_p50", "candidate/heldout/latency_ms_p50"],
                    layout=L(x=0, y=0, w=8, h=8),
                ),
                wr.BarPlot(
                    title="Held-out output tokens (48 tickets)",
                    metrics=["baseline/heldout/completion_tokens_total", "candidate/heldout/completion_tokens_total"],
                    layout=L(x=8, y=0, w=8, h=8),
                ),
                md(
                    "Leaving `enable_thinking` on let the one-line baseline reach 48/48 by spending ~390 reasoning tokens "
                    "and ~3.0 seconds per ticket. With `enable_thinking=False`, the candidate prompt hits the same 48/48 in "
                    "3 tokens and ~191 ms per ticket.",
                    16, 0, 8, 8,
                ),
            ],
        ),
        ws.Section(
            name="5. Per-category accuracy against the 0.75 floor",
            is_open=True,
            layout_settings=LS(columns=3, rows=1),
            panels=[
                wr.BarPlot(
                    title="Baseline, held-out, by category",
                    metrics=[f"baseline/heldout/{c}/accuracy" for c in CATS],
                    range_x=(0, 1),
                    layout=L(x=0, y=0, w=8, h=8),
                ),
                wr.BarPlot(
                    title="Candidate, held-out, by category",
                    metrics=[f"candidate/heldout/{c}/accuracy" for c in CATS],
                    range_x=(0, 1),
                    layout=L(x=8, y=0, w=8, h=8),
                ),
                table("release_gate/category_accuracy", 16, 0, 8, 8),
            ],
        ),
        ws.Section(
            name="6. Negative control: the gate has to reject this",
            is_open=True,
            layout_settings=LS(columns=2, rows=1),
            panels=[
                wr.BarPlot(
                    title="Held-out accuracy: control vs baseline vs candidate",
                    metrics=["control/accuracy", "baseline/heldout/accuracy", "candidate/heldout/accuracy"],
                    range_x=(0, 1),
                    layout=L(x=0, y=0, w=12, h=8),
                ),
                md(
                    "The control prompt ignores the ticket and answers `billing` every time. It gets the 2 billing tickets "
                    "out of 8 (0.25) and fails `candidate_absolute_floor` (0.85) and `every_category_floor` (0.75). "
                    "If `negative_control_rejected` is ever false, the gate holds the release.",
                    12, 0, 12, 8,
                ),
            ],
        ),
        ws.Section(
            name="7. Every response",
            is_open=False,
            layout_settings=LS(columns=1, rows=1),
            panels=[table("release_gate/all_responses", 0, 0, 24, 12)],
        ),
    ],
)

relay = ws.Workspace(
    entity=E,
    project=P,
    name="Relay agent: companion to the Agent Lens conversation",
    runset_settings=ws.RunsetSettings(filters='JobType == "relay-agent-lens"'),
    sections=[
        ws.Section(
            name="1. Three tasks, one injected timeout (EvalTable)",
            is_open=True,
            layout_settings=LS(columns=1, rows=1),
            panels=[],
        ),
        ws.Section(
            name="2. The 20 spans Relay exported",
            is_open=True,
            layout_settings=LS(columns=3, rows=2),
            panels=[
                md(
                    "The conversation lives in Agent Lens: "
                    "[open conversation `ff8af1e6…`](https://forge.coreweave.com/agent-lens/wandb/nvidia-forge-examples/conversations/ff8af1e6-681c-4b65-8f65-5a7f2113a270). "
                    "This companion run records task correctness and the 20-span ledger from Relay's `gen_ai` export.",
                    0, 0, 8, 6,
                ),
                wr.ScalarChart(title="agent/spans_exported", metric="agent/spans_exported", layout=L(x=8, y=0, w=8, h=6)),
                wr.ScalarChart(title="agent/error_spans", metric="agent/error_spans", layout=L(x=16, y=0, w=8, h=6)),
                wr.ScalarChart(title="agent/model_calls", metric="agent/model_calls", layout=L(x=0, y=6, w=8, h=6)),
                wr.ScalarChart(title="agent/input_tokens", metric="agent/input_tokens", layout=L(x=8, y=6, w=8, h=6)),
                wr.ScalarChart(title="agent/output_tokens (includes reasoning)", metric="agent/output_tokens", layout=L(x=16, y=6, w=8, h=6)),
            ],
        ),
        ws.Section(
            name="3. Span ledger: offset, duration, status, tokens",
            is_open=True,
            layout_settings=LS(columns=1, rows=1),
            panels=[table("agent/span_ledger", 0, 0, 24, 12)],
        ),
    ],
)

gate_raw = {
    "2. Held-out answers side by side (EvalTable)": [
        eval_table_panel("nv-gate-heldout", "release_gate/heldout", LIVE_NO_THINK, [LIVE_THINK, OCT_REPLAY], 0, 0, 24, 14)
    ]
}
relay_raw = {
    "1. Three tasks, one injected timeout (EvalTable)": [
        eval_table_panel("nv-relay-tasks", "agent/tasks", RELAY_RUN, [], 0, 0, 24, 9)
    ]
}

if __name__ == "__main__":
    save_with_raw(gate, gate_raw, sys.argv[1] if len(sys.argv) > 1 else None)
    save_with_raw(relay, relay_raw, sys.argv[2] if len(sys.argv) > 2 else None)
