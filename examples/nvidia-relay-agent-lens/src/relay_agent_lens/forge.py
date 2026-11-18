"""Companion W&B run: a 3-task EvalTable, the span ledger, and a verify step that reads Agent Lens back."""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

INPUT_COLUMNS = ["case_id", "task_id", "question", "fault_injected"]
OUTPUT_COLUMNS = ["answer", "turns", "wall_ms", "status"]
SCORE_COLUMNS = ["correct", "recovered_from_timeout"]


def _span_rows(output_dir: Path) -> list[dict]:
    rows = []
    for line in (output_dir / "relay-genai.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        for rs in json.loads(line)["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                for s in ss["spans"]:
                    attrs = {a["key"]: list(a["value"].values())[0] for a in s.get("attributes", [])}
                    start, end = int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"])
                    exc = next((e for e in s.get("events", []) if e.get("name") == "exception"), None)
                    exc_type = ""
                    if exc:
                        exc_type = next((list(a["value"].values())[0] for a in exc.get("attributes", [])
                                         if a["key"] == "exception.type"), "")
                    rows.append({"span_id": s["spanId"], "parent_span_id": s.get("parentSpanId", ""),
                                 "name": s["name"], "operation": attrs.get("gen_ai.operation.name", ""),
                                 "start_ns": start, "duration_ms": round((end - start) / 1e6, 1),
                                 "status_code": s.get("status", {}).get("code", 0),
                                 "status_message": s.get("status", {}).get("message", ""),
                                 "error_type": attrs.get("error.type", ""), "exception_type": exc_type,
                                 "task_id": attrs.get("task_id", ""),
                                 "input_tokens": int(attrs.get("gen_ai.usage.input_tokens", 0) or 0),
                                 "output_tokens": int(attrs.get("gen_ai.usage.output_tokens", 0) or 0)})
    rows.sort(key=lambda r: r["start_ns"])
    t0 = rows[0]["start_ns"] if rows else 0
    for r in rows:
        r["offset_ms"] = round((r["start_ns"] - t0) / 1e6, 1)
        del r["start_ns"]
    return rows


def publish(result: dict, output_dir: Path, *, project: str, run_name: str | None = None) -> dict:
    entity, name = project.split("/", 1)
    spans = _span_rows(output_dir)
    (output_dir / "span-ledger.json").write_text(json.dumps(spans, indent=2))
    run = wandb.init(entity=entity, project=name, name=run_name or "relay-agent-lens", job_type="relay-agent-lens",
                     config={"model": result["model"], "seed": result["seed"], "tasks": result["metrics"]["tasks"],
                             "conversation_id": result["conversation_id"], "otlp_projection": "gen_ai",
                             "exporter": "nemo_relay OpenTelemetryEndpointConfig"})
    try:
        rows = []
        for c in result["cases"]:
            timeouts = [s for s in c["steps"] if "error" in s and "TimeoutError" in s["error"]]
            rows.append([c["case_id"], c["task_id"], c["question"], c["fault_injected"],
                         json.dumps(c["answer"]), c["turns"], c["wall_ms"], c["status"],
                         bool(c["correct"]), bool(c["fault_injected"] and timeouts and c["correct"])])
        eval_table = wandb.EvalTable(columns=INPUT_COLUMNS + OUTPUT_COLUMNS + SCORE_COLUMNS, data=rows,
                                     input_columns=INPUT_COLUMNS, output_columns=OUTPUT_COLUMNS,
                                     score_columns=SCORE_COLUMNS)
        run.log({"agent/tasks": eval_table})
        cols = ["offset_ms", "operation", "name", "duration_ms", "status_code", "status_message", "error_type",
                "exception_type", "task_id", "input_tokens", "output_tokens", "span_id", "parent_span_id"]
        run.log({"agent/span_ledger": wandb.Table(columns=cols, data=[[s[k] for k in cols] for s in spans])})
        error_spans = [s for s in spans if s["status_code"] == 2]
        run.summary.update({
            "agent/tasks_total": result["metrics"]["tasks"], "agent/tasks_correct": result["metrics"]["correct"],
            "agent/model_calls": result["metrics"]["model_calls"], "agent/spans_exported": len(spans),
            "agent/error_spans": len(error_spans),
            "agent/error_types": sorted({s["error_type"] for s in error_spans}),
            "agent/exception_types": sorted({s["exception_type"] for s in error_spans}),
            "agent/input_tokens": sum(s["input_tokens"] for s in spans),
            "agent/output_tokens": sum(s["output_tokens"] for s in spans),
            "agent/conversation_id": result["conversation_id"],
            "agent/agent_lens_url": result["agent_lens_url"] or ""})
        art = wandb.Artifact("relay-agent-lens-evidence", type="evidence",
                             description="Relay gen_ai OTLP export, agent result, and derived span ledger.")
        for f in ("relay-genai.jsonl", "result.json", "span-ledger.json"):
            art.add_file(str(output_dir / f))
        run.log_artifact(art)
        info = {"run_path": f"{entity}/{name}/{run.id}", "run_url": run.url,
                "agent_lens_url": result["agent_lens_url"], "conversation_id": result["conversation_id"]}
    finally:
        run.finish()
    (output_dir / "forge-run.json").write_text(json.dumps(info, indent=2))
    return info


def verify(project: str, conversation_id: str) -> dict:
    """Read the conversation back from Agent Lens and report what landed. Uses only WANDB_API_KEY."""
    from weave.trace_server_bindings.remote_http_trace_server import RemoteHTTPTraceServer

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        import netrc
        key = netrc.netrc().authenticators("api.wandb.ai")[2]
    ts = RemoteHTTPTraceServer("https://trace.wandb.ai", username="api", password=key)
    client, headers = ts._stainless_client, ts._compose_headers()
    spans = client.agents.spans.query(extra_headers=headers, project_id=project, limit=500,
                                      query={"$expr": {"$eq": [{"$getField": "conversation_id"},
                                                                {"$literal": conversation_id}]}}).model_dump()["spans"]
    conv = client.agents.conversations.chat(extra_headers=headers, project_id=project,
                                            conversation_id=conversation_id).model_dump()
    ops = {}
    for s in spans:
        ops[s["operation_name"]] = ops.get(s["operation_name"], 0) + 1
    errors = [{"operation": s["operation_name"], "error_type": s["error_type"], "status_message": s["status_message"],
               "tool_name": s.get("tool_name")} for s in spans if s["status_code"] == "ERROR"]
    return {"project": project, "conversation_id": conversation_id, "spans_in_agent_lens": len(spans),
            "operations": ops, "turns": conv["total_turns"], "total_cost_usd": conv["total_cost_usd"],
            "error_spans": errors,
            "agent_lens_url": f"https://forge.coreweave.com/agent-lens/{project}/conversations/{conversation_id}"}
