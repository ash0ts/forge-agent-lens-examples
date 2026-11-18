"""A small SQL agent instrumented with NeMo Relay. Relay exports GenAI spans; Agent Lens reads them."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import nemo_relay as nr
from nemo_relay import observability as obs

from .database import TASKS, execute_readonly, fixture, schema_text
from .inference import ForgeInference

SYSTEM = """You answer database questions using tools. Return exactly one JSON object each turn.
Actions:
{"action":"schema"} retrieves database schema.
{"action":"query","sql":"one read-only SQLite SELECT statement"} executes SQL.
{"action":"finish","answer":[[value,...],...]} returns final rows after checking tool output.
Never invent tool results. Retry a transient tool error once. Do not wrap JSON in markdown."""

AGENT_LENS_OTLP = "https://trace.wandb.ai/agents/otel/v1/traces"


def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("action") not in ("schema", "query", "finish"):
        raise ValueError("Unknown agent action")
    if value["action"] == "query" and not isinstance(value.get("sql"), str):
        raise ValueError("SQL must be text")
    if value["action"] == "finish" and not isinstance(value.get("answer"), list):
        raise ValueError("Answer must contain rows")
    return value


def observability(output_dir: Path, *, project: str | None, service_name: str) -> obs.ComponentSpec:
    """One gen_ai projection, two destinations: a local JSONL file and, if a project is given, Agent Lens."""
    promoted = ["task_id", "fault_injection"]  # plain metadata becomes Agent Lens custom attributes
    file_sink = obs.OpenTelemetryFileSinkConfig(output_directory=str(output_dir), filename="relay-genai.jsonl",
                                                type="gen_ai", service_name=service_name,
                                                promote_metadata_prefixes=promoted)
    endpoints = []
    if project:
        endpoints.append(obs.OpenTelemetryEndpointConfig(
            type="gen_ai", endpoint=AGENT_LENS_OTLP, service_name=service_name,
            headers={"project_id": project}, header_env={"wandb-api-key": "WANDB_API_KEY"},
            promote_metadata_prefixes=promoted, timeout_millis=30000))
    return obs.ComponentSpec(obs.ObservabilityConfig(
        enable_full_payloads=True,
        opentelemetry=obs.OpenTelemetrySectionConfig(enabled=True, endpoints=endpoints, file_sinks=[file_sink])))


async def run_agent(output_dir: Path, *, project: str | None, model: str, fault_task: str = "largest_paid_order",
                    seed: int = 11, max_turns: int = 6, conversation_id: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    conversation_id = conversation_id or str(uuid4())
    service_name = "relay-sql-agent"
    activation = await nr.plugin.initialize(nr.plugin.PluginConfig(
        components=[observability(output_dir, project=project, service_name=service_name)]))
    client = ForgeInference(project or "wandb/nvidia-forge-examples", model=model)
    cases = []
    try:
        for index, task in enumerate(TASKS):
            db = fixture(seed)
            expected = execute_readonly(db, task.reference_sql)
            fault = task.id == fault_task
            history, steps, injected, answer, status = [], [], False, None, "turn_limit"
            started = time.perf_counter()
            with nr.scope.scope("sql-agent", nr.ScopeType.Agent,
                                input={"question": task.question, "fault_injection": fault},
                                metadata={"gen_ai.conversation.id": conversation_id, "task_id": task.id,
                                          "fault_injection": str(fault).lower()}):
                for turn in range(max_turns):
                    prompt = task.question + "\nTool history:\n" + json.dumps(history)
                    request = nr.LLMRequest({}, {"model": model, "messages": [
                        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})

                    async def invoke(_request, _prompt=prompt):
                        result = client.ask(SYSTEM, _prompt, max_tokens=2048)
                        if result.get("error"):
                            raise RuntimeError(result["error"])
                        return {"id": result["response_id"], "model": result["model"],
                                "choices": [{"message": {"role": "assistant", "content": result["text"]},
                                             "finish_reason": result["finish_reason"]}],
                                "usage": result.get("usage", {})}

                    try:
                        response = await nr.llm.execute("forge-model", request, invoke, model_name=model,
                                                        response_codec=nr.codecs.OpenAIChatCodec())
                        action = parse_action(response["choices"][0]["message"]["content"])
                    except (ValueError, RuntimeError) as exc:
                        steps.append({"turn": turn, "error": str(exc)})
                        status = "model_or_protocol_error"
                        break
                    step = {"turn": turn, "action": action}
                    steps.append(step)
                    if action["action"] == "finish":
                        answer, status = action["answer"], "completed"
                        break

                    async def tool(args, _action=action):
                        nonlocal injected
                        if _action["action"] == "schema":
                            return nr.ToolExecutionResult({"schema": schema_text(db)})
                        if fault and not injected:
                            injected = True
                            raise TimeoutError("Injected one-time database timeout")
                        return nr.ToolExecutionResult({"rows": execute_readonly(db, args["sql"])})

                    try:
                        result = await nr.tools.execute(action["action"], action, tool)
                        value = result.result if hasattr(result, "result") else result
                        step["output"] = value
                        history.append({"request": action, "response": value})
                    except Exception as exc:
                        step["error"] = f"{type(exc).__name__}: {exc}"
                        history.append({"request": action, "error": step["error"]})
            db.close()
            cases.append({"case_id": f"{task.id}-{index}", "task_id": task.id, "question": task.question,
                          "status": status, "answer": answer, "expected": expected,
                          "correct": answer == expected, "fault_injected": injected,
                          "turns": len([s for s in steps if "action" in s]),
                          "wall_ms": round((time.perf_counter() - started) * 1000, 1), "steps": steps})
    finally:
        await nr.subscribers.flush_async()
        await activation.close()
        client.close()

    spans = [json.loads(line) for line in (output_dir / "relay-genai.jsonl").read_text().splitlines() if line.strip()]
    result = {"conversation_id": conversation_id, "project": project, "model": model, "seed": seed,
              "cases": cases, "span_records": len(spans),
              "metrics": {"tasks": len(cases), "correct": sum(c["correct"] for c in cases),
                          "fault_injected": sum(c["fault_injected"] for c in cases),
                          "model_calls": len(client.calls)},
              "agent_lens_url": (f"https://forge.coreweave.com/agent-lens/{project}/conversations/{conversation_id}"
                                 if project else None),
              "limitations": ["One agent, three tasks, one synthetic SQLite database.",
                              "The timeout is injected on purpose to show recovery; it is not a measured failure rate.",
                              "This shows Relay's GenAI export reaching Agent Lens, not Relay's general compatibility."]}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
