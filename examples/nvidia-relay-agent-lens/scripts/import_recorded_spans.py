"""Appendix: replay a recorded Relay gen_ai JSONL file into Agent Lens through the Forge SDK.

Use this when the agent ran somewhere that could not reach Agent Lens (air-gapped box, CI
without credentials) and you only have Relay's file sink. The live path in
``relay_agent_lens.agent`` does not need this; it posts OTLP straight from Relay.

    uv run python scripts/import_recorded_spans.py --file out/run/relay-genai.jsonl \
        --project wandb/nvidia-forge-examples
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from coreweave.forge.agentlens import tracing


def _attrs(span: dict) -> dict:
    return {a["key"]: list(a["value"].values())[0] for a in span.get("attributes", [])}


def _ts(ns: str) -> datetime:
    return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc)


def load_spans(path: Path) -> list[dict]:
    spans = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        for rs in json.loads(line)["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                spans.extend(ss["spans"])
    spans.sort(key=lambda s: int(s["startTimeUnixNano"]))
    return spans


def replay(path: Path, project: str, *, conversation_id: str | None = None) -> str:
    spans = load_spans(path)
    by_parent: dict[str, list[dict]] = {}
    for s in spans:
        by_parent.setdefault(s.get("parentSpanId", ""), []).append(s)
    agents = [s for s in spans if _attrs(s).get("gen_ai.operation.name") == "invoke_agent"]
    conversation_id = conversation_id or f"replay-{uuid4()}"

    session = tracing.init(project, autopatch_integrations=False)
    tracing.start_conversation(agent_name=_attrs(agents[0]).get("gen_ai.agent.name", "agent") if agents else "agent",
                               conversation_id=conversation_id, conversation_name="relay-replay")
    for agent in agents:
        children = by_parent.get(agent["spanId"], [])
        first_chat = next((c for c in children if _attrs(c).get("gen_ai.operation.name") == "chat"), None)
        user_text = ""
        if first_chat:
            msgs = json.loads(_attrs(first_chat).get("gen_ai.input.messages", "[]"))
            user_text = next((p.get("content", "") for m in msgs if m.get("role") == "user"
                              for p in m.get("parts", []) if p.get("type") == "text"), "")
        turn = tracing.start_turn(user_message=user_text, agent_name=_attrs(agent).get("gen_ai.agent.name", ""))
        turn.started_at = _ts(agent["startTimeUnixNano"])
        for child in children:
            a = _attrs(child)
            op = a.get("gen_ai.operation.name")
            if op == "chat":
                llm = tracing.start_llm(model=a.get("gen_ai.request.model", ""), provider_name=a.get("gen_ai.provider.name", ""))
                llm.started_at = _ts(child["startTimeUnixNano"])
                llm.response_id = a.get("gen_ai.response.id", "")
                llm.response_model = a.get("gen_ai.response.model", "")
                llm.usage = tracing.Usage(input_tokens=int(a.get("gen_ai.usage.input_tokens", 0) or 0),
                                          output_tokens=int(a.get("gen_ai.usage.output_tokens", 0) or 0))
                for key, target in (("gen_ai.input.messages", "input_messages"), ("gen_ai.output.messages", "output_messages")):
                    for m in json.loads(a.get(key, "[]")):
                        text = "".join(p.get("content", "") for p in m.get("parts", []) if p.get("type") == "text")
                        getattr(llm, target).append(tracing.Message(role=m.get("role", "assistant"), content=text))
                llm.ended_at = _ts(child["endTimeUnixNano"])
                tracing.end_llm()
            elif op == "execute_tool":
                tool = tracing.start_tool(name=a.get("gen_ai.tool.name", "tool"), arguments=a.get("gen_ai.tool.call.arguments", ""))
                tool.started_at = _ts(child["startTimeUnixNano"])
                tool.result = a.get("gen_ai.tool.call.result", "")
                status = child.get("status", {})
                if status.get("code") == 2:
                    exc = next((e for e in child.get("events", []) if e.get("name") == "exception"), None)
                    exc_type = next((list(x["value"].values())[0] for x in (exc or {}).get("attributes", [])
                                     if x["key"] == "exception.type"), a.get("error.type", "error"))
                    error = type(exc_type, (RuntimeError,), {"__module__": "builtins"})(status.get("message", ""))
                    tool.record_error(error)  # error.type becomes the recorded exception class
                tool.ended_at = _ts(child["endTimeUnixNano"])
                tool.end()
        turn.ended_at = _ts(agent["endTimeUnixNano"])
        tracing.end_turn()
    tracing.end_conversation()
    tracing.force_flush()
    tracing.shutdown()
    return f"https://forge.coreweave.com/agent-lens/{project}/conversations/{conversation_id}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--conversation-id", default=None)
    args = p.parse_args()
    print(replay(args.file, args.project, conversation_id=args.conversation_id))
