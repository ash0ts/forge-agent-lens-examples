# Watch an agent survive a tool timeout

A small SQL agent runs on Forge Serverless Inference, instrumented with NVIDIA's NeMo Relay. Relay's `gen_ai` projection posts OpenTelemetry spans straight to Agent Lens. One tool call is made to fail with a `TimeoutError`; you watch the agent retry and finish in the Agent Lens conversation view.

Not reviewed by NVIDIA. Built on the public `nemo-relay` SDK and Forge APIs.

## What you learn

- Relay to Agent Lens needs no collector and no transform. Relay already speaks OpenTelemetry GenAI semantic conventions (`invoke_agent`, `chat`, `execute_tool`, `gen_ai.conversation.id`). Point an `OpenTelemetryEndpointConfig(type="gen_ai")` at `https://trace.wandb.ai/agents/otel/v1/traces` with `project_id` and `wandb-api-key` headers, and the spans land.
- What a failed tool looks like in Agent Lens when it came from Relay: span status `ERROR`, `error.type = internal_error`, and the real exception class (`TimeoutError`) in the span's exception event and status message.
- Where the seams are. Relay reserves the `gen_ai.*` namespace, so you cannot promote your own metadata into `gen_ai.input.messages` on the agent span; Agent Lens builds the user message of each turn from the first `chat` span instead, which worked here.

## Forge surfaces, in reading order

| Step | Surface | Live |
| --- | --- | --- |
| 1 | Agent Lens conversation: three turns, one with a red `execute_tool query` span followed by a successful retry | [conversation `ff8af1e6…`](https://forge.coreweave.com/agent-lens/wandb/nvidia-forge-examples/conversations/ff8af1e6-681c-4b65-8f65-5a7f2113a270) |
| 2 | Agent Lens spans filtered to `status = ERROR` | [Agent Lens project](https://forge.coreweave.com/agent-lens/wandb/nvidia-forge-examples) |
| 3 | Companion run: `agent/tasks` EvalTable (3 tasks, `correct`, `recovered_from_timeout`) and `agent/span_ledger` (20 spans with offsets, durations, status, tokens) | [run `mv1niark`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/mv1niark) |
| 4 | Artifact `relay-agent-lens-evidence`: Relay's raw `gen_ai` JSONL export, the agent result, the derived ledger | same run |

## Results (2026-11-18, live)

| Task | Fault injected | Turns | Correct |
| --- | --- | --- | --- |
| paid_total_by_country | no | 3 | yes |
| no_paid_orders | no | 3 | yes |
| largest_paid_order | yes, first `query` raises `TimeoutError` | 4 | yes |

20 spans exported by Relay, 20 read back from Agent Lens (10 `chat`, 7 `execute_tool`, 3 `invoke_agent`), 1 error span, model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, 2,144 input and 7,177 output tokens.

## Run it

```bash
cd examples/nvidia-relay-agent-lens
uv sync
export WANDB_API_KEY=...   # or ~/.netrc

# 3 tasks, ~10 model calls, spans posted to Agent Lens, companion run published
uv run relay-agent-lens run --project wandb/nvidia-forge-examples --output out/run-01

# Local only: write Relay's gen_ai JSONL to out/local, send nothing to Forge
uv run relay-agent-lens run --output out/local

# Read a conversation back from Agent Lens
uv run relay-agent-lens verify --project wandb/nvidia-forge-examples --conversation <id>
```

## The export config

```python
obs.OpenTelemetryEndpointConfig(
    type="gen_ai",
    endpoint="https://trace.wandb.ai/agents/otel/v1/traces",
    headers={"project_id": "wandb/nvidia-forge-examples"},
    header_env={"wandb-api-key": "WANDB_API_KEY"},
    promote_metadata_prefixes=["task_id", "fault_injection"],
)
```

`enable_full_payloads=True` keeps the full prompt on each `chat` span so the conversation view shows what the model saw.

## Appendix: replaying a recording

If the agent ran somewhere that could not reach Agent Lens, Relay's file sink (`relay-genai.jsonl`) can be replayed through the Forge SDK:

```bash
uv run python scripts/import_recorded_spans.py --file evidence/live-2026-11-18/relay-genai.jsonl \
  --project wandb/nvidia-forge-examples --conversation-id replay-01
```

The replay uses `coreweave.forge.agentlens.tracing` (`start_conversation`, `start_turn`, `start_llm`, `start_tool`, `record_error`). It reproduces the same 20 spans and 3 turns; `error.type` becomes `TimeoutError` because the SDK records the exception class directly.

## Limits

One agent, three tasks, one synthetic SQLite database. The timeout is injected to show recovery; it is not a measured failure rate. This shows Relay's GenAI export reaching Agent Lens, not Relay's general compatibility. `promote_metadata_prefixes` put `task_id` and `fault_injection` on the three `invoke_agent` spans; Agent Lens lists them in its custom attribute schema, so they are filterable, but the child `chat` and `execute_tool` spans do not inherit them.
