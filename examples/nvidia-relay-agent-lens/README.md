# Watch an agent survive a tool timeout

Companion code and evidence for the CoreWeave Forge Report [**Watch an agent survive a tool timeout**](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/reports/Watch-an-agent-survive-a-tool-timeout--VmlldzoxODMzMTAxNA==).

Runs a small SQLite agent on Forge Serverless Inference inside NVIDIA's NeMo Relay runtime (`nemo-relay==0.11.1`) and exports OpenTelemetry GenAI spans straight to CoreWeave Agent Lens (`type="gen_ai"`, no collector). One database query is forced to fail with a `TimeoutError` so you can inspect the error span, the retry, and the verified answer in one conversation.

Built on the public `nemo-relay` SDK and Forge APIs. Not reviewed by NVIDIA.

## What this example covers

- Pointing Relay's `OpenTelemetryEndpointConfig(type="gen_ai")` directly at `https://trace.wandb.ai/agents/otel/v1/traces` with `project_id` and `wandb-api-key` headers.
- Diffing Relay's local `gen_ai` JSONL file sink against Agent Lens readback to verify exact span parity (`20` spans out, `20` read back across `3` turns).
- Three concrete attribute behaviors to know when building filters on Relay traces:
  1. Relay sets `error.type = internal_error` on a failed tool span and records `TimeoutError` in the span's `exception` event (`exception.type`) and `status.message`. Filter on `status = ERROR` or message text rather than `error.type = TimeoutError`.
  2. Relay writes `gen_ai.input.messages` on `chat` spans rather than `invoke_agent`; Agent Lens populates each turn's user message from the first `chat` span.
  3. `promote_metadata_prefixes=["task_id", "fault_injection"]` attaches those attributes to the `invoke_agent` spans (indexed in Agent Lens custom attributes) without copying them onto child `chat` or `execute_tool` spans.

## Forge surfaces, in reading order

| Step | Surface | Live |
| --- | --- | --- |
| 1 | Agent Lens conversation: 3 turns across 56.9s; Turn 3 carries the red `execute_tool query` error span, the retry, and the final answer | [conversation `ff8af1e6…`](https://forge.coreweave.com/agent-lens/wandb/nvidia-forge-examples/conversations/ff8af1e6-681c-4b65-8f65-5a7f2113a270) (sign-in required) |
| 2 | Public companion Workspace view: 3-task `wandb.EvalTable` (`correct`, `recovered_from_timeout`) + 20-span ledger | [Workspace view `?nw=kuzuutugvcv`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples?nw=kuzuutugvcv) · [run `mv1niark`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/mv1niark) (no login needed) |
| 3 | Artifact `relay-agent-lens-evidence`: Relay's raw `relay-genai.jsonl` export, `result.json`, and `span-ledger.json` | same run |

## Results (2026-11-18, live)

| Turn / Task | Fault injected | Model calls | Tool calls | Correct |
| --- | --- | --- | --- | --- |
| 1 · `paid_total_by_country` | no | 3 | 2 (`schema`, `query`) | yes (`[["GB", 1498], ["US", 3252]]`) |
| 2 · `no_paid_orders` | no | 3 | 2 (`schema`, `query`) | yes (`[[3, "customer_3"], [4, "customer_4"], [6, "customer_6"]]`) |
| 3 · `largest_paid_order` | yes, first `query` raises `TimeoutError` | 4 | 3 (`schema`, failed `query`, retry `query`) | yes (`[[1106, 1244]]`) |

`20` spans exported by Relay, `20` read back from Agent Lens (`3 invoke_agent`, `10 chat`, `7 execute_tool`), `1` error span, model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, `2,144` input and `7,177` output tokens (includes Nemotron thinking tokens).

## Run it

```bash
cd examples/nvidia-relay-agent-lens
uv sync
uv run pytest -q

export WANDB_API_KEY=...   # or ~/.netrc

# Live OTLP export to Agent Lens + companion Forge run
uv run relay-agent-lens run --project wandb/nvidia-forge-examples --output out/run-01

# Local file sink only (writes out/local/relay-genai.jsonl without Forge egress)
uv run relay-agent-lens run --output out/local

# Read a conversation back from Agent Lens
uv run relay-agent-lens verify --project wandb/nvidia-forge-examples --conversation ff8af1e6-681c-4b65-8f65-5a7f2113a270
```

## Appendix: offline JSONL replay

When an agent runs in an air-gapped environment and can only write Relay's file sink (`relay-genai.jsonl`), replay the recorded spans through `coreweave.forge.agentlens.tracing`:

```bash
uv run python scripts/import_recorded_spans.py \
  --file evidence/live-2026-11-18/relay-genai.jsonl \
  --project wandb/nvidia-forge-examples --conversation-id replay-01
```

Replaying the file reproduces the same `20` spans and `3` turns with original nanosecond timestamps; `error.type` becomes `TimeoutError` because the Forge SDK records the exception class directly.

## Limits

One SQL agent, three seeded tasks, one synthetic database, and one planted `TimeoutError`. Demonstrates Relay's `gen_ai` OTLP export landing in Agent Lens, not natural failure rates or Relay throughput/guardrail overhead.
