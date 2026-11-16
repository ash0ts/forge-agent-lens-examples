# Agent Lens retail agent demo

This is a working retail-support agent with a small customer-facing chat UI and
live [Agent Lens](https://forge.coreweave.com/agent-lens) telemetry inside
CoreWeave Forge. A customer can ask about an order, return, delivery, or account
issue; the application routes the request to a specialist, runs a real support
model, calls an account-scoped MCP server, and records the resulting model and
tool activity.

The demo is meant to make an agent system inspectable, not merely make a chat
window look convincing. The UI includes a compact receipt for the current
conversation, while Agent Lens keeps the complete OpenTelemetry trace for
searching and reviewing across conversations.

## What is included

- Three support agents: order management, returns resolution, and a generalist.
- Five selectable support-model routes across Forge Serverless Inference and Anthropic.
- A 16-tool MCP retail catalog, with a reviewed subset exposed to each
  specialist.
- Measured wall-clock tool spans, production-shaped latency, and rare injected
  read failures that the agent can recover from.
- Prepare-and-confirm protection before any mutation.
- OpenTelemetry GenAI and MCP spans emitted to Agent Lens.
- An optional [TypeSafe Jev](https://docs.typesafe.ai/introduction) decision
  layer for routing and post-turn behavior checks.

All bundled customers, accounts, orders, and products are fictional. The
retail environment is derived from the MIT-licensed τ³ retail benchmark, but
the reusable demo does not expose benchmark answers, private review labels, or
provider credentials.

## How the pieces connect

```text
customer message
      │
      ├── optional Jev route decision
      ▼
support agent + selected chat model
      │
      ├── MCP reads and confirmed mutations
      ▼
customer response
      │
      └── optional Jev behavior check

model calls + MCP calls + Jev decisions ──OpenTelemetry──▶ Agent Lens
```

Jev does not write the customer response. It answers narrow, typed questions
that the application can use in code. Agent Lens does not route the request or
repair the agent; it records the work so people can inspect individual traces
and recurring patterns.

## Run it locally

You need Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
# Add the provider keys you intend to use.
set -a
source .env
set +a

uv sync --python 3.12 --extra dev
uv run retail-support-chat
```

Open [http://127.0.0.1:4180](http://127.0.0.1:4180).

The credentials stay on the server:

| Variable | Used for |
| --- | --- |
| `WANDB_API_KEY` | Forge Serverless Inference models and Agent Lens telemetry |
| `ANTHROPIC_API_KEY` | Direct Claude model routes |
| `TYPESAFE_API_KEY` | Optional live Jev decisions |

`WANDB_API_KEY` is required for the default Forge Serverless Inference route and for Agent Lens
export. `ANTHROPIC_API_KEY` adds the direct Claude routes. An Anthropic-only
conversation can run, but it will not appear in Agent Lens without the W&B
key.

By default, interactive conversations join the same project as the existing
v10 retail traffic:

- [Agent Lens: `wandb/agent-lens-retail-support-live-v10`](https://forge.coreweave.com/agent-lens/wandb/agent-lens-retail-support-live-v10)
- Forge Serverless Inference accounting: `wandb/agent-lens-retail-chat-demo-v1`

Set `AGENT_LENS_PROJECT` or `WANDB_INFERENCE_PROJECT` in `.env` to use an
isolated project instead.

## Use Jev

Start with Jev in shadow mode:

```bash
RETAIL_DEMO_JEV_MODE=shadow
RETAIL_DEMO_JEV_BACKEND=live
TYPESAFE_API_KEY=...
```

On the first customer turn, Jev receives the first visible message and chooses
between the three support roles. In `shadow` mode the existing deterministic
router remains authoritative; the Jev choice, probability distribution, and
confidence are recorded for inspection. In `route` mode, a Jev choice above
the configured threshold can select the support agent. Missing or uncertain
decisions fall back to the deterministic router.

After each support response, Jev independently estimates three behaviors:

- repeated confirmation of the same action;
- incomplete handling of a multi-part request; and
- a definitive answer without supporting evidence.

These are observations, not verdicts on the whole conversation. With the live
backend, the TypeSafe requests appear as `evaluate` spans beside the
support-model and MCP spans, rather than as another chat model in the Agent
Lens overview.

| Mode | Behavior |
| --- | --- |
| `off` | Jev is not called. |
| `shadow` | Record Jev decisions without changing routing. |
| `route` | Apply a sufficiently confident route; otherwise use the rules. |

The optional `mock` backend exercises the same interface locally and is always
labeled as a mock in both the UI and telemetry. It is useful for developing
the integration without a TypeSafe key; it is not evidence about Jev quality.

Identity, tool access, confirmation, and database mutations remain enforced by
deterministic code in every mode. Before using `route` beyond a demo, calibrate
the questions and threshold against held-out, human-reviewed conversations.

## Operational boundaries

- Run one application process. Sessions contain an in-memory transcript,
  confirmation ledger, and stateful MCP subprocess.
- API keys, the system prompt, hidden reasoning, and raw backend exceptions are
  never returned to the browser.
- The five model routes receive different amounts of traffic in the preloaded
  project. This repository is not a model leaderboard.
- The batch workload and the interactive app share runtime components, but
  starting the chat server does not launch the batch study.
- A clean clone reproduces the interactive application. It does not contain
  the private task truth or run artifacts needed to recreate the preloaded
  cohort.

## Close the loop: evaluate Insights failure clusters in Forge

Companion script [`eval_clustered_failures.py`](eval_clustered_failures.py) takes the two recurring failure clusters Agent Lens Insights surfaced in **“Where Do Agent Lens Insights Come From?”** (`Redundant confirmation of authorized actions` and `Partial processing of multi-part requests`), runs baseline vs. repaired prompt contracts on **Forge Serverless Inference** while tracing each turn with the `coreweave` Forge SDK (`coreweave.forge.agentlens.tracing`), and logs a `wandb.EvalTable` to your CoreWeave Forge Workspace:

```bash
pip install "coreweave==0.1.0b0" openai wandb
export WANDB_API_KEY=YOUR_WANDB_API_KEY

# Offline dry-run check
python eval_clustered_failures.py --dry-run

# Live Forge Serverless Inference + Agent Lens + EvalTable run
python eval_clustered_failures.py --project YOUR_ENTITY/northstar-insights-eval
```

## Validate the repository

```bash
uv run ruff check src tests
uv run pytest
```

See [`telemetry-schema.json`](telemetry-schema.json) for the emitted span
contract and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source and
license details.
