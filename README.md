# CoreWeave Forge & Agent Lens — Field Studies, Benchmarks & Examples

Canonical companion code, reproducible datasets, and linked multi-product studies for Anish Shah's **Agent Lens** and **CoreWeave Forge** engineering series.

Every example in this repository connects **Agent Lens** (`https://forge.coreweave.com/agent-lens/{entity}/{project}`) to the rest of **CoreWeave Forge**:
- **Ingest**: Shipped Python (`pip install coreweave`, `from coreweave.forge.agentlens import tracing`) and TypeScript (`@coreweave/forge-sdk@beta`) OpenTelemetry GenAI spans (`invoke_agent`, `chat`, `execute_tool`, `gen_ai.conversation.id`).
- **Observe & Triage**: Runtime errors, online LLM-judge **Tags** (`Nemotron-3.5-Lightning-30B-A3B`), **Saved Views**, and **Insights** signature clustering (`gemma-4-31B-it` + `Qwen3-Embedding-0.6B` UMAP/HDBSCAN).
- **Investigate & Verify Across Forge**: Native **ARIA** actions inside Agent Lens (`Add filters with ARIA`, `Summarize with ARIA`, `Create tag with ARIA`), the **W&B MCP Server**'s 17 read-only agent tools, **Forge Serverless Inference**, **Forge Notebooks** (Marimo), and side-by-side **Evaluation Tables (`wandb.EvalTable`)** in shared `{entity}/{project}` Forge Workspaces.

---

## Article-to-Code & Live Forge Index

### A. Consolidated Examples in This Repository (`examples/`)

| Article | Code Directory | Forge Report & Workspace Surfaces |
|---|---|---|
| **Where Do Agent Lens Insights Come From?** | [`examples/northstar-retail-support/`](examples/northstar-retail-support/) | [Forge Report](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Where-Do-Agent-Lens-Insights-Come-From--VmlldzoxNzk5OTk0NA==) · [Agent Lens Demo (`wandb/agent-lens-retail-support-live-v10`)](https://forge.coreweave.com/agent-lens/wandb/agent-lens-retail-support-live-v10) · [`eval_clustered_failures.py`](examples/northstar-retail-support/eval_clustered_failures.py) (`wandb.EvalTable` loop) |
| **We Benchmarked Jev Against Gemma Inside Agent Lens** | [`examples/jev-vs-gemma-judge-bench/`](examples/jev-vs-gemma-judge-bench/) | [Forge Report](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/We-Benchmarked-Jev-Against-Gemma-Inside-Agent-Lens--VmlldzoxNzk5OTk1MA==) · [`reproduce_metrics.py`](examples/jev-vs-gemma-judge-bench/scripts/reproduce_metrics.py) · [`replay_to_agent_lens.py`](examples/jev-vs-gemma-judge-bench/scripts/replay_to_agent_lens.py) |
| **Find Your Agent’s Capability Gaps with Agent Lens Insights** | [`examples/capability-gaps/`](examples/capability-gaps/) | [Agent Lens & Forge Workspace (`wandb/agent-lens-capability-gaps-demo`)](https://forge.coreweave.com/agent-lens/wandb/agent-lens-capability-gaps-demo) · [`emit_capability_gap_turns.py`](examples/capability-gaps/emit_capability_gap_turns.py) · [`evaluate_restart_capability.py`](examples/capability-gaps/evaluate_restart_capability.py) |

### B. Linked Flagship Multi-Product Forge Studies (`studies/` Git Submodules)

Standalone research forks and multi-product experiments remain in their dedicated repositories and are pinned here as Git submodules (`git clone --recurse-submodules`):

| Article / Series | Submodule & Standalone Repo | Live Forge Reports, Workspaces & Notebooks |
|---|---|---|
| **Recreating Harness²** & **Testing structured diagnostics in Harness²** | `studies/harness2-wandb` → [`ash0ts/harness2-wandb` (`study-v4`)](https://github.com/ash0ts/harness2-wandb/tree/study-v4) | [Part 1 Report](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4/reports/Recreating-Harness%C2%B2--VmlldzoxODE5OTAxNQ) · [Part 2 Report](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4/reports/Testing-structured-diagnostics-in-Harness%C2%B2--VmlldzoxODE4NDczNA==) · [EvalTables Workspace](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4?nw=4o9wknfo8yi) · [Agent Lens](https://forge.coreweave.com/agent-lens/wandb/harness2-agent-lens-v4) |
| **Would you give this investment committee money?** (*TradingAgents*) | `studies/tradingagents-election-lab` → [`ash0ts/TradingAgents` (`forge-showcase-v1`)](https://github.com/ash0ts/TradingAgents/tree/forge-showcase-v1) | [Forge Report](https://forge.coreweave.com/wandb/wandb/tradingagents-election-lab/reports/Would-you-give-this-investment-committee-money--VmlldzoxODI2MzIwNQ==) · [Forge Workspace](https://forge.coreweave.com/wandb/wandb/tradingagents-election-lab?nw=71cv89hjh9u) · [Notebooks (`15260`, `15271`, `15272`)](https://forge.coreweave.com/notebooks/Tm90ZWJvb2s6MTUyNzI%3D) |
| **Recovering app behavior, and measuring what REA adds** (*REA*) | `studies/rea-forge-recovery` → [`ash0ts/rea/examples/forge-recovery` (`forge-recovery-v1`)](https://github.com/ash0ts/rea/tree/forge-recovery-v1/examples/forge-recovery) | [Format Recovery Report](https://forge.coreweave.com/wandb/wandb/rea-showcase/reports/Standalone-format-recovery-%E2%80%94-ordinary-tools,-REA,-and-sample-only-control--VmlldzoxODI3ODcyMQ==) · [Animation Report](https://forge.coreweave.com/wandb/wandb/rea-showcase/reports/REA-animation-recovery-%E2%80%94-24-of-24-recorded--VmlldzoxODI2NjA3NA==) · [Dependency Report](https://forge.coreweave.com/wandb/wandb/rea-showcase/reports/Dependency-review-%E2%80%94-contained-release-pair-experiment--VmlldzoxODI3NzYwOQ==) |

### C. Hands-On Reactive Python Workshops ([`ash0ts/forge-workshop-kit`](https://github.com/ash0ts/forge-workshop-kit))

For self-serve reactive Marimo workshops combining Agent Lens, `wandb.EvalTable`, W&B Artifacts, and **ARIA** investigations:
- **Workshop Hub & Decks**: [https://ash0ts.github.io/forge-workshop-kit/](https://ash0ts.github.io/forge-workshop-kit/) ([Release `v0.12.0`](https://github.com/ash0ts/forge-workshop-kit/releases/tag/v0.12.0))
- **Pending, Inc. (Agent harnesses, tool verification & Harness²)**: [Forge Report](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit/reports/Build-an-agent-you-can-inspect%2C-then-test-changes-to-its-harness--VmlldzoxODMxMzUzMQ==) · [EvalTables](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit?nw=qeo6ytk9yxz)
- **Airline Monitoring (Saved Views, scoped AI Tags, Alignment & regression gates)**: [Forge Report](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit/reports/Turn-an-airline-agent%27s-behavior-into-monitoring-rules-and-regression-tests--VmlldzoxODMxMzU0MA==) · [EvalTables](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit?nw=4wzxelfxod1)

---

## Quickstart & Authentication

1. Create or copy your API key from [CoreWeave Forge Settings](https://forge.coreweave.com/settings#apikeys):
   ```bash
   export WANDB_API_KEY=YOUR_WANDB_API_KEY
   ```
2. Clone with linked study submodules:
   ```bash
   git clone --recurse-submodules https://github.com/ash0ts/forge-agent-lens-examples.git
   cd forge-agent-lens-examples
   ```
3. Run any self-contained offline verification without credentials:
   ```bash
   python examples/jev-vs-gemma-judge-bench/scripts/reproduce_metrics.py
   python examples/capability-gaps/emit_capability_gap_turns.py --manifest examples/capability-gaps/corpus_manifest_v2.json --dry-run --receipt /tmp/cap-v2-dry.json
   ```

## Official Documentation

- [Agent Lens Overview & Quickstart](https://docs.coreweave.com/products/agent-lens)
- [Agent Lens Insights](https://docs.coreweave.com/products/agent-lens/insights)
- [Agent Lens Framework & Harness Integrations](https://docs.coreweave.com/products/agent-lens/get-started/integrations)
- [Forge Serverless Inference Catalog](https://forge.coreweave.com/inference/models)
