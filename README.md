# CoreWeave Forge & Agent Lens — Centralized Public Code Hub, Benchmarks & Field Studies

Centralized public code, reproducible datasets, evaluation suites, and pinned Git submodules for Anish Shah's **Agent Lens** and **CoreWeave Forge** technical articles and hands-on workshops.

Every study in this repository connects **Agent Lens** (`https://forge.coreweave.com/agent-lens/{entity}/{project}`) to the rest of **CoreWeave Forge**:
- **Ingest**: Shipped Python (`pip install coreweave`, `from coreweave.forge.agentlens import tracing`) and TypeScript (`@coreweave/forge-sdk@beta`) OpenTelemetry GenAI spans (`invoke_agent`, `chat`, `execute_tool`, `gen_ai.conversation.id`).
- **Observe & Triage**: Runtime errors, online LLM-judge **Tags** (`Nemotron-3.5-Lightning-30B-A3B`), **Saved Views**, and **Insights** signature clustering (`gemma-4-31B-it` + `Qwen3-Embedding-0.6B` UMAP/HDBSCAN).
- **Investigate, Train & Verify Across Forge**: Native **ARIA** actions inside Agent Lens (`Add filters with ARIA`, `Summarize with ARIA`, `Create tag with ARIA`), the **W&B MCP Server**'s 17 read-only agent tools, **Forge Serverless Inference**, **Forge Serverless SFT**, **CoreWeave Sandboxes**, **Forge Notebooks** (Marimo), **Registry**, and side-by-side **Evaluation Tables (`wandb.EvalTable`)** in shared `{entity}/{project}` Forge Workspaces.

---

## Complete Article-to-Code & Live Forge Index

### A. Consolidated Examples & Reproducibility Bundles in This Repository (`examples/`)

| Article | Code Directory | Live Forge Report & Workspace Surfaces |
|---|---|---|
| **Where Do Agent Lens Insights Come From?** | [`examples/northstar-retail-support/`](examples/northstar-retail-support/) | [Forge Report (`VmlldzoxNzk5OTk0NA==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Where-Do-Agent-Lens-Insights-Come-From--VmlldzoxNzk5OTk0NA==) · [`eval_clustered_failures.py`](examples/northstar-retail-support/eval_clustered_failures.py) (`wandb.EvalTable` loop) |
| **We Benchmarked Jev Against Gemma Inside Agent Lens** | [`examples/jev-vs-gemma-judge-bench/`](examples/jev-vs-gemma-judge-bench/) | [Forge Report (`VmlldzoxNzk5OTk1MA==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/We-Benchmarked-Jev-Against-Gemma-Inside-Agent-Lens--VmlldzoxNzk5OTk1MA==) · [Paired EvalTables](https://forge.coreweave.com/wandb/wandb/agent-lens-insights-judge-bench) · [`reproduce_metrics.py`](examples/jev-vs-gemma-judge-bench/scripts/reproduce_metrics.py) |
| **Find Your Agent’s Capability Gaps with Agent Lens Insights** | [`examples/capability-gaps/`](examples/capability-gaps/) | [Forge Report (`VmlldzoxODMxOTg5MQ==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Find-Your-Agents-Capability-Gaps-with-Agent-Lens-Insights--VmlldzoxODMxOTg5MQ==) · [`emit_capability_gap_turns.py`](examples/capability-gaps/emit_capability_gap_turns.py) · [`evaluate_restart_capability.py`](examples/capability-gaps/evaluate_restart_capability.py) |
| **How I Built Signal Brief With Coding Agents** | [`examples/signal-brief-build-story/`](examples/signal-brief-build-story/) | [Forge Report (`VmlldzoxNzU0OTcyMA==`)](https://forge.coreweave.com/wandb/wandb/news-research-agent/reports/Building-Signal-Brief-With-Coding-Agents:-A-Trace-Backed-Engineering-Field-Report--VmlldzoxNzU0OTcyMA==) · [`editorial-snapshot.json`](examples/signal-brief-build-story/editorial-snapshot.json) |

### B. Centralized Public Multi-Product Forge Repositories (`studies/` Git Submodules)

All standalone engineering forks, applications, and workshops are public on GitHub and pinned here under `studies/` (`git clone --recurse-submodules`):

| Article / Series | Submodule Path → Public GitHub Repo | Live Forge Reports, Workspaces & Interactive Viewers |
|---|---|---|
| **Recreating Harness²** & **Testing structured diagnostics in Harness²** | `studies/harness2-wandb` → [`ash0ts/harness2-wandb` (`study-v4`)](https://github.com/ash0ts/harness2-wandb/tree/study-v4) | [Part 1 Report (`VmlldzoxODE5OTAxNQ`)](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4/reports/Recreating-Harness%C2%B2--VmlldzoxODE5OTAxNQ) · [Part 2 Report (`VmlldzoxODE4NDczNA==`)](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4/reports/Testing-structured-diagnostics-in-Harness%C2%B2--VmlldzoxODE4NDczNA==) · [EvalTables](https://forge.coreweave.com/wandb/wandb/harness2-agent-lens-v4?nw=4o9wknfo8yi) |
| **Before we let an agent optimize the harness** (*Meta-Harness Controls*) | `studies/meta-harness-wandb` → [`ash0ts/meta-harness-wandb` (`forge-controls-v1`)](https://github.com/ash0ts/meta-harness-wandb/tree/forge-controls-v1) | [Forge Report (`VmlldzoxODMyMDIwMQ==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Before-we-let-an-agent-optimize-the-harness--VmlldzoxODMyMDIwMQ==) · [4 Weave EvalTables (`1,648` cells)](https://forge.coreweave.com/wandb/wandb/mh-forge-preflight-20261110) |
| **Would you give this investment committee money?** (*TradingAgents*) | `studies/tradingagents-election-lab` → [`ash0ts/TradingAgents` (`forge-showcase-v1`)](https://github.com/ash0ts/TradingAgents/tree/forge-showcase-v1) | [Forge Report (`VmlldzoxODI2MzIwNQ==`)](https://forge.coreweave.com/wandb/wandb/tradingagents-election-lab/reports/Would-you-give-this-investment-committee-money--VmlldzoxODI2MzIwNQ==) · [Workspace](https://forge.coreweave.com/wandb/wandb/tradingagents-election-lab?nw=71cv89hjh9u) · [Notebooks](https://forge.coreweave.com/notebooks/Tm90ZWJvb2s6MTUyNzI%3D) |
| **Recovering app behavior, and measuring what REA adds** (*REA*) | `studies/rea-forge-recovery` → [`ash0ts/rea` (`forge-recovery-v1`)](https://github.com/ash0ts/rea/tree/forge-recovery-v1/examples/forge-recovery) | [Forge Report (`VmlldzoxODMyMDIwMA==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Recovering-app-behavior-and-measuring-what-REA-adds--VmlldzoxODMyMDIwMA==) · [Interactive Sprite Viewer](https://ash0ts.github.io/forge-agent-lens-examples/rea-animation-demo/) |
| **Forge Workshop Kit** (*Pending, Inc.* & *Airline Monitoring*) | `studies/forge-workshop-kit` → [`ash0ts/forge-workshop-kit` (`v0.12.0`)](https://github.com/ash0ts/forge-workshop-kit/tree/v0.12.0) | [Pending Report (`VmlldzoxODMxMzUzMQ==`)](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit/reports/Build-an-agent-you-can-inspect%2C-then-test-changes-to-its-harness--VmlldzoxODMxMzUzMQ==) · [Airline Report (`VmlldzoxODMxMzU0MA==`)](https://forge.coreweave.com/wandb/wandb/forge-workshop-kit/reports/Turn-an-airline-agent%27s-behavior-into-monitoring-rules-and-regression-tests--VmlldzoxODMxMzU0MA==) · [Workshop Hub](https://ash0ts.github.io/forge-workshop-kit/) |
| **Building FlowBlock: from a camera move to a measured correction** | `studies/flowblock` → [`ash0ts/flowblock` (`forge-shot-planning-v1`)](https://github.com/ash0ts/flowblock/tree/forge-shot-planning-v1) | [Forge Report (`VmlldzoxODMyMDIwMg==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Building-FlowBlock-from-a-camera-move-to-a-measured-correction--VmlldzoxODMyMDIwMg==) |
| **Building TabToPitch** & **Learn the fretboard through a song** | `studies/tab-to-pitch` → [`ash0ts/tab-to-pitch` (`forge-voicing-teacher-v1`)](https://github.com/ash0ts/tab-to-pitch/tree/forge-voicing-teacher-v1) | [Builder Report (`VmlldzoxODMyMDIwNA==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Building-TabToPitch-from-guitar-voicings-to-a-teaching-agent--VmlldzoxODMyMDIwNA==) · [Practice Report (`VmlldzoxODMyMDIwNQ==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/TabToPitch-learn-the-fretboard-through-a-song-you-can-already-play--VmlldzoxODMyMDIwNQ==) |
| **Evals Are How We Make our Slop Safe and Tasty** & **Why I Built Fugue** | `studies/fugue` → [`ash0ts/fugue` (`main`)](https://github.com/ash0ts/fugue) | [Safe & Tasty Report (`VmlldzoxODMyMDIwNg==`)](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/Evals-Are-How-We-Make-our-Slop-Safe-and-Tasty--VmlldzoxODMyMDIwNg==) · [BRC Live Post](https://www.coreweave.com/blog/why-i-built-fugue-controlled-experiments-for-agent-systems) |
| **Why We Added an AI Guide to a Stormwater Tool** (*GI Scenario Studio*) | `studies/gi-scenario-studio` → [`ash0ts/gi-scenario-studio` (`stormwater-evidence-guide-v1`)](https://github.com/ash0ts/gi-scenario-studio/tree/stormwater-evidence-guide-v1) | [Evidence Guide v2 Study](https://github.com/ash0ts/gi-scenario-studio/blob/stormwater-evidence-guide-v1/docs/evidence-guide-v2-study.md) |
| **Why I Built EcoCorridor** (*EcoCorridor Scenario Studio*) | `studies/ecocorridor-scenario-studio` → [`ash0ts/gi-scenario-studio` (`ecocorridor-review-brief-v1`)](https://github.com/ash0ts/gi-scenario-studio/tree/ecocorridor-review-brief-v1) | [Forge Report (`VmlldzoxNzY1OTU0MQ==`)](https://forge.coreweave.com/wandb/wandb/ecocorridor-scenario-studio/reports/EcoCorridor-historical-search-benchmark--VmlldzoxNzY1OTU0MQ==) |

---

## Quickstart & Authentication

1. Create or copy your API key from [CoreWeave Forge Settings](https://forge.coreweave.com/settings#apikeys):
   ```bash
   export WANDB_API_KEY=YOUR_WANDB_API_KEY
   ```
2. Clone with all linked public study submodules:
   ```bash
   git clone --recurse-submodules https://github.com/ash0ts/forge-agent-lens-examples.git
   cd forge-agent-lens-examples
   ```
3. Run offline metric reproductions without credentials:
   ```bash
   python examples/jev-vs-gemma-judge-bench/scripts/reproduce_metrics.py
   python examples/capability-gaps/emit_capability_gap_turns.py --manifest examples/capability-gaps/corpus_manifest_v2.json --dry-run --receipt /tmp/cap-v2-dry.json
   ```

## Official Documentation

- [Agent Lens Overview & Quickstart](https://docs.coreweave.com/products/agent-lens)
- [Agent Lens Insights](https://docs.coreweave.com/products/agent-lens/insights)
- [Agent Lens Framework & Harness Integrations](https://docs.coreweave.com/products/agent-lens/get-started/integrations)
- [Forge Serverless Inference Catalog](https://forge.coreweave.com/inference/models)
