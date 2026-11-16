# Agent Lens Capability-Boundary Corpus & Forge Qualification Eval

Companion code for **“Find Your Agent’s Capability Gaps with Agent Lens Insights.”**

All spans are emitted with the shipped CoreWeave Forge SDK (`coreweave==0.1.0b0`, `from coreweave.forge.agentlens import tracing`), and candidate verification runs against **Forge Serverless Inference** (`https://api.inference.wandb.ai/v1`) and logs an Evaluation Table (`wandb.EvalTable`) into the shared CoreWeave Forge Workspace (`wandb/agent-lens-capability-gaps-demo`).

## What Is Included

1. **`emit_capability_gap_turns.py`**:
   - `corpus_manifest.json` (`v1`, `180` turns): Original frozen controlled batch demonstrating why `27` templated `Capability limitation` turns with fewer than `15` distinct wordings land in the top-level category without forming HDBSCAN sub-topic clusters.
   - `corpus_manifest_v2.json` (`v2`, `240` turns): Expanded controlled batch with `20+` distinct operational capability-limitation templates exceeding Agent Lens Insights clustering floors (`>=100` signatures in the 7-day window and `>=15` distinct wordings per category).
2. **`evaluate_restart_capability.py`**:
   - Evaluates `read_only_baseline` vs. `scoped_approval_candidate` across `10` curated training-operations scenarios spanning all four human-review buckets (scoped unhealthy-worker restart requests, discoverability misses, intentional safety boundaries, and diagnostic-only controls), logging both Agent Lens traces and a `wandb.EvalTable` in CoreWeave Forge.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Run offline unit tests and dry-run checks (no network or API key required)
pytest -q
python emit_capability_gap_turns.py --manifest corpus_manifest_v2.json --dry-run --receipt artifacts/capability-gaps-v2.dry-run.receipt.json
python evaluate_restart_capability.py --dry-run

# 2. Emit corpus & run the bounded Forge Serverless Inference + EvalTable qualification
export FORGE_TRACE_PROJECT=wandb/agent-lens-capability-gaps-demo
export WANDB_API_KEY=YOUR_WANDB_API_KEY

python emit_capability_gap_turns.py --manifest corpus_manifest_v2.json --receipt artifacts/capability-gaps-v2.export.receipt.json
python evaluate_restart_capability.py --project wandb/agent-lens-capability-gaps-demo
```
