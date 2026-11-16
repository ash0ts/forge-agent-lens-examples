# Jev vs. Gemma Inside Agent Lens — Public Reproduction Package

Companion benchmark data and standalone reproduction tools for **“We Benchmarked Jev Against Gemma Inside Agent Lens”** ([Forge Report](https://forge.coreweave.com/wandb/wandb/agent-lens-fc/reports/We-Benchmarked-Jev-Against-Gemma-Inside-Agent-Lens--VmlldzoxNzk5OTk1MA==)).

All conversation-derived free text has been removed from this public package; only structured labels, option probabilities, token counts, latencies, and adjudicated disagreement winners remain.

## Quickstart: Reproduce Every Number Offline

Requires Python 3.10+ (zero external packages needed for offline verification):

```bash
python scripts/reproduce_metrics.py
```

This recomputes directly from `data/turns.csv`, `data/audit_calls.csv`, and `data/jev_raw.jsonl`:
- **Failure detection (`n = 100`, `29` reviewed failures)**:
  - `google/gemma-4-31B-it` (served on **Forge Serverless Inference**): `18` TP, `1` FP, `11` FN (`94.7%` precision, `62.1%` recall)
  - `jev-1.13.0` at `0.5` threshold: `29` TP, `2` FP, `0` FN (`93.5%` precision, `100.0%` recall); at `0.8` threshold: `19` TP, `0` FP, `10` FN (`100.0%` precision, `65.5%` recall)
- **Adjudicated taxonomy field accuracy**: Turn status (`99/100` Gemma vs. `97/100` Jev), Intent category (`98/100` vs. `89/100`), Sentiment + target (`99/100` vs. `93/100`), Language (`100/100` vs. `98/100`), Failure category when both fired (`18/18` vs. `15/18`).
- **Latency & cost (`200` calls per judge)**: Jev `0.143s` p50 (`0.214s` p95, `0.595s` max) and `$0.0556` vs. Gemma's `~8s` concurrent single-call estimate (`200` calls in `87s` at concurrency `50`, while also generating signature prose) and `$0.1424`.

## Optional: Log Comparison `EvalTable` & Scrubbed Spans to CoreWeave Forge

```bash
pip install "coreweave==0.1.0b0" wandb
export WANDB_API_KEY=YOUR_WANDB_API_KEY

# 1. Log a 100-row wandb.EvalTable into your Forge Workspace
python scripts/reproduce_metrics.py --log-forge-project YOUR_ENTITY/jev-vs-gemma-bench

# 2. Replay the 100 scrubbed judge-comparison turns into Agent Lens via the Forge SDK
python scripts/replay_to_agent_lens.py --project YOUR_ENTITY/jev-vs-gemma-bench --emit
```

Open `https://forge.coreweave.com/agent-lens/YOUR_ENTITY/jev-vs-gemma-bench` and `https://forge.coreweave.com/wandb/YOUR_ENTITY/jev-vs-gemma-bench` to inspect the spans and `EvalTable` side by side.

## Included Files

- `data/turns.csv`: 100 turns with parsed Gemma & Jev labels, probabilities, latencies, token counts, and audited failure truth.
- `data/jev_probabilities.csv`: 5,700 long-format option probabilities across all Jev questions.
- `data/audit_calls.csv`: Scrubbed per-field disagreement winners (`turn`, `field`, `gemma`, `jev`, `winner`).
- `data/gemma_labels.jsonl` & `data/jev_raw.jsonl`: Structured per-call outputs (free text removed).
- `scripts/jev_port.py`: Exact typed question definitions used to port the Insights intent and failure contracts to Jev.
- `scripts/reproduce_metrics.py`: Standalone offline metric verifier + optional `wandb.EvalTable` logger.
- `scripts/replay_to_agent_lens.py`: Standalone `coreweave.forge.agentlens.tracing` replay emitter.
- `REPORT.md` & `GLOSSARY.md`: Summary findings and taxonomy definitions.
