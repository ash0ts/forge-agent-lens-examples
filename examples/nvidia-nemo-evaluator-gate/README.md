# A release gate you have seen say no

Companion code and evidence for the CoreWeave Forge Report [**A release gate you have seen say no**](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/reports/A-release-gate-you-have-seen-say-no--VmlldzoxODMzMTAxMw==).

Scores a candidate support-routing prompt against the current one-line baseline using NVIDIA's NeMo Evaluator (`exact_match`) on Forge Serverless Inference, enforces a nine-check policy in `data/policy.json`, requires a deliberately broken negative-control prompt to be rejected, and logs the full decision record to one Forge run.

Built on public NeMo Evaluator and Forge APIs. Not reviewed by NVIDIA.

## What this example covers

- Turning "the notebook looked fine" into nine named checks that a CI step (`evaluator-gate verify`) reads directly from a Forge run summary.
- Running a negative control (`always answer "billing"`) inside every gate run so a broken scorer or loosened floor fails closed.
- Logging latency and output tokens next to accuracy. When we extracted this package and forgot `chat_template_kwargs={"enable_thinking": False}`, the one-line baseline jumped from `42 / 48` to `48 / 48` by spending `18,830` output tokens (`~392` reasoning tokens and `3.0 s` per ticket) instead of `132` tokens (`179 ms` per ticket). Restoring `enable_thinking=False` reproduced `42 / 48` vs. `48 / 48` down to the same six ticket IDs.

## Forge surfaces, in reading order

Open the public [**saved Workspace view (`?nw=kzxnkpkxl5n`)**](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples?nw=kzxnkpkxl5n) (no login required):

| Section | Surface | Live runs |
| --- | --- | --- |
| 1 | `gate/ci_exit_code`, `gate/heldout_delta`, `gate/paired_regressions`, `control/correct` | [live no-thinking `e0ce3715`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/e0ce3715) · [live thinking-on `imy7ehrq`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/imy7ehrq) · [Oct 29 replay `gv5j7kfg`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/gv5j7kfg) |
| 2 | `release_gate/heldout` `wandb.EvalTable`: baseline, candidate, and control outputs side by side | same three runs (public, loads signed out) |
| 3 | `release_gate/policy_checks` (all 9 checks) and `release_gate/paired_changes` (the 6 fixed tickets) | same runs |
| 4 | `baseline/heldout/latency_ms_p50` and `completion_tokens_total`: thinking-off vs. thinking-on cost | same runs |
| 5 | `release_gate/category_accuracy` against the `0.75` per-queue floor + `release-gate-inputs` / `release-gate-evidence` artifacts | same runs |

## Results on `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`

| Run | Thinking | Baseline held-out | Candidate held-out | Control | Baseline median ms | Held-out output tokens (base / cand) | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-29 capture, rescored (`gv5j7kfg`) | off | 42 / 48 (`billing` 12, `obs` 11, `train` 10, `inf` 9) | 48 / 48 | 2 / 8, rejected | 99.5 ms | 132 / 132 | pass (`+0.125`) |
| 2026-11-18 live (`e0ce3715`) | off | 42 / 48 (same 6 tickets missed) | 48 / 48 | 2 / 8, rejected | 178.8 ms | 132 / 132 | pass (`+0.125`) |
| 2026-11-18 live (`imy7ehrq`) | on | 48 / 48 | 48 / 48 | 2 / 8, rejected | 3,038.5 ms | 18,830 / 26,272 | pass (`0.000`) |

## Run it

```bash
cd examples/nvidia-nemo-evaluator-gate
uv sync
uv run pytest -q

export WANDB_API_KEY=...   # or ~/.netrc

# Live gate (136 calls: 64 tickets x 2 prompts + 8 control, thinking off by default)
uv run evaluator-gate run --project wandb/nvidia-forge-examples --output out/run-01

# Optional: compare with Nemotron thinking enabled
uv run evaluator-gate run --project wandb/nvidia-forge-examples --enable-thinking --output out/run-thinking

# Offline replay of saved responses (no API key or model calls required)
uv run evaluator-gate replay --responses evidence/live-2026-11-18/responses.jsonl --output out/replay

# CI check: read the stored decision from a Forge run summary
uv run evaluator-gate verify --run wandb/nvidia-forge-examples/e0ce3715
```

## Layout

- `data/tickets.json`, `data/prompts.json`, `data/policy.json`: frozen 64-ticket suite, prompts, and nine-check policy.
- `src/evaluator_gate/gate.py`: threaded collection with JSONL journaling, `nemo_evaluator.exact_match` scoring, Wilson 95% intervals, token/latency summaries, and `decide()`.
- `src/evaluator_gate/forge.py`: logs the `wandb.EvalTable`, policy/category/diff tables, input and evidence artifacts, and summary keys; implements `verify()`.
- `scripts/build_workspace_views.py` and `scripts/draw_article_figures.py`: reproduce the saved Forge Workspace views and article figures.
- `tests/test_gate.py`: offline unit tests covering replay reconciliation, paired regression holds, label-contract failures, and broken-control holds.

## Limits

Synthetic hand-authored tickets test this four-queue routing task, not general model quality. Held-out means excluded from prompt authoring, not secret from repository readers. Each pair runs once and failures stay in the denominator. The gate writes an exit code and deploys nothing.
