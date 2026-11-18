# A release gate you have seen say no

A prompt change wants to ship. This example scores it with NVIDIA's NeMo Evaluator `exact_match` scorer on Forge Serverless Inference, applies a written policy, and publishes everything the policy looked at to one Forge run. The gate has to reject a deliberately bad prompt before anyone trusts its yes.

Not reviewed by NVIDIA. Built on public NeMo Evaluator and Forge APIs.

## What you learn

- How to turn "the eval looked fine" into a policy with named checks that a CI job can read back from a Forge run summary.
- Why a negative control belongs in every gate run. The control prompt answers `billing` for every ticket and scores 2/8; if the gate ever passes it, the gate is broken.
- What a model drift looks like in evidence. The same baseline prompt scored 42/48 held-out on 2026-10-29 and 48/48 on 2026-11-18 on the same served model name. Both runs are published; neither is hidden.

## Forge surfaces, in reading order

| Step | Surface | Live |
| --- | --- | --- |
| 1 | `release_gate/heldout` EvalTable: baseline, candidate, and control answers side by side with correctness scores | [live run `imy7ehrq`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/imy7ehrq) · [October replay `gv5j7kfg`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples/runs/gv5j7kfg) |
| 2 | `release_gate/policy_checks` table and `gate/check/*` summary keys: the nine checks CI reads | same runs |
| 3 | `release_gate/paired_changes`: which cases the candidate fixed or broke relative to baseline | same runs |
| 4 | `release_gate/category_accuracy` and `release_gate/category_bar/*`: per-category accuracy against the 0.75 floor | same runs |
| 5 | Artifacts `release-gate-inputs` (tickets, prompts, policy with SHA-256) and `release-gate-evidence` (journaled responses, report) | same runs |

Project: [`wandb/nvidia-forge-examples`](https://forge.coreweave.com/wandb/wandb/nvidia-forge-examples).

## Results

| Run | Baseline held-out | Candidate held-out | Control | Decision |
| --- | --- | --- | --- | --- |
| 2026-10-29 responses, replayed | 42/48 (inference 9/12, training 10/12, observability 11/12, billing 12/12) | 48/48 | 2/8, rejected | pass, delta +0.125 |
| 2026-11-18 live | 48/48 | 48/48 | 2/8, rejected | pass, delta 0.0 |

The live run passes on the "no regressions, meets floors" checks, not on improvement. `min_heldout_delta` is 0.0 in `data/policy.json`; set it above zero if your gate should demand a measurable gain.

## Run it

```bash
cd examples/nvidia-nemo-evaluator-gate
uv sync
export WANDB_API_KEY=...   # or ~/.netrc

# 136 model calls (64 cases x 2 prompts + 8 control), then publish to Forge
uv run evaluator-gate run --project wandb/nvidia-forge-examples --output out/run-01

# Re-score journaled responses with no model calls (offline), optionally publish
uv run evaluator-gate replay --responses evidence/live-2026-11-18/responses.jsonl --output out/replay

# CI: read the decision from the run summary
uv run evaluator-gate verify --run wandb/nvidia-forge-examples/imy7ehrq
```

`replay` works offline from `evidence/*/responses.jsonl`. `run` and `verify` need a Forge API key.

## How it is built

- `data/tickets.json`: 64 hand-authored support tickets, 4 labels, 16 development + 48 held-out.
- `data/prompts.json`: `baseline`, `candidate`, and the deliberately broken `control`.
- `data/policy.json`: the gate. Floors, zero paired regressions, complete responses, strict label contract.
- `src/evaluator_gate/gate.py`: collect (threaded, journaled), score with `nemo_evaluator` `exact_match`, summarize with Wilson 95% intervals, `decide()`.
- `src/evaluator_gate/forge.py`: one `wandb.EvalTable`, four tables, bar plots, artifacts, summary keys, and `verify()`.
- `nemo-evaluator` is pinned to a git revision because the `ScorerInput` API used here is newer than the PyPI release.

## Limits

Synthetic tickets test this routing task, not general model quality. Held-out means excluded from prompt development, not secret from repository readers. Each request is attempted once; failures stay in the denominator. The gate deploys nothing.
