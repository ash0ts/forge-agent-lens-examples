# Recorded Codex CLI Session Prompts (`v1` & `v2`)

Codex runs from an isolated temporary directory (`/private/tmp/looped-transformer-chat-v2`) with an isolated `CODEX_HOME` (`/private/tmp/codex-home-looped-v2`) so global `~/.codex` state and unrelated repos never enter the Agent Lens trace. Every W&B MCP call explicitly passes `entity="wandb"` and `project="looped-transformer-study"`.

---

## Study v2 — Teach First, One Experiment Per Argument (`5 seeds` per arm + frozen `holdout_seed=31415`)

Pinned benchmark artifact: `wandb/looped-transformer-study/looped-transformer-lab:v1`.
Polling rule: run a local `sleep` between `aria_send_message` and `aria_get_turn(wait_seconds=30)` so multi-minute ARIA PyTorch jobs finish in 1–3 polls rather than 29 rapid polls.

### Turn 1 — Parallel ARIA fan-out across the four arguments (`capacity-v2`, `loop-vs-unrolled-v2`, `injection-sweep-v2`)

```text
We are expanding our looped-transformer study into a 5-seed v2 matrix organized around the four live arguments people have about looped transformers: (1) capacity & how a loop learns across steps, (2) whether a loop is just an unrolled deeper untied network and what it costs vs KV-cached scratchpad CoT, and (3a/3b) carry vs re-read input injection strength, pass-1 scaling (`2*x0` vs skipping pass 1), and entry-state noise perturbation.
Skim `./looped_transformer_lab.py` locally, then call the `wandb` MCP tool `aria_send_message` FOUR TIMES IN PARALLEL (`entity="wandb"`, `project="looped-transformer-study"` — never call `list_entities_tool` or touch other projects). Tell every ARIA session NOT to browse external URLs, to download the frozen v2 harness once into `/workspace` via `python3 -c 'import wandb; wandb.Api().artifact("wandb/looped-transformer-study/looped-transformer-lab:v1").download("/workspace")'`, to keep every bash call under the 240s sandbox timeout, and to return the printed JSON summaries:
1. Title `Looped v2 — Q1: Capacity & Log-Depth Ceiling`: run `--arm arm_a_standard --train-max-k 8 --steps 1600 --eval-every 100 --group capacity-v2 --question capacity --arm-tag cap_std4_k8 --label "Standard 4-Layer (trained k=1..8)"` first with `--seeds 0,1,2 --run-weave-eval` then `--seeds 3,4`; next run `--arm arm_b_looped_fixed --train-max-k 8 --steps 1600 --eval-every 100 --group capacity-v2 --question capacity --arm-tag cap_loop4_k8 --label "Looped Fixed T=4 (trained k=1..8)"` first with `--seeds 0,1,2 --run-weave-eval` then `--seeds 3,4`; finally run `--arm arm_b_looped_fixed --k-encoding scalar --train-max-k 4 --steps 800 --eval-every 50 --seeds 0,1,2,3,4 --group capacity-v2 --question capacity --arm-tag cap_loop4_scalar_k4 --label "Looped Fixed T=4 (scalar k, trained k=1..4)" --run-weave-eval`.
2. Title `Looped v2 — Q2: Loop vs Unrolled Twin & KV Cost`: run `--arm arm_e_unrolled_extrap --train-max-k 4 --steps 800 --eval-every 50 --seeds 0,1,2,3,4 --group loop-vs-unrolled-v2 --question loop_vs_unrolled --arm-tag unrolled8_k4 --label "Unrolled 8-Layer Untied (trained k=1..4)" --run-weave-eval`; next run `--arm arm_e_unrolled_extrap --train-max-k 8 --steps 1600 --eval-every 100 --group loop-vs-unrolled-v2 --question loop_vs_unrolled --arm-tag unrolled8_k8 --label "Unrolled 8-Layer Untied (trained k=1..8)"` first with `--seeds 0,1,2 --run-weave-eval` then `--seeds 3,4`; finally run `--arm arm_d_scratchpad --train-max-k 4 --steps 800 --eval-every 50 --seeds 0,1,2,3,4 --group loop-vs-unrolled-v2 --question compute_cost --arm-tag scratchpad_kv --label "Scratchpad CoT 2-Layer (KV-cached)" --save-checkpoint --run-weave-eval`.
3. Title `Looped v2 — Q3a: Carry vs Full Re-Read Injection`: run 5 seeds (`--seeds 0,1,2,3,4 --steps 800 --eval-every 50 --group injection-sweep-v2 --question injection_sweep --save-checkpoint --run-weave-eval`) for three arms: `--arm arm_c_looped_extrap --arm-tag inj_0p0_carry --label "Carry Loop (0.0x injection)"`, `--arm arm_c_looped_extrap --input-inject --inject-scale 1.0 --inject-from-pass 1 --arm-tag inj_1p0_from1 --label "Re-read 1.0x from Pass 1 (2*x0)"`, and `--arm arm_c_looped_extrap --input-inject --inject-scale 1.0 --inject-from-pass 2 --arm-tag inj_1p0_from2 --label "Re-read 1.0x from Pass 2 (skip pass 1)"`.
4. Title `Looped v2 — Q3b: Injection Scale Sweep & Noise Robustness`: run 5 seeds (`--seeds 0,1,2,3,4 --steps 800 --eval-every 50 --group injection-sweep-v2 --question injection_sweep --save-checkpoint`) for three arms: `--arm arm_c_looped_extrap --input-inject --inject-scale 0.25 --inject-from-pass 1 --arm-tag inj_0p25_from1 --label "Re-read 0.25x from Pass 1"`, `--arm arm_c_looped_extrap --input-inject --inject-scale 0.5 --inject-from-pass 1 --arm-tag inj_0p5_from1 --label "Re-read 0.5x from Pass 1" --run-weave-eval`, and `--arm arm_c_looped_extrap --input-inject --inject-scale 0.5 --inject-from-pass 2 --arm-tag inj_0p5_from2 --label "Re-read 0.5x from Pass 2 (skip pass 1)" --run-weave-eval`.
IMPORTANT POLLING RULE: To avoid rapid MCP polling bloat, run a local shell `sleep 150` right after firing all four `aria_send_message` calls before your first `aria_get_turn(wait_seconds=30)` check, and `sleep 60` between any subsequent checks until all four `turn_id`s reach `is_terminal=true`. Summarize the 5-seed findings across all four threads.
```

### Turn 2 — ARIA-proposed follow-up arms (`aria-followups-v2`)

```text
Continue the `Looped v2 — Q3b: Injection Scale Sweep & Noise Robustness` ARIA thread via `parent_turn_id` (no external web fetches, shell timeout 240s per command). Ask ARIA to query all six `injection-sweep-v2` arms in `wandb/looped-transformer-study` via `wandb.Api()` — inspecting 5-seed mean/min/max `eval/extrap_acc`, `monitor/logit_lens_hop_acc`, and entry-state perturbation `robustness/acc_noise_a2` & `robustness/acc_noise_a4` — and then propose and execute TWO follow-up looped-model experiments of its own under the <=65k parameter, 1-emitted-token budget (`--seeds 0,1,2,3,4 --steps 800 --eval-every 50 --group aria-followups-v2 --question aria_followup --save-checkpoint --run-weave-eval`, e.g. testing `--step-embed`, gated skip-pass-1, or a small architectural tweak in `/workspace/looped_transformer_lab.py` with distinct `--arm-tag` names `aria_f1_*` and `aria_f2_*`). Ask ARIA to report honestly whether either follow-up beat the sweep leader on clean extrapolation or noise robustness.
Remember the polling rule: run a local shell `sleep 120` after `aria_send_message` before calling `aria_get_turn(wait_seconds=30)`, and `sleep 45` between any further checks.
```

### Turn 3 — Freeze model selection & evaluate once on fresh `holdout_seed=31415` (`final-holdout-v2`)

```text
Freeze model selection now before touching `--holdout-seed 31415`. Continue that same ARIA thread via `parent_turn_id` and have ARIA run `/workspace/looped_transformer_lab.py --eval-only --holdout-seed 31415 --seeds 0,1,2,3,4 --group final-holdout-v2 --study-version v2` once each on the saved W&B checkpoint artifacts (`ckpt-<arm>-s<seed>:latest`) for: (1) `inj_0p0_carry`, (2) `inj_1p0_from1`, (3) `inj_1p0_from2`, (4) `inj_0p5_from1`, (5) `inj_0p5_from2`, (6) `scratchpad_kv`, and (7) the stronger of ARIA's two `aria-followups-v2` arms.
Run a local shell `sleep 45` before polling `aria_get_turn(wait_seconds=30)` (and `sleep 30` if needed) until terminal, then report the frozen 5-seed holdout table.
```

### Turn 4 — Build the Organized v2 Saved Workspace & Verify Agent Lens Trace

```text
Launch one final synthesis ARIA thread in `wandb/looped-transformer-study` titled `Looped v2 — Study Workspace & Verification` (no external web browsing) asking ARIA to run a short Python script with `wandb` and `wandb_workspaces.workspaces` that:
(1) verifies run counts across `capacity-v2`, `loop-vs-unrolled-v2`, `injection-sweep-v2`, `aria-followups-v2`, and `final-holdout-v2`;
(2) saves a clean Workspace view on `wandb/looped-transformer-study` named `Looped Transformers v2 — One Experiment Per Argument` grouped by `config.arm` with sections for Watching a Loop Learn (`train_eval/in_dist_acc`, `train_eval/extrap_acc`, `train_eval/logit_lens_hop_acc`, `train/loss`), Per-Hop Extrapolation & Capacity (`curve_hop/accuracy`), Entry-State Noise Robustness (`curve_noise/extrap_acc`), and Internal Pass Readout (`curve_pass/logit_lens_true_hop_acc`, `curve_pass/residual_norm`); and
(3) prints the saved Workspace `?nw=...` URL alongside our live Agent Lens conversation URL `https://forge.coreweave.com/agent-lens/wandb/looped-transformer-study/conversations/__SESSION_ID__`.
Run a local shell `sleep 45` before polling `aria_get_turn(wait_seconds=30)` until terminal, then print the Workspace URL and Agent Lens URL.
```

---

## Study v1 — Original 21-Run Discovery Pass (`four-arms-v1` + `autoresearch-v1`, 3 seeds)

Pinned benchmark artifact: `wandb/looped-transformer-study/looped-transformer-lab:v0` (preserved in W&B with `tag="v1"` and `config.study_version="v1"`).
See git history (`41157cf`) or `session_timeline.json` (`01a20d47-0793-7ca3-a745-31ad25702522`) for the exact 4-turn v1 prompts.
