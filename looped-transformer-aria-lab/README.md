# Looped Transformer ARIA Lab

When someone asks what a looped transformer actually changes, the cleanest explanation is three lines of code. A standard transformer stacks distinct blocks with separate weight matrices and runs each block once per emitted token; a looped transformer wraps `for _ in range(T):` around shared weights so hidden states refine across multiple passes before emitting a token.

This directory holds the reproducible CPU PyTorch benchmark, proxy configuration, arm definitions, and Codex prompts behind our Forge study (`wandb/looped-transformer-study`), where **one Codex CLI session** (running `moonshotai/Kimi-K2.7-Code` on **CoreWeave Forge Serverless Inference** via [`purokishi`](https://github.com/kmio11/purokishi)) fans out **parallel ARIA sessions** over the W&B MCP Server (`https://mcp.withwandb.com/mcp`), lets ARIA propose and execute bounded follow-ups, evaluates frozen checkpoints once on a locked holdout seed (`holdout_seed=31415`), saves a shared Forge Workspace, and traces the orchestrating Codex session itself into **CoreWeave Agent Lens** via `@coreweave/forge-agent-lens-for-codex`.

## What the benchmark tests (toy scale, CPU PyTorch)

Every arm trains from scratch (`d_model=64`, `n_heads=4`, `d_ff=256`, AdamW `lr=3e-3`) on synthetic **pointer chasing** over shuffled 12-item permutation tables:

- **Validation & secondary graph splits during exploration:** `seed=2026` (`2,048` fresh graphs across $k \in \{1..4\}$ in-distribution and $k \in \{5, 6, 7, 8\}$ longer lookup chains), alongside a secondary split (`seed=9090` — inspected across rounds in Round 1 and during training in Round 2, **not** an untouched holdout) and 48-example Weave Evaluations (`seed=7777`).
- **Locked single-shot final holdout (`final-holdout-v2`):** after model architectures and hyperparameters are frozen, saved W&B checkpoint artifacts are scored **once** via `--eval-only --holdout-seed 31415` (`4,096` unseen graphs across $k=1..8$ per checkpoint, zero parameter updates).

### Round 1 (`four-arms-v1` + `autoresearch-v1`, 21 runs, 3 seeds × 800 steps)

Our initial 21-run exploratory pass trained four arms (`arm_a_standard`, `arm_b_looped_fixed`, `arm_c_looped_extrap`, `arm_d_scratchpad`) plus three autoresearch variants (`auto_r1_input_inject`, `auto_r2_gated`, `auto_r3_half_inject`) for `800` steps across seeds `0, 1, 2` (preserved in `wandb/looped-transformer-study` with tag `v1`).

Auditing Round 1 exposed three baseline issues that motivated Round 2:
1. **Undertraining & out-of-vocabulary hop tokens:** `arm_a_standard` reached only `53.29%` in-distribution at `800` steps, and discrete hop tokens (`H5..H8`) had random untrained embeddings whenever a fixed-depth run trained only on $k \in \{1..4\}$.
2. **Uncached Scratchpad CoT timing:** greedy decode in Round 1 recomputed the full prefix on every step without a KV cache, overstating the Scratchpad latency gap (`2.9x` in Weave).
3. **Pass-1 double-scaling vs. later-pass anchoring:** adding `1.0 * x0` starting on pass 1 made pass 1 see `2 * x0` (`x = x0` initially), causing seed 0 to regress (`76.56% -> 57.33%`), whereas [Yang et al. (2023)](https://arxiv.org/abs/2311.12424) initialize the recurrent state at zero so pass 1 sees `x0` once.

### Round 2 (`*-v2` groups, 105 runs: 70 training across 14 arms × 5 seeds + 35 single-shot holdout evaluations)

Round 2 runs 5 seeds (`0, 1, 2, 3, 4`) per arm (`1,600` steps when training on $k=1..8$, `800` steps when training on $k=1..4$), logs in-training accuracy curves every `50`–`100` steps (loss every `10` steps), caches K/V tensors during Scratchpad CoT decode, measures pass-1 Gaussian state perturbations ($\alpha \in \{0, 0.5, 1, 2, 4\}$), and isolates one live engineering argument per experiment group:

1. **Watching a loop learn & log-depth compression (`capacity-v2`):**
   - `cap_loop4_k8` (`54,732` params, 1 shared block run fixed $T=4$ times, trained on $k=1..8$ for `1,600` steps): reaches **`87.36%`** `[39.84%, 100.00%]` on $k=1..4$ and **`75.68%`** `[31.05%, 100.00%]` on $k=5..8$ (seed 0 hits `100.00%`, 3/5 seeds $\ge 89.75\%$), compressing two hops per pass via pointer doubling (`27.72%` single-hop `lm_head` readout), consistent with [Sanford, Hsu & Telgarsky (2024)](https://arxiv.org/abs/2402.09268).
   - `cap_std4_k8` (`204,684` params, 4 untied blocks, trained on $k=1..8$ for `1,600` steps): **`62.99%`** `[42.87%, 81.25%]` on $k=1..4$, **`28.61%`** `[25.29%, 30.47%]` on $k=5..8$.
   - `cap_loop4_scalar_k4` (`59,084` params, continuous `--k-encoding scalar` trained on $k=1..4$ for `800` steps): **`70.27%`** `[35.94%, 98.44%]` on $k=1..4$, **`20.35%`** `[17.97%, 22.66%]` zero-shot on $k=5..8$.

2. **Weight-tied loop vs. unrolled untied twin & KV-cached Scratchpad cost (`loop-vs-unrolled-v2`):**
   - `unrolled8_k4` (`404,620` params, 8 untied blocks where blocks 5–8 are untrained on $k \le 4$): **`63.77%`** on $k=1..4$, **`19.96%`** `[19.04%, 20.51%]` on $k=5..8$.
   - `unrolled8_k8` (`404,620` params, all 8 untied blocks trained directly on $k=1..8$ for `1,600` steps): **`63.26%`** on $k=1..4$, **`23.42%`** `[22.26%, 25.10%]` on $k=5..8$.
   - `scratchpad_kv` (`104,716` params, 2-layer causal Scratchpad CoT with KV cache): **`100.00%`** on $k=1..4$, **`99.16%`** `[98.44%, 99.71%]` on $k=5..8$; KV caching cuts $k=8$ greedy decode latency from `3.45 ms` uncached to **`2.06 ms`** cached (`42` token-block visits, `4.26M` FLOPs vs. `112` token-block visits, `11.41M` FLOPs for an 8-pass full-sequence loop).

3. **Carry vs. re-reading `x0`, pass-1 double-scaling, and ARIA follow-ups (`injection-sweep-v2` + `aria-followups-v2`):**
   - `inj_0p0_carry` (pure carry, `0.0 * x0`): **`80.76%`** `[48.15%, 94.73%]` clean $k=5..8$; drops to **`19.41%` / `10.90%`** under pass-1 Gaussian noise $\alpha=2.0 / 4.0$.
   - `inj_1p0_from1` vs. `inj_1p0_from2`: skipping pass-1 injection (`--inject-from-pass 2`) lifts full `1.0 * x0` from **`89.34%`** `[57.33%, 98.73%]` to **`97.07%`** `[93.85%, 98.93%]`, eliminating the seed-0 `2 * x0` failure (`57.33% -> 98.93%`).
   - `inj_0p25_from1`: **`94.47%`** `[85.35%, 99.12%]` clean $k=5..8$.
   - `inj_0p5_from1` vs. `inj_0p5_from2`: skipping pass-1 injection lifts `0.5 * x0` from **`93.56%`** `[83.10%, 99.32%]` to **`99.10%`** `[98.44%, 99.90%]`, with **`100.00%`** `lm_head` intermediate-hop readout on passes 1–4 and **`99.69%`** on pass 6.
   - ARIA-proposed follow-ups (`aria-followups-v2`): `aria_f1_gated_0p5_from2` (`58,892` params, **`90.06%`** `[69.82%, 97.07%]`) and `aria_f2_inj_2p0_from1` (`54,732` params, **`83.69%`** `[39.55%, 98.63%]` clean, **`73.09%` / `44.88%`** at noise $\alpha=2.0 / 4.0$ — the highest noise resilience in the study).

4. **Locked single-shot holdout (`final-holdout-v2`, `holdout_seed=31415`, 4,096 unseen graphs per checkpoint):**
   - `inj_0p5_from2`: **`99.19%`** `[98.92%, 99.56%]` (`54,732` params, 1 emitted token)
   - `scratchpad_kv`: **`99.04%`** `[98.10%, 99.90%]` (`104,716` params, $k$ emitted tokens)
   - `inj_1p0_from2`: **`97.33%`** `[94.58%, 98.88%]`
   - `inj_0p5_from1`: **`93.83%`** `[85.16%, 99.07%]`
   - `inj_1p0_from1`: **`89.57%`** `[59.03%, 98.54%]`
   - `aria_f2_inj_2p0_from1`: **`84.05%`** `[63.09%, 94.73%]`
   - `inj_0p0_carry`: **`80.95%`** `[52.20%, 94.19%]`

## Routing Codex CLI through CoreWeave Forge Serverless Inference

`proxy/purokishi.yaml` configures [`purokishi`](https://github.com/kmio11/purokishi) (`v0.2.0+`) to translate OpenAI `/v1/responses` calls into `/v1/chat/completions` on `https://api.inference.wandb.ai/v1` (`moonshotai/Kimi-K2.7-Code` with `zai-org/GLM-5.3` fallback), and `proxy/codex_namespace_bridge.py` unwraps Codex `0.167+` `type: "namespace"` MCP tool definitions (`mcp__wandb`) into flat function tools and back over streaming SSE:

```bash
export WANDB_API_KEY="..."
export PUROKISHI_API_KEY="local-purokishi"
purokishi -config looped-transformer-aria-lab/proxy/purokishi.yaml -frozen &
python3 looped-transformer-aria-lab/proxy/codex_namespace_bridge.py &
```

Point Codex CLI (`~/.codex/config.toml` or an isolated `CODEX_HOME`) at the local bridge and hosted W&B MCP Server:

```toml
model = "moonshotai/Kimi-K2.7-Code"
model_provider = "forge_inference"

[model_providers.forge_inference]
name = "CoreWeave Forge Serverless Inference"
base_url = "http://127.0.0.1:18079/v1"
env_key = "PUROKISHI_API_KEY"
wire_api = "responses"
http_headers = { "OpenAI-Project" = "wandb/looped-transformer-study" }

[mcp_servers.wandb]
url = "https://mcp.withwandb.com/mcp"
bearer_token_env_var = "WANDB_API_KEY"
```

## Run benchmark locally without credentials

```bash
# Run built-in architecture, KV-cache parity, and flag self-tests:
uv run --with torch --with numpy python looped-transformer-aria-lab/looped_transformer_lab.py --self-test

# Train the winning weight-tied loop arm locally (zero network calls):
uv run --with torch --with numpy python looped-transformer-aria-lab/looped_transformer_lab.py \
  --arm arm_c_looped_extrap --input-inject --inject-scale 0.5 --inject-from-pass 2 \
  --arm-tag inj_0p5_from2 --seeds 0,1,2,3,4 --steps 800 --dry-run

# Train the 8-layer untied twin locally:
uv run --with torch --with numpy python looped-transformer-aria-lab/looped_transformer_lab.py \
  --arm arm_e_unrolled_extrap --train-max-k 8 --arm-tag unrolled8_k8 --seeds 0 --steps 1600 --dry-run
```

## Files

- `looped_transformer_lab.py` — self-contained task generator, `PointerModel` (`standard`, `unrolled`, `looped`, and KV-cached `scratchpad`), `--inject-from-pass`, `--k-encoding scalar`, pass-by-pass `lm_head` readout, pass-1 Gaussian noise test, checkpoint artifact saving, `--eval-only --holdout-seed` single-shot evaluation, W&B run logger, and Weave Evaluation runner.
- `proxy/purokishi.yaml` & `proxy/codex_namespace_bridge.py` — local Responses-API + MCP namespace proxy routing Codex CLI to CoreWeave Forge Serverless Inference.
- `arm_specs.json` — manifest of Round 1 (`v1`) and Round 2 (`v2`) arms, exact CLI flags, holdout evaluation specs, and verified primary references.
- `prompts/codex_session_prompts.md` — the 4-turn Codex CLI prompt sequence used in the study (`Turn 1` parallel ARIA fan-out, `Turn 2` ARIA-proposed follow-ups, `Turn 3` frozen `holdout_seed=31415` evaluation, `Turn 4` Forge Workspace assembly).
