# Looped Transformer ARIA Lab

When someone asks what a looped transformer actually changes, the cleanest explanation is three lines of code. A normal transformer runs each layer once and reasons across tokens by writing its steps out loud; a looped transformer wraps `for _ in range(T):` around shared weights so hidden states refine across passes before emitting a single token.

This directory holds the reproducible CPU benchmark, proxy configuration, arm definitions, and Codex prompts behind our Forge walkthrough (`wandb/looped-transformer-study`), where **one Codex CLI session** (running `moonshotai/Kimi-K2.7-Code` on **CoreWeave Forge Serverless Inference** via [`purokishi`](https://github.com/wandb/purokishi)) fans out **parallel ARIA sessions** over the W&B MCP Server (`https://mcp.withwandb.com/mcp`), runs ARIA's **autoresearch** loop on unseen depth extrapolation, publishes a shared Workspace and Forge Report, and traces the orchestrating Codex chat itself into **CoreWeave Agent Lens** via `@coreweave/forge-agent-lens-for-codex`.

## What the benchmark tests (toy scale, CPU PyTorch)

Every arm trains on the same **pointer-chasing** task over permutations of 12 nodes:
- **Train only on $k \in \{1, 2, 3, 4\}$ hops** (`3` seeds `0,1,2`, `800` AdamW steps, `d_model=64`, `n_heads=4`).
- **Evaluate on held-out `seed=2026`** across $k \in \{1..4\}$ (in-distribution) and **$k \in \{5, 6, 7, 8\}$ (unseen depth extrapolation)**, plus an untouched post-selection holdout (`seed=9090`) and a 48-example Weave Evaluation (`seed=7777`) per arm.

### Four controlled arms (`four-arms-v1`)

1. **`arm_a_standard`** — 4 distinct untied transformer blocks (`204,684` parameters), 1 pass, 1 emitted token (`~20.1%` unseen $k=5..8$).
2. **`arm_b_looped_fixed`** — 1 weight-tied block applied fixed $T=4$ times (`54,732` parameters, 3.7x smaller), 1 emitted token (`~18.4%` unseen $k=5..8$).
3. **`arm_c_looped_extrap`** — Same 1 weight-tied block (`54,732` parameters) trained at $T=k$ ($k \le 4$) and dialed to $T=5..8$ loops at test time with 0 chain-of-thought tokens (`71.06% ± 14.25%` unseen $k=5..8$, `98.57%` internal logit-lens intermediate hop accuracy).
4. **`arm_d_scratchpad`** — 2-layer causal transformer (`104,716` parameters) that writes every intermediate hop out loud as tokens (`99.42%` unseen $k=5..8$ at `8` sequential decode steps for $k=8$).

### Autoresearch rounds (`autoresearch-v1`)

Starting from `arm_c_looped_extrap`, ARIA tests hypotheses for unseen $k=5..8$ extrapolation under a 65k-parameter, 1-token budget:
- **Round 1 (`auto_r1_input_inject`)**: full input injection `x + 1.0 * x0` ([Yang et al., 2023](https://arxiv.org/abs/2311.12424)) — raises the 3-seed mean from `71.06%` to `84.54%`, while regressing seed 0 (`78.23% -> 57.33%`) because `x = x0` makes pass 1 see `2 * x0`.
- **Round 2 (`auto_r2_gated`)**: per-pass learned residual gating `x + sigmoid(gate(x)) * (block(x + x0) - x)` (mirroring gated progressive updates in [Z.ai's GLM-5.4](https://z.ai/blog/glm-5.4), `58,892` parameters) — recovers worst-seed accuracy (`62.99%`) and tightens variance (`13.97%` std) at `82.75%` mean.
- **Round 3 (`auto_r3_half_inject`)**: half-strength input injection `x + 0.5 * x0` (`54,732` parameters, zero added parameters) — reaches **`98.76% ± 0.60%`** on unseen $k=5..8$ (`24/24` unseen Weave Evaluation rows), matching the 8-token scratchpad within `0.7` percentage points using half the parameters and 1 emitted token.

## Routing Codex CLI through CoreWeave Forge Serverless Inference

`proxy/purokishi.yaml` configures `purokishi` (`v0.2.0+`) to translate OpenAI `/v1/responses` calls into `/v1/chat/completions` on `https://api.inference.wandb.ai/v1` (`moonshotai/Kimi-K2.7-Code` with `zai-org/GLM-5.3` fallback), and `proxy/codex_namespace_bridge.py` unwraps Codex `0.167+` `type: "namespace"` MCP tool definitions (`mcp__wandb`) into flat function tools and back over streaming SSE:

```bash
export WANDB_API_KEY="..."
purokishi --config looped-transformer-aria-lab/proxy/purokishi.yaml --frozen &
python3 looped-transformer-aria-lab/proxy/codex_namespace_bridge.py &
```

## Run benchmark locally without credentials

```bash
uv run --with torch --with numpy python looped-transformer-aria-lab/looped_transformer_lab.py \
  --arm arm_c_looped_extrap --input-inject --inject-scale 0.5 --seeds 0,1,2 --steps 800 --dry-run
```

## Files

- `looped_transformer_lab.py` — self-contained task generator, `PointerModel`, pass-by-pass logit-lens & residual-norm diagnostics, untouched holdout split (`seed=9090`), W&B run logger, and Weave Evaluation runner.
- `proxy/purokishi.yaml` & `proxy/codex_namespace_bridge.py` — local Responses-API + MCP namespace proxy routing Codex CLI to Forge Serverless Inference.
- `arm_specs.json` — arm manifest plus verified arXiv IDs (`1807.03819`, `2311.12424`, `2409.15647`, `2502.17416`) and primary shipped-model sources ([Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/), [GLM-5.4](https://z.ai/blog/glm-5.4)).
- `prompts/codex_session_prompts.md` — the 4-turn Codex CLI prompt sequence used in the walkthrough.
