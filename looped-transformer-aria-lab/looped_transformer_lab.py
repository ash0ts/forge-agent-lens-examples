#!/usr/bin/env python3
"""Shared toy benchmark for comparing standard, weight-tied looped, and scratchpad transformers.

Task: k-hop pointer chasing over a random permutation of N=12 nodes.
  - Context tokens: [p[0], p[1], ..., p[11], QUERY, start] (+ optional HOP_k token when loop count is fixed)
  - Training distribution: k in {1, 2, 3, 4} hops only
  - Held-out evaluation:   k in {1, 2, 3, 4} (in-distribution) and k in {5, 6, 7, 8} (unseen depth extrapolation)

All arms share the same data generator, AdamW schedule, step budget, and held-out seed=2026 evaluation split.
The architectural difference between the standard stack and the looped model is Aaron Batilo's single loop:
  - Standard: apply L distinct transformer blocks once
  - Looped:   apply 1 weight-tied transformer block T times before emitting a single answer token
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODES = 12
QUERY_TOK = N_NODES          # 12
HOP_TOKEN_BASE = N_NODES + 1 # 13..24 encode hop count k=1..12 when fixed-depth models need k in the prompt
VOCAB_SIZE = N_NODES + 16    # 28 total tokens


def make_pointer_batch(
    batch_size: int,
    hops_choices: list[int],
    rng: np.random.Generator,
    include_k_token: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of pointer-chasing graphs over N_NODES=12."""
    max_k = max(max(hops_choices), 8)
    seq_len = N_NODES + (3 if include_k_token else 2)
    seqs = np.zeros((batch_size, seq_len), dtype=np.int64)
    all_hops = np.zeros((batch_size, max_k + 1), dtype=np.int64)
    ks = np.zeros((batch_size,), dtype=np.int64)

    for b in range(batch_size):
        perm = rng.permutation(N_NODES)
        start = int(rng.integers(0, N_NODES))
        k = int(rng.choice(hops_choices))
        seqs[b, :N_NODES] = perm
        seqs[b, N_NODES] = QUERY_TOK
        seqs[b, N_NODES + 1] = start
        if include_k_token:
            seqs[b, N_NODES + 2] = HOP_TOKEN_BASE + k
        cur = start
        all_hops[b, 0] = cur
        for step in range(1, max_k + 1):
            cur = int(perm[cur])
            all_hops[b, step] = cur
        ks[b] = k

    return torch.from_numpy(seqs), torch.from_numpy(all_hops), torch.from_numpy(ks)


def make_scratchpad_batch(
    batch_size: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced causal batch where the model emits every intermediate hop token y_1..y_k."""
    seqs = np.zeros((batch_size, N_NODES + 2 + k - 1), dtype=np.int64)
    targets = np.zeros((batch_size, k), dtype=np.int64)
    for b in range(batch_size):
        perm = rng.permutation(N_NODES)
        start = int(rng.integers(0, N_NODES))
        seqs[b, :N_NODES] = perm
        seqs[b, N_NODES] = QUERY_TOK
        seqs[b, N_NODES + 1] = start
        cur = start
        for t in range(k):
            cur = int(perm[cur])
            targets[b, t] = cur
            if t < k - 1:
                seqs[b, N_NODES + 2 + t] = cur
    return torch.from_numpy(seqs), torch.from_numpy(targets)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = None
        if self.causal:
            s = x.size(1)
            attn_mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln2(x))


class PointerModel(nn.Module):
    """Unified module supporting standard untied, weight-tied looped, and causal scratchpad arms."""

    def __init__(
        self,
        arch: str,
        n_layers: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        input_inject: bool = False,
        inject_scale: float = 1.0,
        gated: bool = False,
        step_embed: bool = False,
    ) -> None:
        super().__init__()
        self.arch = arch  # "standard" | "looped" | "scratchpad"
        self.n_layers = n_layers
        self.input_inject = input_inject
        self.inject_scale = inject_scale
        self.gated = gated
        self.step_embed = step_embed

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(32, d_model)

        if arch == "standard":
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=False) for _ in range(n_layers)])
        elif arch == "looped":
            self.block = TransformerBlock(d_model, n_heads, causal=False)
            if gated:
                self.gate = nn.Linear(d_model, d_model)
            if step_embed:
                self.loop_emb = nn.Embedding(16, d_model)
        elif arch == "scratchpad":
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=True) for _ in range(n_layers)])
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_NODES)

    def forward_latent(self, idx: torch.Tensor, loops: int | None = None) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
        s = idx.size(1)
        x0 = self.tok_emb(idx) + self.pos_emb(torch.arange(s, device=idx.device))[None, :]
        x = x0
        pass_logits: list[torch.Tensor] = []
        pass_norms: list[float] = []

        if self.arch == "standard":
            for block in self.blocks:
                x = block(x)
                pass_norms.append(float(x[:, -1, :].detach().norm(dim=-1).mean().item()))
                pass_logits.append(self.head(self.ln_f(x[:, -1, :])))
        elif self.arch == "looped":
            total_loops = self.n_layers if loops is None else loops
            for t in range(total_loops):
                inp = x + self.inject_scale * x0 if self.input_inject else x
                if self.step_embed:
                    inp = inp + self.loop_emb(torch.tensor(min(t, 15), device=idx.device))
                cand = self.block(inp)
                if self.gated:
                    g = torch.sigmoid(self.gate(x))
                    x = x + g * (cand - x)
                else:
                    x = cand
                pass_norms.append(float(x[:, -1, :].detach().norm(dim=-1).mean().item()))
                pass_logits.append(self.head(self.ln_f(x[:, -1, :])))
        else:
            raise RuntimeError("Use forward_scratchpad for causal scratchpad arm")

        return pass_logits[-1], pass_logits, pass_norms

    def forward_scratchpad(self, idx: torch.Tensor) -> torch.Tensor:
        s = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(s, device=idx.device))[None, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "arm_a_standard": {
        "arch": "standard",
        "n_layers": 4,
        "adaptive_loops": False,
        "include_k_token": True,
        "input_inject": False,
        "gated": False,
        "step_embed": False,
        "label": "Arm A: Standard 4-Layer (untied)",
    },
    "arm_b_looped_fixed": {
        "arch": "looped",
        "n_layers": 4,
        "adaptive_loops": False,
        "include_k_token": True,
        "input_inject": False,
        "gated": False,
        "step_embed": False,
        "label": "Arm B: Looped 1-Block (fixed T=4)",
    },
    "arm_c_looped_extrap": {
        "arch": "looped",
        "n_layers": 4,
        "adaptive_loops": True,
        "include_k_token": False,
        "input_inject": False,
        "gated": False,
        "step_embed": False,
        "label": "Arm C: Looped 1-Block (adaptive T=k)",
    },
    "arm_d_scratchpad": {
        "arch": "scratchpad",
        "n_layers": 2,
        "adaptive_loops": False,
        "include_k_token": False,
        "input_inject": False,
        "gated": False,
        "step_embed": False,
        "label": "Arm D: Scratchpad CoT (tokens out loud)",
    },
}


def train_and_evaluate_one_seed(
    cfg: dict[str, Any],
    seed: int,
    steps: int = 900,
    batch_size: int = 128,
    lr: float = 3e-3,
    eval_n: int = 256,
    wandb_run: Any = None,
) -> tuple[PointerModel, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    model = PointerModel(
        arch=cfg["arch"],
        n_layers=cfg["n_layers"],
        d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4),
        input_inject=cfg["input_inject"],
        inject_scale=float(cfg.get("inject_scale", 1.0)),
        gated=cfg["gated"],
        step_embed=cfg["step_embed"],
    )
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if cfg["arch"] == "scratchpad":
            k = int(rng.integers(1, 5))
            x, ys = make_scratchpad_batch(batch_size, k, rng)
            logits = model.forward_scratchpad(x)[:, N_NODES + 1 :, :]
            loss = F.cross_entropy(logits.reshape(-1, N_NODES), ys.reshape(-1))
            with torch.no_grad():
                step_acc = float((logits[:, -1, :].argmax(-1) == ys[:, -1]).float().mean().item())
        elif cfg["adaptive_loops"]:
            k = int(rng.integers(1, 5))
            x, all_hops, _ = make_pointer_batch(batch_size, [k], rng, include_k_token=False)
            logits, _, _ = model.forward_latent(x, loops=k)
            loss = F.cross_entropy(logits, all_hops[:, k])
            with torch.no_grad():
                step_acc = float((logits.argmax(-1) == all_hops[:, k]).float().mean().item())
        else:
            x, all_hops, ks = make_pointer_batch(batch_size, [1, 2, 3, 4], rng, include_k_token=True)
            targets = all_hops[torch.arange(batch_size), ks]
            logits, _, _ = model.forward_latent(x, loops=cfg["n_layers"])
            loss = F.cross_entropy(logits, targets)
            with torch.no_grad():
                step_acc = float((logits.argmax(-1) == targets).float().mean().item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if wandb_run is not None and (step == 1 or step % 100 == 0):
            wandb_run.log({"train/step": step, "train/loss": float(loss.item()), "train/acc": step_acc})

    train_wall_s = time.perf_counter() - t0

    # Deterministic held-out evaluation across k=1..8
    model.eval()
    eval_rng = np.random.default_rng(2026)
    acc_by_k: dict[int, float] = {}
    tokens_by_k: dict[int, int] = {}

    with torch.no_grad():
        for k in range(1, 9):
            x, all_hops, _ = make_pointer_batch(eval_n, [k], eval_rng, include_k_token=cfg["include_k_token"])
            y_true = all_hops[:, k]
            if cfg["arch"] == "scratchpad":
                cur = x
                for _ in range(k):
                    nxt = model.forward_scratchpad(cur)[:, -1, :].argmax(-1, keepdim=True)
                    cur = torch.cat([cur, nxt], dim=1)
                preds = cur[:, -1]
                tokens_by_k[k] = k
            else:
                loops = k if cfg["adaptive_loops"] else cfg["n_layers"]
                logits, _, _ = model.forward_latent(x, loops=loops)
                preds = logits.argmax(-1)
                tokens_by_k[k] = 1
            acc_by_k[k] = round(float((preds == y_true).float().mean().item()), 4)

        # Pass-by-pass monitorability + residual norm diagnostics on unseen 6-hop chains (and 4-hop in-dist)
        probe_rng = np.random.default_rng(4242)
        x6, hops6, _ = make_pointer_batch(eval_n, [6], probe_rng, include_k_token=cfg["include_k_token"])
        pass_rows: list[dict[str, Any]] = []
        sample_traces: list[dict[str, Any]] = []

        if cfg["arch"] == "scratchpad":
            cur = x6
            for t in range(1, 7):
                nxt = model.forward_scratchpad(cur)[:, -1, :].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
                hop_acc = float((nxt.squeeze(-1) == hops6[:, t]).float().mean().item())
                pass_rows.append({"iter": t, "true_hop_acc": round(hop_acc, 4), "residual_norm": 0.0})
            for i in range(4):
                sample_traces.append({
                    "example_id": i,
                    "start_node": int(hops6[i, 0].item()),
                    "true_hops_1_to_6": str(hops6[i, 1:7].tolist()),
                    "emitted_cot_tokens": str(cur[i, N_NODES + 2 :].tolist()),
                    "latent_logit_lens_passes": "n/a (explicit token CoT)",
                    "final_correct": bool(cur[i, -1].item() == hops6[i, 6].item()),
                })
            logit_lens_k4 = 1.0
            norm_ratio = 1.0
        else:
            probe_loops = 6 if cfg["adaptive_loops"] else cfg["n_layers"]
            _, pass_logits, pass_norms = model.forward_latent(x6, loops=probe_loops)
            for t in range(probe_loops):
                hop_idx = min(t + 1, 6)
                hop_acc = float((pass_logits[t].argmax(-1) == hops6[:, hop_idx]).float().mean().item())
                pass_rows.append({
                    "iter": t + 1,
                    "true_hop_acc": round(hop_acc, 4),
                    "residual_norm": round(pass_norms[t], 3),
                })
            logit_lens_k4 = round(float(np.mean([r["true_hop_acc"] for r in pass_rows[: min(4, len(pass_rows))]])), 4)
            norm_ratio = round(pass_norms[-1] / max(pass_norms[0], 1e-6), 3)
            for i in range(4):
                decoded_passes = [int(pass_logits[t][i].argmax(-1).item()) for t in range(probe_loops)]
                sample_traces.append({
                    "example_id": i,
                    "start_node": int(hops6[i, 0].item()),
                    "true_hops_1_to_6": str(hops6[i, 1:7].tolist()),
                    "emitted_cot_tokens": f"[{decoded_passes[-1]}] (1 final token only; 0 CoT tokens)",
                    "latent_logit_lens_passes": str(decoded_passes),
                    "final_correct": bool(decoded_passes[-1] == int(hops6[i, 6].item())),
                })

    in_dist_acc = round(float(np.mean([acc_by_k[k] for k in (1, 2, 3, 4)])), 4)
    extrap_acc = round(float(np.mean([acc_by_k[k] for k in (5, 6, 7, 8)])), 4)

    # Independent untouched holdout graphs (seed=9090) for post-selection confirmation
    holdout_rng = np.random.default_rng(9090)
    holdout_accs: list[float] = []
    with torch.no_grad():
        for k in range(5, 9):
            xh, hopsh, _ = make_pointer_batch(eval_n, [k], holdout_rng, include_k_token=cfg["include_k_token"])
            yh = hopsh[:, k]
            if cfg["arch"] == "scratchpad":
                curh = xh
                for _ in range(k):
                    nxth = model.forward_scratchpad(curh)[:, -1, :].argmax(-1, keepdim=True)
                    curh = torch.cat([curh, nxth], dim=1)
                predsh = curh[:, -1]
            else:
                loopsh = k if cfg["adaptive_loops"] else cfg["n_layers"]
                logitsh, _, _ = model.forward_latent(xh, loops=loopsh)
                predsh = logitsh.argmax(-1)
            holdout_accs.append(float((predsh == yh).float().mean().item()))
    holdout_extrap_acc = round(float(np.mean(holdout_accs)), 4)

    metrics: dict[str, Any] = {
        "model/param_count": param_count,
        "train/wall_seconds": round(train_wall_s, 3),
        "eval/in_dist_acc": in_dist_acc,
        "eval/extrap_acc": extrap_acc,
        "eval/holdout_extrap_acc": holdout_extrap_acc,
        "eval/emitted_tokens_k8": tokens_by_k[8],
        "monitor/text_cot_legible": 1.0 if cfg["arch"] == "scratchpad" else 0.0,
        "monitor/logit_lens_hop_acc": logit_lens_k4,
        "monitor/norm_ratio_last_first": norm_ratio,
        "pass_rows": pass_rows,
        "acc_by_k": acc_by_k,
        "tokens_by_k": tokens_by_k,
    }
    for k, v in acc_by_k.items():
        metrics[f"eval/acc_k{k}"] = v

    return model, metrics, sample_traces


def run_weave_eval_for_model(entity: str, project: str, arm_key: str, cfg: dict[str, Any], model: PointerModel) -> str | None:
    """Log a compact 48-example (6 graphs x 8 hop depths) Weave Evaluation on held-out cases."""
    import weave

    weave.init(f"{entity}/{project}")
    rng = np.random.default_rng(7777)
    dataset_rows: list[dict[str, Any]] = []
    for k in range(1, 9):
        x, all_hops, _ = make_pointer_batch(6, [k], rng, include_k_token=cfg["include_k_token"])
        for i in range(6):
            dataset_rows.append({
                "hop_k": k,
                "tokens": x[i].tolist(),
                "start_node": int(all_hops[i, 0].item()),
                "target_node": int(all_hops[i, k].item()),
            })

    @weave.op()
    def predict_pointer(hop_k: int, tokens: list[int]) -> dict[str, Any]:
        x_t = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            if cfg["arch"] == "scratchpad":
                cur = x_t
                for _ in range(hop_k):
                    nxt = model.forward_scratchpad(cur)[:, -1, :].argmax(-1, keepdim=True)
                    cur = torch.cat([cur, nxt], dim=1)
                pred = int(cur[0, -1].item())
                emitted = hop_k
            else:
                loops = hop_k if cfg["adaptive_loops"] else cfg["n_layers"]
                logits, _, _ = model.forward_latent(x_t, loops=loops)
                pred = int(logits[0].argmax(-1).item())
                emitted = 1
        return {"pred_node": pred, "emitted_tokens": emitted}

    @weave.op()
    def exact_match(target_node: int, hop_k: int, output: dict[str, Any]) -> dict[str, Any]:
        ok = int(output["pred_node"] == target_node)
        return {
            "correct": ok,
            "in_dist_correct": ok if hop_k <= 4 else None,
            "extrap_correct": ok if hop_k >= 5 else None,
            "emitted_tokens": int(output["emitted_tokens"]),
        }

    evaluation = weave.Evaluation(
        name=f"pointer-chase-{arm_key}",
        dataset=dataset_rows,
        scorers=[exact_match],
    )
    summary = asyncio.run(evaluation.evaluate(predict_pointer))
    return json.dumps(summary, default=str)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=[*ARM_CONFIGS.keys(), "custom"], required=True)
    p.add_argument("--arch", choices=["standard", "looped", "scratchpad"], default="looped")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--adaptive-loops", action="store_true")
    p.add_argument("--input-inject", action="store_true")
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--gated", action="store_true")
    p.add_argument("--step-embed", action="store_true")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--arm-tag", type=str, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--entity", type=str, default="wandb")
    p.add_argument("--project", type=str, default="looped-transformer-study")
    p.add_argument("--group", type=str, default="four-arms-v1")
    p.add_argument("--run-weave-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.arm == "custom":
        arm_key = args.arm_tag or "autoresearch_custom"
        cfg = {
            "arch": args.arch,
            "n_layers": args.n_layers,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "adaptive_loops": args.adaptive_loops,
            "include_k_token": not args.adaptive_loops and args.arch != "scratchpad",
            "input_inject": args.input_inject,
            "inject_scale": args.inject_scale,
            "gated": args.gated,
            "step_embed": args.step_embed,
            "label": args.label or arm_key,
        }
    else:
        arm_key = args.arm_tag or args.arm
        cfg = dict(ARM_CONFIGS[args.arm])
        cfg["d_model"] = args.d_model
        cfg["n_heads"] = args.n_heads
        cfg["inject_scale"] = args.inject_scale
        if args.input_inject:
            cfg["input_inject"] = True
        if args.gated:
            cfg["gated"] = True
        if args.step_embed:
            cfg["step_embed"] = True
        if args.label:
            cfg["label"] = args.label

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        run = None
        if not args.dry_run:
            import wandb

            run = wandb.init(
                entity=args.entity,
                project=args.project,
                group=args.group,
                job_type=arm_key,
                name=f"{arm_key}-seed{seed}",
                tags=[arm_key, cfg["arch"], args.group],
                config={
                    "arm": arm_key,
                    "arm_label": cfg["label"],
                    **cfg,
                    "seed": seed,
                    "steps": args.steps,
                    "lr": args.lr,
                    "task": "pointer_chase_n12_train_k1to4_eval_k1to8",
                },
                settings=wandb.Settings(save_code=True),
            )
            run.define_metric("train/step")
            run.define_metric("train/*", step_metric="train/step")
            run.define_metric("curve_hop/k")
            run.define_metric("curve_hop/*", step_metric="curve_hop/k")
            run.define_metric("curve_pass/iter")
            run.define_metric("curve_pass/*", step_metric="curve_pass/iter")

        model, metrics, sample_traces = train_and_evaluate_one_seed(
            cfg=cfg, seed=seed, steps=args.steps, lr=args.lr, wandb_run=run
        )

        if run is not None:
            import wandb

            for k in range(1, 9):
                run.log({
                    "curve_hop/k": k,
                    "curve_hop/accuracy": metrics["acc_by_k"][k],
                    "curve_hop/emitted_tokens": metrics["tokens_by_k"][k],
                })
            for pr in metrics["pass_rows"]:
                run.log({
                    "curve_pass/iter": pr["iter"],
                    "curve_pass/logit_lens_true_hop_acc": pr["true_hop_acc"],
                    "curve_pass/residual_norm": pr["residual_norm"],
                })
            for key, val in metrics.items():
                if key not in ("pass_rows", "acc_by_k", "tokens_by_k"):
                    run.summary[key] = val

            if seed == seeds[0]:
                tbl = wandb.Table(
                    columns=[
                        "example_id",
                        "start_node",
                        "true_hops_1_to_6",
                        "emitted_cot_tokens",
                        "latent_logit_lens_passes",
                        "final_correct",
                    ],
                    data=[[r[c] for c in (
                        "example_id",
                        "start_node",
                        "true_hops_1_to_6",
                        "emitted_cot_tokens",
                        "latent_logit_lens_passes",
                        "final_correct",
                    )] for r in sample_traces],
                )
                run.log({"monitor/sample_traces": tbl})

            run_url = run.url
            run_id = run.id
            run.finish()
        else:
            run_url = None
            run_id = f"dry-{arm_key}-s{seed}"

        weave_summary = None
        if not args.dry_run and args.run_weave_eval and seed == seeds[0]:
            weave_summary = run_weave_eval_for_model(args.entity, args.project, arm_key, cfg, model)

        all_seed_results.append({
            "seed": seed,
            "run_id": run_id,
            "run_url": run_url,
            "param_count": metrics["model/param_count"],
            "in_dist_acc": metrics["eval/in_dist_acc"],
            "extrap_acc": metrics["eval/extrap_acc"],
            "holdout_extrap_acc": metrics["eval/holdout_extrap_acc"],
            "acc_by_k": metrics["acc_by_k"],
            "emitted_tokens_k8": metrics["eval/emitted_tokens_k8"],
            "logit_lens_hop_acc": metrics["monitor/logit_lens_hop_acc"],
            "norm_ratio_last_first": metrics["monitor/norm_ratio_last_first"],
            "train_wall_seconds": metrics["train/wall_seconds"],
            "weave_eval": weave_summary,
        })

    in_mean = round(float(np.mean([r["in_dist_acc"] for r in all_seed_results])), 4)
    ex_mean = round(float(np.mean([r["extrap_acc"] for r in all_seed_results])), 4)
    ex_std = round(float(np.std([r["extrap_acc"] for r in all_seed_results])), 4)
    ho_mean = round(float(np.mean([r["holdout_extrap_acc"] for r in all_seed_results])), 4)
    print(json.dumps({
        "arm": arm_key,
        "label": cfg["label"],
        "seeds": seeds,
        "param_count": all_seed_results[0]["param_count"],
        "mean_in_dist_acc_k1_4": in_mean,
        "mean_extrap_acc_k5_8": ex_mean,
        "std_extrap_acc_k5_8": ex_std,
        "mean_holdout_extrap_acc_k5_8": ho_mean,
        "emitted_tokens_k8": all_seed_results[0]["emitted_tokens_k8"],
        "runs": all_seed_results,
    }, indent=2))


if __name__ == "__main__":
    main()
