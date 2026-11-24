#!/usr/bin/env python3
"""Controlled toy benchmark for teaching and testing looped transformers.

Task: k-hop pointer chasing over a random permutation of N=12 nodes.
  - Context tokens: [p[0], p[1], ..., p[11], QUERY, start] (+ optional hop indicator when loop count is fixed)
  - Default training distribution: k in {1..train_max_k} (default 4; configurable up to 8 for capacity-ceiling checks)
  - Held-out evaluation (seed=2026): k in {1..4} (in-distribution) and k in {5..8} (unseen depth extrapolation)
  - Secondary split (seed=9090): logged per run alongside seed=2026 (not an untouched post-selection holdout)
  - Final post-selection holdout: run once via `--eval-only --holdout-seed <fresh_seed>` on saved checkpoints

Supported architectures (`--arch`):
  - `standard`: L distinct untied transformer blocks applied every forward pass (1 emitted token)
  - `unrolled`: L distinct untied transformer blocks where adaptive mode runs the first `k` blocks (1 emitted token)
  - `looped`:   1 weight-tied transformer block applied `T` times before emitting 1 answer token
  - `scratchpad`: L-layer causal transformer emitting all `k` intermediate hops out loud (KV-cached decode)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODES = 12
QUERY_TOK = N_NODES          # 12
HOP_TOKEN_BASE = N_NODES + 1 # 13..24 encode hop count k=1..12 when fixed-depth models receive k in the prompt
VOCAB_SIZE = N_NODES + 16    # 28 total tokens
NOISE_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]


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
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
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

    def forward_cached(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Exact causal attention step with KV caching using the same MultiheadAttention weights."""
        bsz, seq_len, d_model = x.shape
        h = self.ln1(x)
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        attn_out = F.linear(attn_out, self.attn.out_proj.weight, self.attn.out_proj.bias)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class PointerModel(nn.Module):
    """Unified module supporting standard untied, unrolled untied, weight-tied looped, and causal scratchpad arms."""

    def __init__(
        self,
        arch: str,
        n_layers: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        input_inject: bool = False,
        inject_scale: float = 1.0,
        inject_from_pass: int = 1,
        gated: bool = False,
        step_embed: bool = False,
        k_encoding: str = "token",
    ) -> None:
        super().__init__()
        self.arch = arch  # "standard" | "unrolled" | "looped" | "scratchpad"
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.input_inject = input_inject
        self.inject_scale = inject_scale
        self.inject_from_pass = inject_from_pass
        self.gated = gated
        self.step_embed = step_embed
        self.k_encoding = k_encoding

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(32, d_model)
        if k_encoding == "scalar":
            self.k_proj = nn.Sequential(
                nn.Linear(2, d_model),
                nn.Tanh(),
                nn.Linear(d_model, d_model),
            )

        if arch in ("standard", "unrolled"):
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=False) for _ in range(n_layers)])
            if arch == "unrolled" and input_inject:
                pass
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

    def _embed_prompt(self, idx: torch.Tensor) -> torch.Tensor:
        s = idx.size(1)
        pos = self.pos_emb(torch.arange(s, device=idx.device))[None, :]
        if self.k_encoding == "scalar" and s == N_NODES + 3:
            base_idx = idx.clone()
            k_vals = (idx[:, -1] - HOP_TOKEN_BASE).float().clamp(min=1.0, max=16.0)
            base_idx[:, -1] = QUERY_TOK
            emb = self.tok_emb(base_idx) + pos
            ang = (k_vals / 8.0) * (math.pi / 2.0)
            feats = torch.stack([k_vals / 8.0, torch.sin(ang)], dim=-1)
            emb[:, -1, :] = emb[:, -1, :] + self.k_proj(feats)
            return emb
        return self.tok_emb(idx) + pos

    def forward_latent(
        self,
        idx: torch.Tensor,
        loops: int | None = None,
        entry_noise_alpha: float = 0.0,
        noise_gen: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
        x0 = self._embed_prompt(idx)
        x = x0
        if entry_noise_alpha > 0.0:
            # Perturb initial loop state on the query position (Lens 5, arXiv:2610.18127) while keeping x0 intact
            z = torch.randn(x[:, -1, :].shape, generator=noise_gen, device=x.device, dtype=x.dtype)
            state_norm = x[:, -1, :].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            x = x.clone()
            x[:, -1, :] = x[:, -1, :] + entry_noise_alpha * (state_norm / z_norm) * z

        pass_logits: list[torch.Tensor] = []
        pass_norms: list[float] = []

        if self.arch == "standard":
            for block in self.blocks:
                x = block(x)
                pass_norms.append(float(x[:, -1, :].detach().norm(dim=-1).mean().item()))
                pass_logits.append(self.head(self.ln_f(x[:, -1, :])))
        elif self.arch == "unrolled":
            total_loops = self.n_layers if loops is None else min(loops, len(self.blocks))
            for t in range(total_loops):
                use_inject = self.input_inject and ((t + 1) >= self.inject_from_pass)
                inp = x + self.inject_scale * x0 if use_inject else x
                x = self.blocks[t](inp)
                pass_norms.append(float(x[:, -1, :].detach().norm(dim=-1).mean().item()))
                pass_logits.append(self.head(self.ln_f(x[:, -1, :])))
        elif self.arch == "looped":
            total_loops = self.n_layers if loops is None else loops
            for t in range(total_loops):
                use_inject = self.input_inject and ((t + 1) >= self.inject_from_pass)
                inp = x + self.inject_scale * x0 if use_inject else x
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

    @torch.no_grad()
    def decode_scratchpad_cached(self, idx_prefix: torch.Tensor, steps_k: int) -> torch.Tensor:
        """Autoregressively decode `steps_k` scratchpad tokens with per-layer KV caching.

        Returns tensor of emitted tokens with shape `(B, steps_k)`.
        """
        s0 = idx_prefix.size(1)
        x = self.tok_emb(idx_prefix) + self.pos_emb(torch.arange(s0, device=idx_prefix.device))[None, :]
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, kv = block.forward_cached(x, kv_cache=None)
            caches.append(kv)
        nxt = self.head(self.ln_f(x[:, -1, :])).argmax(-1, keepdim=True)
        emitted = [nxt]

        for step in range(1, steps_k):
            pos_idx = torch.tensor([s0 + step - 1], device=idx_prefix.device)
            x_step = self.tok_emb(nxt) + self.pos_emb(pos_idx)[None, :]
            for li, block in enumerate(self.blocks):
                x_step, caches[li] = block.forward_cached(x_step, kv_cache=caches[li])
            nxt = self.head(self.ln_f(x_step[:, -1, :])).argmax(-1, keepdim=True)
            emitted.append(nxt)

        return torch.cat(emitted, dim=1)


def estimate_inference_cost(cfg: dict[str, Any], k: int = 8) -> dict[str, Any]:
    """Compute analytic inference cost metrics for a single k-hop query."""
    d = int(cfg.get("d_model", 64))
    n_layers = int(cfg["n_layers"])
    # One TransformerBlock has 4*d^2 (QKV+O) + 8*d^2 (MLP) = 12*d^2 weight multiply-adds (24*d^2 FLOPs/token)
    linear_flops_per_tok = 24 * d * d

    if cfg["arch"] == "scratchpad":
        s0 = N_NODES + 2  # 14 prompt tokens
        # With KV cache: prefill runs s0 tokens through n_layers; (k-1) decode steps run 1 token through n_layers
        prefill_visits = n_layers * s0
        decode_visits = n_layers * max(k - 1, 0)
        token_block_visits = prefill_visits + decode_visits
        seq_block_calls = n_layers * k  # k sequential autoregressive steps x n_layers
        # Attention QK^T + AV FLOPs: 4 * d * sum_t seq_len(t) per layer
        prefill_attn_pairs = s0 * (s0 + 1) // 2
        decode_attn_pairs = sum(s0 + step for step in range(1, k))
        attn_flops = 4 * d * n_layers * (prefill_attn_pairs + decode_attn_pairs)
        head_flops = 2 * d * N_NODES * k
        emitted = k
    else:
        s0 = N_NODES + (3 if cfg.get("include_k_token", False) else 2)
        loops = k if cfg.get("adaptive_loops", False) else n_layers
        if cfg["arch"] == "unrolled":
            loops = min(loops, n_layers)
        seq_block_calls = loops
        token_block_visits = loops * s0
        attn_flops = 4 * d * loops * (s0 * s0)
        gate_flops = (2 * d * d * s0 * loops) if (cfg["arch"] == "looped" and cfg.get("gated", False)) else 0
        head_flops = 2 * d * N_NODES + gate_flops
        emitted = 1

    flops_est = int(token_block_visits * linear_flops_per_tok + attn_flops + head_flops)
    return {
        "block_applications": int(seq_block_calls),
        "token_block_visits": int(token_block_visits),
        "flops_est": flops_est,
        "emitted_tokens": int(emitted),
    }


ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "arm_a_standard": {
        "arch": "standard",
        "n_layers": 4,
        "adaptive_loops": False,
        "include_k_token": True,
        "input_inject": False,
        "inject_scale": 1.0,
        "inject_from_pass": 1,
        "gated": False,
        "step_embed": False,
        "k_encoding": "token",
        "label": "Arm A: Standard 4-Layer (untied)",
    },
    "arm_b_looped_fixed": {
        "arch": "looped",
        "n_layers": 4,
        "adaptive_loops": False,
        "include_k_token": True,
        "input_inject": False,
        "inject_scale": 1.0,
        "inject_from_pass": 1,
        "gated": False,
        "step_embed": False,
        "k_encoding": "token",
        "label": "Arm B: Looped 1-Block (fixed T=4)",
    },
    "arm_c_looped_extrap": {
        "arch": "looped",
        "n_layers": 4,
        "adaptive_loops": True,
        "include_k_token": False,
        "input_inject": False,
        "inject_scale": 1.0,
        "inject_from_pass": 1,
        "gated": False,
        "step_embed": False,
        "k_encoding": "token",
        "label": "Arm C: Looped 1-Block (adaptive T=k)",
    },
    "arm_d_scratchpad": {
        "arch": "scratchpad",
        "n_layers": 2,
        "adaptive_loops": False,
        "include_k_token": False,
        "input_inject": False,
        "inject_scale": 1.0,
        "inject_from_pass": 1,
        "gated": False,
        "step_embed": False,
        "k_encoding": "token",
        "label": "Arm D: Scratchpad CoT (tokens out loud)",
    },
    "arm_e_unrolled_extrap": {
        "arch": "unrolled",
        "n_layers": 8,
        "adaptive_loops": True,
        "include_k_token": False,
        "input_inject": False,
        "inject_scale": 1.0,
        "inject_from_pass": 1,
        "gated": False,
        "step_embed": False,
        "k_encoding": "token",
        "label": "Arm E: Unrolled 8-Layer Untied Twin (adaptive depth k)",
    },
}


@torch.no_grad()
def evaluate_split(
    model: PointerModel,
    cfg: dict[str, Any],
    split_seed: int,
    eval_n: int = 256,
) -> tuple[dict[int, float], dict[int, int]]:
    """Deterministic accuracy across k=1..8 on a given RNG seed."""
    was_training = model.training
    model.eval()
    eval_rng = np.random.default_rng(split_seed)
    acc_by_k: dict[int, float] = {}
    tokens_by_k: dict[int, int] = {}

    for k in range(1, 9):
        x, all_hops, _ = make_pointer_batch(eval_n, [k], eval_rng, include_k_token=cfg["include_k_token"])
        y_true = all_hops[:, k]
        if cfg["arch"] == "scratchpad":
            emitted = model.decode_scratchpad_cached(x, steps_k=k)
            preds = emitted[:, -1]
            tokens_by_k[k] = k
        else:
            loops = k if cfg["adaptive_loops"] else cfg["n_layers"]
            logits, _, _ = model.forward_latent(x, loops=loops)
            preds = logits.argmax(-1)
            tokens_by_k[k] = 1
        acc_by_k[k] = round(float((preds == y_true).float().mean().item()), 4)

    if was_training:
        model.train()
    return acc_by_k, tokens_by_k


@torch.no_grad()
def evaluate_probe_metrics(
    model: PointerModel,
    cfg: dict[str, Any],
    probe_seed: int = 4242,
    eval_n: int = 128,
) -> tuple[float, float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute pass-by-pass logit-lens readout accuracy and residual norm ratio on 6-hop chains."""
    was_training = model.training
    model.eval()
    probe_rng = np.random.default_rng(probe_seed)
    x6, hops6, _ = make_pointer_batch(eval_n, [6], probe_rng, include_k_token=cfg["include_k_token"])
    pass_rows: list[dict[str, Any]] = []
    sample_traces: list[dict[str, Any]] = []

    if cfg["arch"] == "scratchpad":
        emitted = model.decode_scratchpad_cached(x6, steps_k=6)
        for t in range(1, 7):
            hop_acc = float((emitted[:, t - 1] == hops6[:, t]).float().mean().item())
            pass_rows.append({"iter": t, "true_hop_acc": round(hop_acc, 4), "residual_norm": 0.0})
        for i in range(min(4, eval_n)):
            sample_traces.append({
                "example_id": i,
                "start_node": int(hops6[i, 0].item()),
                "true_hops_1_to_6": str(hops6[i, 1:7].tolist()),
                "emitted_cot_tokens": str(emitted[i].tolist()),
                "latent_logit_lens_passes": "n/a (explicit token CoT)",
                "final_correct": bool(emitted[i, -1].item() == hops6[i, 6].item()),
            })
        logit_lens_k4 = 1.0
        norm_ratio = 1.0
    else:
        probe_loops = 6 if cfg["adaptive_loops"] else cfg["n_layers"]
        if cfg["arch"] == "unrolled":
            probe_loops = min(probe_loops, cfg["n_layers"])
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
        for i in range(min(4, eval_n)):
            decoded_passes = [int(pass_logits[t][i].argmax(-1).item()) for t in range(probe_loops)]
            sample_traces.append({
                "example_id": i,
                "start_node": int(hops6[i, 0].item()),
                "true_hops_1_to_6": str(hops6[i, 1:7].tolist()),
                "emitted_cot_tokens": f"[{decoded_passes[-1]}] (1 final token only; 0 CoT tokens)",
                "latent_logit_lens_passes": str(decoded_passes),
                "final_correct": bool(decoded_passes[-1] == int(hops6[i, 6].item())),
            })

    if was_training:
        model.train()
    return logit_lens_k4, norm_ratio, pass_rows, sample_traces


@torch.no_grad()
def evaluate_noise_robustness(
    model: PointerModel,
    cfg: dict[str, Any],
    eval_n: int = 128,
) -> list[dict[str, float]]:
    """Entry-state Gaussian perturbation sweep (carry vs re-read loops, Lens 5 of arXiv:2610.18127)."""
    if cfg["arch"] == "scratchpad":
        return []
    model.eval()
    rows: list[dict[str, float]] = []
    for alpha in NOISE_ALPHAS:
        rng = np.random.default_rng(5150)
        ngen = torch.Generator().manual_seed(5150)
        accs: list[float] = []
        for k in range(5, 9):
            x, hops, _ = make_pointer_batch(eval_n, [k], rng, include_k_token=cfg["include_k_token"])
            loops = k if cfg["adaptive_loops"] else cfg["n_layers"]
            logits, _, _ = model.forward_latent(x, loops=loops, entry_noise_alpha=alpha, noise_gen=ngen)
            accs.append(float((logits.argmax(-1) == hops[:, k]).float().mean().item()))
        rows.append({"alpha": alpha, "extrap_acc": round(float(np.mean(accs)), 4)})
    return rows


@torch.no_grad()
def benchmark_decode_latency_ms(model: PointerModel, cfg: dict[str, Any], k: int = 8, n_queries: int = 64) -> dict[str, float]:
    """Measure single-query CPU decode latency at hop k (cached vs uncached for scratchpad)."""
    model.eval()
    rng = np.random.default_rng(8088)
    xs, _, _ = make_pointer_batch(n_queries, [k], rng, include_k_token=cfg["include_k_token"])

    # Warmup
    for i in range(4):
        xi = xs[i : i + 1]
        if cfg["arch"] == "scratchpad":
            model.decode_scratchpad_cached(xi, steps_k=k)
        else:
            loops = k if cfg["adaptive_loops"] else cfg["n_layers"]
            model.forward_latent(xi, loops=loops)

    t0 = time.perf_counter()
    for i in range(n_queries):
        xi = xs[i : i + 1]
        if cfg["arch"] == "scratchpad":
            model.decode_scratchpad_cached(xi, steps_k=k)
        else:
            loops = k if cfg["adaptive_loops"] else cfg["n_layers"]
            model.forward_latent(xi, loops=loops)
    cached_ms = round(((time.perf_counter() - t0) * 1000.0) / n_queries, 3)

    if cfg["arch"] == "scratchpad":
        t1 = time.perf_counter()
        for i in range(n_queries):
            cur = xs[i : i + 1]
            for _ in range(k):
                nxt = model.forward_scratchpad(cur)[:, -1, :].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
        uncached_ms = round(((time.perf_counter() - t1) * 1000.0) / n_queries, 3)
    else:
        uncached_ms = cached_ms

    return {"latency_ms_k8": cached_ms, "latency_uncached_ms_k8": uncached_ms}


def build_model_from_cfg(cfg: dict[str, Any]) -> PointerModel:
    return PointerModel(
        arch=cfg["arch"],
        n_layers=int(cfg["n_layers"]),
        d_model=int(cfg.get("d_model", 64)),
        n_heads=int(cfg.get("n_heads", 4)),
        input_inject=bool(cfg.get("input_inject", False)),
        inject_scale=float(cfg.get("inject_scale", 1.0)),
        inject_from_pass=int(cfg.get("inject_from_pass", 1)),
        gated=bool(cfg.get("gated", False)),
        step_embed=bool(cfg.get("step_embed", False)),
        k_encoding=str(cfg.get("k_encoding", "token")),
    )


def train_and_evaluate_one_seed(
    cfg: dict[str, Any],
    seed: int,
    steps: int = 800,
    batch_size: int = 128,
    lr: float = 3e-3,
    eval_n: int = 256,
    log_every: int = 10,
    eval_every: int = 50,
    wandb_run: Any = None,
) -> tuple[PointerModel, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    model = build_model_from_cfg(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    train_max_k = int(cfg.get("train_max_k", 4))
    train_ks = list(range(1, train_max_k + 1))

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if cfg["arch"] == "scratchpad":
            k = int(rng.integers(1, train_max_k + 1))
            x, ys = make_scratchpad_batch(batch_size, k, rng)
            logits = model.forward_scratchpad(x)[:, N_NODES + 1 :, :]
            loss = F.cross_entropy(logits.reshape(-1, N_NODES), ys.reshape(-1))
            with torch.no_grad():
                step_acc = float((logits[:, -1, :].argmax(-1) == ys[:, -1]).float().mean().item())
        elif cfg["adaptive_loops"]:
            k = int(rng.integers(1, train_max_k + 1))
            x, all_hops, _ = make_pointer_batch(batch_size, [k], rng, include_k_token=False)
            logits, _, _ = model.forward_latent(x, loops=k)
            loss = F.cross_entropy(logits, all_hops[:, k])
            with torch.no_grad():
                step_acc = float((logits.argmax(-1) == all_hops[:, k]).float().mean().item())
        else:
            x, all_hops, ks = make_pointer_batch(batch_size, train_ks, rng, include_k_token=True)
            targets = all_hops[torch.arange(batch_size), ks]
            logits, _, _ = model.forward_latent(x, loops=cfg["n_layers"])
            loss = F.cross_entropy(logits, targets)
            with torch.no_grad():
                step_acc = float((logits.argmax(-1) == targets).float().mean().item())

        opt.zero_grad()
        loss.backward()
        grad_sq = sum(float(p.grad.detach().pow(2).sum().item()) for p in model.parameters() if p.grad is not None)
        grad_norm = round(math.sqrt(grad_sq), 4)
        opt.step()

        need_log = step == 1 or (log_every > 0 and step % log_every == 0) or step == steps
        need_eval = step == 1 or (eval_every > 0 and step % eval_every == 0) or step == steps
        if need_log or need_eval:
            payload: dict[str, Any] = {
                "train/step": step,
                "train/loss": round(float(loss.item()), 5),
                "train/acc": round(step_acc, 4),
                "train/grad_norm": grad_norm,
            }
            if need_eval:
                saved_torch_rng = torch.get_rng_state()
                prog_acc, _ = evaluate_split(model, cfg, split_seed=2027, eval_n=96)
                prog_ll, prog_nr, _, _ = evaluate_probe_metrics(model, cfg, probe_seed=4243, eval_n=96)
                torch.set_rng_state(saved_torch_rng)
                payload["train_eval/in_dist_acc"] = round(float(np.mean([prog_acc[hk] for hk in (1, 2, 3, 4)])), 4)
                payload["train_eval/extrap_acc"] = round(float(np.mean([prog_acc[hk] for hk in (5, 6, 7, 8)])), 4)
                payload["train_eval/logit_lens_hop_acc"] = prog_ll
                payload["train_eval/norm_ratio"] = prog_nr
                for hk in range(1, 9):
                    payload[f"train_eval/acc_k{hk}"] = prog_acc[hk]
            if wandb_run is not None:
                wandb_run.log(payload)

    train_wall_s = time.perf_counter() - t0

    # Primary held-out evaluation across k=1..8 (seed=2026)
    acc_by_k, tokens_by_k = evaluate_split(model, cfg, split_seed=2026, eval_n=eval_n)
    logit_lens_k4, norm_ratio, pass_rows, sample_traces = evaluate_probe_metrics(
        model, cfg, probe_seed=4242, eval_n=eval_n
    )
    in_dist_acc = round(float(np.mean([acc_by_k[k] for k in (1, 2, 3, 4)])), 4)
    extrap_acc = round(float(np.mean([acc_by_k[k] for k in (5, 6, 7, 8)])), 4)

    # Secondary validation split (seed=9090) logged alongside seed=2026 across runs
    sec_acc_by_k, _ = evaluate_split(model, cfg, split_seed=9090, eval_n=eval_n)
    sec_extrap_acc = round(float(np.mean([sec_acc_by_k[k] for k in (5, 6, 7, 8)])), 4)

    # Cost accounting & latency + noise robustness
    cost_k8 = estimate_inference_cost(cfg, k=8)
    lat_k8 = benchmark_decode_latency_ms(model, cfg, k=8, n_queries=64)
    noise_rows = evaluate_noise_robustness(model, cfg, eval_n=128)
    noise_map = {r["alpha"]: r["extrap_acc"] for r in noise_rows}

    metrics: dict[str, Any] = {
        "model/param_count": param_count,
        "train/wall_seconds": round(train_wall_s, 3),
        "eval/in_dist_acc": in_dist_acc,
        "eval/extrap_acc": extrap_acc,
        "eval/sec_split_extrap_acc": sec_extrap_acc,
        "eval/holdout_extrap_acc": sec_extrap_acc,  # retained alias for v1 Workspace compatibility
        "eval/emitted_tokens_k8": tokens_by_k[8],
        "cost/block_applications_k8": cost_k8["block_applications"],
        "cost/token_block_visits_k8": cost_k8["token_block_visits"],
        "cost/flops_est_k8": cost_k8["flops_est"],
        "cost/emitted_tokens_k8": cost_k8["emitted_tokens"],
        "cost/latency_ms_k8": lat_k8["latency_ms_k8"],
        "cost/latency_uncached_ms_k8": lat_k8["latency_uncached_ms_k8"],
        "monitor/text_cot_legible": 1.0 if cfg["arch"] == "scratchpad" else 0.0,
        "monitor/logit_lens_hop_acc": logit_lens_k4,
        "monitor/norm_ratio_last_first": norm_ratio,
        "robustness/acc_noise_a1": noise_map.get(1.0),
        "robustness/acc_noise_a2": noise_map.get(2.0),
        "robustness/acc_noise_a4": noise_map.get(4.0),
        "pass_rows": pass_rows,
        "noise_rows": noise_rows,
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
                emitted_toks = model.decode_scratchpad_cached(x_t, steps_k=hop_k)
                pred = int(emitted_toks[0, -1].item())
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


def run_eval_only_holdout(args: argparse.Namespace) -> None:
    """Evaluate saved checkpoints once on a fresh never-inspected holdout seed."""
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    arm_key = args.arm_tag or args.arm
    results: list[dict[str, Any]] = []

    for seed in seeds:
        ckpt_file: Path | None = None
        if args.checkpoint_dir:
            ckpt_file = Path(args.checkpoint_dir) / f"{arm_key}-s{seed}.pt"
        elif not args.dry_run:
            import wandb

            api = wandb.Api()
            art_ref = f"{args.entity}/{args.project}/ckpt-{arm_key}-s{seed}:latest"
            art = api.artifact(art_ref)
            dl_dir = Path(art.download())
            ckpt_file = dl_dir / f"{arm_key}-s{seed}.pt"
        if ckpt_file is None or not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {arm_key} seed {seed}: {ckpt_file}")

        blob = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        cfg = blob["cfg"]
        model = build_model_from_cfg(cfg)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        acc_by_k, tokens_by_k = evaluate_split(model, cfg, split_seed=args.holdout_seed, eval_n=512)
        in_dist = round(float(np.mean([acc_by_k[k] for k in (1, 2, 3, 4)])), 4)
        extrap = round(float(np.mean([acc_by_k[k] for k in (5, 6, 7, 8)])), 4)
        param_count = sum(p.numel() for p in model.parameters())

        run_id = f"dry-holdout-{arm_key}-s{seed}"
        run_url = None
        if not args.dry_run:
            import wandb

            run = wandb.init(
                entity=args.entity,
                project=args.project,
                group=args.group,
                job_type="final_holdout",
                name=f"holdout-{arm_key}-s{seed}",
                tags=[arm_key, cfg["arch"], args.group, args.study_version, "final_holdout"],
                config={
                    "arm": arm_key,
                    "arm_label": cfg.get("label", arm_key),
                    "study_version": args.study_version,
                    "question": "final_holdout",
                    "holdout_seed": args.holdout_seed,
                    "eval_n_per_hop": 512,
                    "seed": seed,
                    **cfg,
                },
                settings=wandb.Settings(save_code=True),
            )
            run.define_metric("curve_hop/k")
            run.define_metric("curve_hop/*", step_metric="curve_hop/k")
            for k in range(1, 9):
                run.log({
                    "curve_hop/k": k,
                    "curve_hop/accuracy": acc_by_k[k],
                    "curve_hop/emitted_tokens": tokens_by_k[k],
                })
                run.summary[f"holdout/acc_k{k}"] = acc_by_k[k]
            run.summary["holdout/in_dist_acc"] = in_dist
            run.summary["holdout/extrap_acc"] = extrap
            run.summary["model/param_count"] = param_count
            run_id = run.id
            run_url = run.url
            run.finish()

        results.append({
            "seed": seed,
            "run_id": run_id,
            "run_url": run_url,
            "holdout_in_dist_acc": in_dist,
            "holdout_extrap_acc": extrap,
            "acc_by_k": acc_by_k,
        })

    vals = [r["holdout_extrap_acc"] for r in results]
    print(json.dumps({
        "mode": "eval_only_holdout",
        "arm": arm_key,
        "holdout_seed": args.holdout_seed,
        "seeds": seeds,
        "mean_holdout_extrap_acc": round(float(np.mean(vals)), 4),
        "std_holdout_extrap_acc": round(float(np.std(vals)), 4),
        "min_holdout_extrap_acc": round(float(np.min(vals)), 4),
        "max_holdout_extrap_acc": round(float(np.max(vals)), 4),
        "runs": results,
    }, indent=2))


def run_self_test() -> None:
    """Offline verification of KV-cached scratchpad parity, new arms, flags, and holdout checkpoint reload."""
    import tempfile

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 1. Verify exact parity between KV-cached and uncached causal scratchpad decoding
    sp_model = PointerModel(arch="scratchpad", n_layers=2, d_model=64, n_heads=4)
    sp_model.eval()
    x_test, _, _ = make_pointer_batch(16, [8], rng, include_k_token=False)
    cached_toks = sp_model.decode_scratchpad_cached(x_test, steps_k=8)
    cur = x_test
    for _ in range(8):
        nxt = sp_model.forward_scratchpad(cur)[:, -1, :].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    uncached_toks = cur[:, N_NODES + 2 :]
    assert torch.equal(cached_toks, uncached_toks), "Cached scratchpad tokens diverged from uncached decode!"

    # Also verify step-by-step logit equivalence on prefill + 1 decode step
    x_step0 = sp_model.tok_emb(x_test) + sp_model.pos_emb(torch.arange(x_test.size(1)))[None, :]
    caches = []
    for b in sp_model.blocks:
        x_step0, kv = b.forward_cached(x_step0, None)
        caches.append(kv)
    logits_cached_0 = sp_model.head(sp_model.ln_f(x_step0[:, -1, :]))
    logits_full_0 = sp_model.forward_scratchpad(x_test)[:, -1, :]
    max_diff = float((logits_cached_0 - logits_full_0).abs().max().item())
    assert max_diff < 1e-5, f"Prefill logit mismatch: {max_diff}"

    # 2. Verify unrolled, inject_from_pass=2, scalar k-encoding, train_max_k=8, and eval_only reload
    with tempfile.TemporaryDirectory() as tmp:
        for cfg_test in [
            {"arch": "unrolled", "n_layers": 8, "adaptive_loops": True, "include_k_token": False, "train_max_k": 4},
            {"arch": "looped", "n_layers": 4, "adaptive_loops": True, "include_k_token": False, "input_inject": True, "inject_scale": 1.0, "inject_from_pass": 2, "train_max_k": 4},
            {"arch": "standard", "n_layers": 4, "adaptive_loops": False, "include_k_token": True, "k_encoding": "scalar", "train_max_k": 8},
        ]:
            m, met, _ = train_and_evaluate_one_seed(cfg_test, seed=0, steps=10, eval_n=16)
            assert "cost/flops_est_k8" in met and met["cost/flops_est_k8"] > 0
            ckpt_p = Path(tmp) / "test_arm-s0.pt"
            torch.save({"cfg": cfg_test, "state_dict": m.state_dict()}, ckpt_p)

        ns = argparse.Namespace(
            arm="custom",
            arm_tag="test_arm",
            seeds="0",
            checkpoint_dir=tmp,
            dry_run=True,
            entity="wandb",
            project="looped-transformer-study",
            group="final-holdout-v2",
            study_version="v2",
            holdout_seed=31415,
        )
        run_eval_only_holdout(ns)

    print("SELF_TEST_OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true", help="Run offline unit checks and exit")
    p.add_argument("--arm", choices=[*ARM_CONFIGS.keys(), "custom"], default="arm_c_looped_extrap")
    p.add_argument("--arch", choices=["standard", "unrolled", "looped", "scratchpad"], default="looped")
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--adaptive-loops", action="store_true")
    p.add_argument("--input-inject", action="store_true")
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--inject-from-pass", type=int, default=1, help="1-indexed pass at which x0 injection starts (2 skips pass 1)")
    p.add_argument("--gated", action="store_true")
    p.add_argument("--step-embed", action="store_true")
    p.add_argument("--k-encoding", choices=["token", "scalar"], default="token")
    p.add_argument("--train-max-k", type=int, default=4, help="Max hop count k sampled during training (default 4; 8 for full capacity ceiling)")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--arm-tag", type=str, default=None)
    p.add_argument("--study-version", type=str, default="v2")
    p.add_argument("--question", type=str, default="general")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--entity", type=str, default="wandb")
    p.add_argument("--project", type=str, default="looped-transformer-study")
    p.add_argument("--group", type=str, default="four-arms-v1")
    p.add_argument("--run-weave-eval", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--holdout-seed", type=int, default=31415)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.eval_only:
        run_eval_only_holdout(args)
        return

    if args.arm == "custom":
        arm_key = args.arm_tag or "autoresearch_custom"
        default_layers = 8 if args.arch == "unrolled" else (2 if args.arch == "scratchpad" else 4)
        cfg: dict[str, Any] = {
            "arch": args.arch,
            "n_layers": args.n_layers if args.n_layers is not None else default_layers,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "adaptive_loops": args.adaptive_loops,
            "include_k_token": (not args.adaptive_loops) and (args.arch != "scratchpad"),
            "input_inject": args.input_inject,
            "inject_scale": args.inject_scale,
            "inject_from_pass": args.inject_from_pass,
            "gated": args.gated,
            "step_embed": args.step_embed,
            "k_encoding": args.k_encoding,
            "train_max_k": args.train_max_k,
            "label": args.label or arm_key,
        }
    else:
        arm_key = args.arm_tag or args.arm
        cfg = dict(ARM_CONFIGS[args.arm])
        if args.n_layers is not None:
            cfg["n_layers"] = args.n_layers
        cfg["d_model"] = args.d_model
        cfg["n_heads"] = args.n_heads
        cfg["inject_scale"] = args.inject_scale
        cfg["inject_from_pass"] = args.inject_from_pass
        cfg["k_encoding"] = args.k_encoding
        cfg["train_max_k"] = args.train_max_k
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
                name=f"{arm_key}-s{seed}",
                tags=[arm_key, cfg["arch"], args.group, args.study_version, args.question],
                config={
                    "arm": arm_key,
                    "arm_label": cfg["label"],
                    "study_version": args.study_version,
                    "question": args.question,
                    **cfg,
                    "seed": seed,
                    "steps": args.steps,
                    "lr": args.lr,
                    "task": f"pointer_chase_n12_train_k1to{cfg['train_max_k']}_eval_k1to8",
                },
                settings=wandb.Settings(save_code=True),
            )
            run.define_metric("train/step")
            run.define_metric("train/*", step_metric="train/step")
            run.define_metric("train_eval/*", step_metric="train/step")
            run.define_metric("curve_hop/k")
            run.define_metric("curve_hop/*", step_metric="curve_hop/k")
            run.define_metric("curve_pass/iter")
            run.define_metric("curve_pass/*", step_metric="curve_pass/iter")
            run.define_metric("curve_noise/alpha")
            run.define_metric("curve_noise/*", step_metric="curve_noise/alpha")

        model, metrics, sample_traces = train_and_evaluate_one_seed(
            cfg=cfg,
            seed=seed,
            steps=args.steps,
            lr=args.lr,
            log_every=args.log_every,
            eval_every=args.eval_every,
            wandb_run=run,
        )

        if args.checkpoint_dir or args.save_checkpoint:
            ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("/tmp/looped-lab-ckpts")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{arm_key}-s{seed}.pt"
            torch.save({"cfg": cfg, "state_dict": model.state_dict(), "seed": seed, "arm": arm_key}, ckpt_path)
            if run is not None and args.save_checkpoint:
                import wandb

                art = wandb.Artifact(
                    name=f"ckpt-{arm_key}-s{seed}",
                    type="model",
                    metadata={"arm": arm_key, "seed": seed, "study_version": args.study_version},
                )
                art.add_file(str(ckpt_path), name=f"{arm_key}-s{seed}.pt")
                run.log_artifact(art)

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
            for nr in metrics["noise_rows"]:
                run.log({
                    "curve_noise/alpha": nr["alpha"],
                    "curve_noise/extrap_acc": nr["extrap_acc"],
                })
            for key, val in metrics.items():
                if key not in ("pass_rows", "noise_rows", "acc_by_k", "tokens_by_k") and val is not None:
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
            "sec_split_extrap_acc": metrics["eval/sec_split_extrap_acc"],
            "acc_by_k": metrics["acc_by_k"],
            "emitted_tokens_k8": metrics["eval/emitted_tokens_k8"],
            "block_applications_k8": metrics["cost/block_applications_k8"],
            "token_block_visits_k8": metrics["cost/token_block_visits_k8"],
            "flops_est_k8": metrics["cost/flops_est_k8"],
            "latency_ms_k8": metrics["cost/latency_ms_k8"],
            "latency_uncached_ms_k8": metrics["cost/latency_uncached_ms_k8"],
            "logit_lens_hop_acc": metrics["monitor/logit_lens_hop_acc"],
            "norm_ratio_last_first": metrics["monitor/norm_ratio_last_first"],
            "acc_noise_a2": metrics["robustness/acc_noise_a2"],
            "acc_noise_a4": metrics["robustness/acc_noise_a4"],
            "train_wall_seconds": metrics["train/wall_seconds"],
            "weave_eval": weave_summary,
        })

    in_vals = [r["in_dist_acc"] for r in all_seed_results]
    ex_vals = [r["extrap_acc"] for r in all_seed_results]
    sec_vals = [r["sec_split_extrap_acc"] for r in all_seed_results]
    print(json.dumps({
        "arm": arm_key,
        "label": cfg["label"],
        "study_version": args.study_version,
        "question": args.question,
        "seeds": seeds,
        "param_count": all_seed_results[0]["param_count"],
        "mean_in_dist_acc_k1_4": round(float(np.mean(in_vals)), 4),
        "mean_extrap_acc_k5_8": round(float(np.mean(ex_vals)), 4),
        "std_extrap_acc_k5_8": round(float(np.std(ex_vals)), 4),
        "min_extrap_acc_k5_8": round(float(np.min(ex_vals)), 4),
        "max_extrap_acc_k5_8": round(float(np.max(ex_vals)), 4),
        "mean_sec_split_extrap_acc_k5_8": round(float(np.mean(sec_vals)), 4),
        "emitted_tokens_k8": all_seed_results[0]["emitted_tokens_k8"],
        "block_applications_k8": all_seed_results[0]["block_applications_k8"],
        "token_block_visits_k8": all_seed_results[0]["token_block_visits_k8"],
        "flops_est_k8": all_seed_results[0]["flops_est_k8"],
        "mean_latency_ms_k8": round(float(np.mean([r["latency_ms_k8"] for r in all_seed_results])), 3),
        "mean_latency_uncached_ms_k8": round(float(np.mean([r["latency_uncached_ms_k8"] for r in all_seed_results])), 3),
        "runs": all_seed_results,
    }, indent=2))


if __name__ == "__main__":
    main()
