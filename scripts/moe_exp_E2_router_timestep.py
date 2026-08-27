#!/usr/bin/env python3
"""
EXPERIMENTO E2 — Router MoE condicionado al paso de denoising (timestep).

Frontera de dominio dLLM+MoE: en masked-diffusion, el modelo puede conocer el paso de
denoising (cuántos pasos van / fracción de mascara). El gate MoE top-1 actual usa solo el
token (`gate(x)`). Este experimento añade un embedding del timestep al gate:
    gate(x + t_emb(timestep))
de modo que los expertos se activan SEGÚN el paso de denoising (primeros pasos = expertos
gruesos/reconstrucción; últimos = expertos finos), además del token.

Cambios vs train_mdlm_moe.py:
  1. MoEMLP acepta `t` (timestep por posición) y usa t_emb en el gate.
  2. Block pasa `t` hacia el MoE.
  3. MdLMMoE recibe `timestep` (int 0..seq_len-1 de mascara aplicada) y lo propaga.

NO rompe el entrenamiento actual: es un archivo separado.
"""
import argparse, math, random, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--expert_k", type=int, default=1)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--mask_p", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="train_ids_v3.npy", help="ruta al npy de ids")
    p.add_argument("--timestep_emb_dim", type=int, default=64, help="dim del emb de timestep (sumado al gate)")
    return p.parse_args()

# ARGS se parsea SOLO en __main__ (no al importar), para no romper el import desde el trainer

class MoEMLP(nn.Module):
    """MoE FFN top-k con gate condicionado al TIMESTEP de denoising."""
    def __init__(self, dim, ff, n_experts, k, tdim=64, n_steps=100, shared=True):
        super().__init__()
        self.n, self.k = n_experts, k
        self.gate = nn.Linear(dim+tdim, n_experts, bias=False)
        self.t_emb = nn.Embedding(n_steps+1, tdim)  # timestep 0..n_steps
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_experts)])
        self.shared = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim)) if shared else None
        # ---- métricas aux (para loss aux de balance + monitoreo) ----
        self._gate_probs = None   # probs softmax del último forward (para balance loss)
        self._tokens = 0
        self.register_buffer("_fcount", torch.zeros(n_experts), persistent=False)
    def forward(self, x, t=None):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        if t is None:
            # timestep neutro: emb del índice 0 (consistente con gate[dim+tdim])
            te = self.t_emb(torch.zeros(flat.shape[0], device=flat.device, dtype=torch.long))
        else:
            te = self.t_emb(t.reshape(-1).clamp(0, self.t_emb.num_embeddings-1))
        gate_in = torch.cat([flat, te], dim=-1)
        g = torch.softmax(self.gate(gate_in).float(), dim=-1)
        gv, gi = g.topk(self.k, dim=-1)
        # ---- registrar para balance loss (sin retener grafo de routing discreto) ----
        self._gate_probs = g.detach() if self.training else None
        if self.training:
            N = gi.numel()
            self._fcount.zero_()
            self._fcount.scatter_add_(0, gi.reshape(-1), torch.ones(N, device=gi.device))
            self._tokens = N
        out = torch.zeros_like(flat)
        # E4: experto compartido SIEMPRE activo (aporta base/tool-calling) + top-k especializados
        if self.shared is not None:
            out = self.shared(flat)
        for rank in range(self.k):
            ids = gi[:, rank]; w = gv[:, rank]
            for e in range(self.n):
                sel = (ids == e)
                if sel.any():
                    out[sel] += w[sel, None] * self.experts[e](flat[sel])
        return out.reshape(B, T, D)
    def reset_router_stats(self):
        self._gate_probs = None
        self._tokens = 0
        self._fcount.zero_()
    def balance_loss(self, alpha=0.01):
        """Loss aux de balance de carga (estilo Switch-Transformer).

        L_aux = alpha * N * sum_e f_e * P_e
          f_e = fracción de tokens ruteados a experto e (del routing top-k, detach)
          P_e = media de probs del gate para e (del propio gate, detach)
        Minimizar empuja la distribución de asignación hacia uniforme.
        """
        if self._gate_probs is None or self._tokens == 0:
            return torch.zeros((), device=self.gate.weight.device)
        f = self._fcount.to(self._gate_probs.dtype) / max(self._tokens, 1)
        P = self._gate_probs.mean(0)
        return alpha * self.n * (f * P.detach()).sum()
    def router_stats(self):
        """Métricas de monitoreo tras un forward: (entropía media, fracción top-1 por experto, N efectivo)."""
        if self._gate_probs is None or self._tokens == 0:
            return None
        g = self._gate_probs
        ent = (-g * (g + 1e-12).log()).sum(-1).mean().item()
        f = self._fcount.to(g.dtype) / max(self._tokens, 1)
        eff = torch.exp(-torch.sum(f * torch.log(f + 1e-12))).item()
        return {"entropy": ent, "top1_freq": f.tolist(), "eff_n": eff}

class Block(nn.Module):
    def __init__(self, dim, ff, heads, n_experts, k, tdim=64, n_steps=100, shared=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MoEMLP(dim, ff, n_experts, k, tdim, n_steps, shared)
    def forward(self, x, t=None):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x), t)

class MdLMMoE(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, ff_mult, seq_len, n_experts, k,
                 tdim=64, n_steps=100, shared=True):
        super().__init__()
        self.vocab = vocab
        self.tok_emb = nn.Embedding(vocab + 1, hidden)  # +1 MASK
        self.pos = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, hidden*ff_mult, heads, n_experts, k, tdim, n_steps, shared) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, ids, timestep=None):
        B, T = ids.shape
        h = self.tok_emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b(h, timestep)
        return self.head(self.ln_f(h))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

# ---- smoke test (tiny) ----
if __name__ == "__main__":
    torch.manual_seed(0)
    m = MdLMMoE(vocab=1000, hidden=64, layers=2, heads=4, ff_mult=2, seq_len=32,
                n_experts=4, k=1, tdim=16, n_steps=20)
    ids = torch.randint(0, 1000, (2, 32))
    t = torch.randint(0, 20, (2, 32))           # timestep por posición
    out = m(ids, timestep=t)
    print("E2 model OK. out:", out.shape, "params:", m.n_params())
    out_none = m(ids, timestep=None)
    print("sin timestep OK:", out_none.shape)