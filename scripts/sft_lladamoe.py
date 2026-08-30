#!/usr/bin/env python3
"""
sft_lladamoe.py — SFT del LLaDA-MoE-7B-A1B-Instruct con las trazas L1
(tool-calling ecologico). Receta del paper LLaDA (Secc 2.3):
  - p0 (prompt) NO se enmascara; r0 (response) se enmascara (t~U[0,1]);
    CE loss solo en tokens enmascarados de la response.
  - 3 epochs, lr lineal 0->2.5e-5 en 50 iters, constante, decay 10% final a 2.5e-6.
  - weight decay 0.1, |EOS| tras pares cortos (control de longitud).
LoRA MANUAL (sin peft): adaptadores en self_attn.{q,k,v,o}_proj + mlp.gate,
rank 64, alpha 128, dropout 0.1; el resto del modelo congelado en bf16.

Modo: DDP world=2 (2x pro6000). Tokenizacion previa a npz para no re-tokenizar.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/beegfs/a474r867/ecoreasoner"
MODEL_DIR = os.path.join(BASE, "models/LLaDA-MoE-7B-A1B-Instruct")
MASK_ID = 156895
EOS_ID = 156892


# ---------------- LoRA manual ----------------
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r, alpha, dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_f, r, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=dtype))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.drop(x) @ self.lora_A @ self.lora_B * self.scale


def add_lora(model, r=64, alpha=128, dropout=0.1, verbose=True):
    """Envuelve q/k/v/o_proj y mlp.gate con LoRA; congela todo lo demas."""
    lora_modules = {}
    for name, mod in model.named_modules():
        if name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate")):
            # solo self_attn.*_proj y mlp.gate (router) — no expert gates
            if ".mlp.gate" not in name and "self_attn" not in name:
                continue
            if "experts" in name:
                continue
            if not isinstance(mod, nn.Linear):
                continue
            lora_modules[name] = mod
    # congelar todo
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in lora_modules.items():
        in_f, out_f = mod.weight.shape[1], mod.weight.shape[0]
        lora = LoRALinear(in_f, out_f, r, alpha, dropout,
                          dtype=mod.weight.dtype).to(mod.weight.device)
        setattr(mod, "lora", lora)
        mod.forward_orig = mod.forward

        def _fwd(x, m=mod, l=lora):
            return m.forward_orig(x) + l(x)
        mod.forward = _fwd
        for p in lora.parameters():
            p.requires_grad_(True)
    if verbose:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[sft] LoRA en {len(lora_modules)} capas | params entrenables: {n_tr}", flush=True)
    return lora_modules


# ---------------- datos ----------------
def tokenize_pairs(pairs_path, tokenizer, max_len=1536):
    """Tokeniza (prompt, response) y guarda machine-readable. Devuelve listas."""
    items = []
    with open(pairs_path, encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            items.append(d)
    print(f"[sft] {len(items)} pares", flush=True)
    return items


def build_batches(items, tokenizer, max_len=1536, batch_size=2, seed=0):
    """Empaqueta pares en secuencias; response queda marcada para enmascarar."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items)).tolist()
    batches = []
    cur = []  # (input_ids, resp_start, resp_len) acumulados
    cur_len = 0
    for i in order:
        d = items[i]
        ptxt = d["prompt"] + "\n" + d["response"]
        ids = tokenizer(ptxt)["input_ids"][:max_len]
        resp = tokenizer(d["response"])["input_ids"]
        resp_start = len(ids) - len(resp)
        if resp_start < 0:
            resp_start = 0
        if len(ids) > max_len:
            # recortar respuesta si excede
            keep = max_len - (len(ids) - len(resp))
            if keep < 16:
                continue
            ids = ids[:max_len]
            resp = resp[:keep]
            resp_start = len(ids) - len(resp)
        if cur and cur_len + len(ids) > max_len:
            batches.append(cur); cur = []; cur_len = 0
        cur.append((ids, resp_start, len(resp)))
        cur_len += len(ids)
        if len(cur) >= batch_size * 8:  # empaquetar de a varios
            batches.append(cur); cur = []; cur_len = 0
    if cur:
        batches.append(cur)
    return batches


# ---------------- SFT step ----------------
def sft_step(model, batch_tokens, device):
    """batch_tokens: (input_ids [B,T], resp_starts [B], resp_lens [B]).
    Enmascara solo la response con t~U[0,1], loss CE en mascaras."""
    ids, rstart, rlen = batch_tokens
    ids = ids.to(device)
    B, T = ids.shape
    # prompt sin mascara + response enmascarada parcialmente (t uniforme por muestra)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    t_vals = torch.rand(B, device=device)
    for b in range(B):
        rs = rstart[b]; rl = rlen[b]
        n_mask = max(1, int(rl * t_vals[b].item()))
        # enmascarar n_mask tokens aleatorios de la response
        perm = torch.randperm(rl, device=device)[:n_mask]
        mask[b, rs:rs + rl][perm] = True
    xm = ids.clone()
    xm[mask] = MASK_ID
    out = model(xm).logits
    # loss CE solo en posiciones enmascaradas
    lg = out[mask]
    tg = ids[mask]
    loss = F.cross_entropy(lg, tg)
    return loss, mask.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(BASE, "data/l1/sft_moe_pairs.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(BASE, "outputs/sft_moe"))
    ap.add_argument("--max-steps", type=int, default=0)  # 0 = todas
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      dtype=torch.bfloat16, local_files_only=True)
    device = "cuda"
    model = model.to(device)
    add_lora(model, r=a.rank)
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sft] modelo cargado ({time.time()-t0:.0f}s), entrenables={n_train}", flush=True)

    items = tokenize_pairs(a.pairs, tok, a.max_len)
    batches = build_batches(items, tok, a.max_len, a.batch)
    print(f"[sft] {len(batches)} batches", flush=True)

    # optimizer AdamW con weight decay sobre LoRA
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    total_epoch_steps = len(batches) * a.epochs
    print(f"[sft] {total_epoch_steps} steps totales (3 epochs), lr={a.lr}", flush=True)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        frac = step / max(1, total_epoch_steps)
        if frac > 0.9:
            return a.lr * (1 - (frac - 0.9) / 0.1)
        return a.lr

    step = 0
    logf = open(os.path.join(a.out, "train.log"), "w")
    for epoch in range(a.epochs):
        for bi, b in enumerate(batches):
            if a.max_steps and step >= a.max_steps:
                break
            B = len(b)
            T = max(len(x[0]) for x in b)
            ids = torch.full((B, T), EOS_ID, dtype=torch.long)
            rstart = []; rlen = []
            for j, (ids_j, rs, rl) in enumerate(b):
                ids[j, :len(ids_j)] = torch.tensor(ids_j, dtype=torch.long)
                rstart.append(rs); rlen.append(rl)
            rstart = torch.tensor(rstart); rlen = torch.tensor(rlen)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, n_mask = sft_step(model, (ids, rstart, rlen), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                msg = f"epoch {epoch} step {step}/{total_epoch_steps} loss {loss.item():.4f} lr {lr_at(step):.2e} [{time.time()-t0:.0f}s]"
                print(f"[sft] {msg}", flush=True)
                logf.write(msg + "\n"); logf.flush()
            if step % 200 == 0:
                ckpt = os.path.join(a.out, f"lora-g{step}")
                os.makedirs(ckpt, exist_ok=True)
                # guardar solo adaptadores + modelo base path
                sd = {}
                for name, mod in model.named_modules():
                    lora = getattr(mod, "lora", None)
                    if lora is not None:
                        sd[name + ".lora_A"] = lora.lora_A.detach().float().cpu()
                        sd[name + ".lora_B"] = lora.lora_B.detach().float().cpu()
                torch.save({"adapters": sd, "base": MODEL_DIR, "step": step},
                           os.path.join(ckpt, "lora.pt"))
    logf.close()
    # checkpoint final
    ckpt = os.path.join(a.out, "lora-final")
    os.makedirs(ckpt, exist_ok=True)
    sd = {}
    for name, mod in model.named_modules():
        lora = getattr(mod, "lora", None)
        if lora is not None:
            sd[name + ".lora_A"] = lora.lora_A.detach().float().cpu()
            sd[name + ".lora_B"] = lora.lora_B.detach().float().cpu()
    torch.save({"adapters": sd, "base": MODEL_DIR, "step": step},
               os.path.join(ckpt, "lora.pt"))
    print(f"[sft] DONE {step} steps -> {a.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()