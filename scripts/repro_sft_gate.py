#!/usr/bin/env python3
"""Repro del assert 'index out of bounds' en el SFT v3 (batch ~1881).

Carga el ultimo ckpt lora-g*, itera los batches desde el paso de crash con
CUDA_LAUNCH_BLOCKING=1 para obtener el stacktrace exacto del modeling.
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/beegfs/a474r867/ecoreasoner"
MODEL_DIR = os.path.join(BASE, "models/LLaDA-MoE-7B-A1B-Instruct")
MASK_ID = 156895
EOS_ID = 156892

from train_hybrid import tokenize_pairs, build_batches, add_lora  # noqa: E402

def load_adapters_repro(model, ckpt_dir):
    sd = torch.load(os.path.join(ckpt_dir, "lora.pt"), map_location="cpu")["adapters"]
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            lora = getattr(mod, "lora", None)
            if lora is None:
                continue
            ka, kb = name + ".lora_A", name + ".lora_B"
            if ka in sd and kb in sd:
                lora.lora_A.copy_(torch.as_tensor(sd[ka]).to(
                    dtype=lora.lora_A.dtype, device=lora.lora_A.device))
                lora.lora_B.copy_(torch.as_tensor(sd[kb]).to(
                    dtype=lora.lora_B.dtype, device=lora.lora_B.device))
                n += 1
    return n

def sft_step(model, batch_tokens, device):
    ids, rstart, rlen = batch_tokens
    ids = ids.to(device)
    B, T = ids.shape
    mask = torch.zeros_like(ids, dtype=torch.bool)
    t_vals = torch.rand(B, device=device)
    for b in range(B):
        rs, rl = rstart[b], rlen[b]
        if rl < 1:
            continue
        n_mask = max(1, int(rl * t_vals[b].item()))
        n_mask = min(n_mask, rl)
        perm = torch.randperm(rl, device=device)[:n_mask]
        mask[b, rs:rs + rl][perm] = True
    xm = ids.clone()
    xm[mask] = MASK_ID
    out = model(xm).logits
    lg = out[mask]
    tg = ids[mask]
    loss = F.cross_entropy(lg, tg)
    return loss, mask.sum().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-bi", type=int, default=1880)
    ap.add_argument("--n-batches", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(BASE, "outputs/sft_moe_v3"))
    a = ap.parse_args()

    import torch.utils.checkpoint as cp
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      dtype=torch.bfloat16, local_files_only=True).to("cuda")
    add_lora(model, r=64)
    model.train()

    # ultimo ckpt
    steps = [int(m.group(1)) for d in os.listdir(a.out)
             if (m := re.search(r"lora-g(\d+)$", d))]
    last = max(steps)
    n = load_adapters_repro(model, os.path.join(a.out, f"lora-g{last}"))
    print(f"[repro] adapters {n} cargados de lora-g{last}", flush=True)

    items = tokenize_pairs(f"{BASE}/data/l1/sft_v3_pairs.jsonl")
    batches = build_batches(items, tok, 1536, 8)
    print(f"[repro] {len(batches)} batches; probando bi={a.start_bi}..{a.start_bi+a.n_batches-1}", flush=True)

    # validacion ESTATICA de la invariante del masking (sin GPU): rs+rl <= T
    bad = 0
    for bi in range(len(batches)):
        b = batches[bi]
        T = max(len(x[0]) for x in b)
        for j, (ids_j, rs, rl) in enumerate(b):
            if rs < 0 or rl < 1 or rs + rl > T or rs + rl > len(ids_j) + 2:
                bad += 1
                if bad < 6:
                    print(f"[repro] INVALIDO bi={bi} j={j} rs={rs} rl={rl} T={T} len_ids={len(ids_j)}", flush=True)
    print(f"[repro] invariante: batches invalidos = {bad}", flush=True)
    if bad:
        print("[repro] FIX INCOMPLETO: hay batches invalidos", flush=True)
        return

    for it, bi in enumerate(range(a.start_bi, min(a.start_bi + a.n_batches, len(batches)))):
        b = batches[bi]
        B = len(b); T = max(len(x[0]) for x in b)
        ids = torch.full((B, T), EOS_ID, dtype=torch.long)
        rstart, rlen = [], []
        for j, (ids_j, rs, rl) in enumerate(b):
            ids[j, :len(ids_j)] = torch.tensor(ids_j, dtype=torch.long)
            rstart.append(rs); rlen.append(rl)
        t0 = time.time()
        loss, n_mask = sft_step(model, (ids, torch.tensor(rstart), torch.tensor(rlen)), "cuda")
        torch.cuda.synchronize()
        if it % 100 == 0:
            print(f"[repro] bi={bi} loss={loss.item():.4f} n_mask={n_mask} ({time.time()-t0:.2f}s)", flush=True)

    print(f"[repro] OK: {a.n_batches} batches SIN crash ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()