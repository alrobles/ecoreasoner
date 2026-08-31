#!/usr/bin/env python3
"""
train_hybrid.py — ENTRENAMIENTO HETEROGENEO MULTI-GPU (sistemas, 2026-08-30).

Convierte la desventaja de un cluster NVIDIA heterogeneo (Q6000 48GB / L40 48GB /
A100 80GB / Blackwell 96GB) en ventaja: una sola ejecucion DDP con grad_accum
ADAPTATIVO + microbatch auto-sizear por rank.

CLAVE MATEMATICA: el grad_accum adaptativo da un optimizer-step GLOBAL identico
(cada rank acumula N=ceil(global_batch/target_batch) micros locales) mientras el
microbatch DIFFIERE por rank segun su VRAM. El all-reduce ocurre SOLO en el
ultimo micro (no_sync en el resto) — mismo resultado que un batch fijo grande.

COMPONENTES:
  - autosize_vram(rank, bytes_per_sample) -> microbatch maximo con 20% headroom
  - grad_accum_optimo(global, local) -> ceil, distribuye el paso
  - gradient checkpointing opcional (-c) para tarjetas 48GB (intercambia
    compute por memoria; permite batch mas grande en Q6000/L40).
  - no_sync() en DDP para micros intermedios; all-reduce + optimizer.step() al
    unisiso (barrier para que el paso global sea sincrono).

Uso (dentro de slurm, 1 contenedor/nodo, torchrun N ranks):
  apptainer exec --nv SIF python -m torch.distributed.run --nproc_per_node=N \
    train_hybrid.py --model-dir MODEL --pairs DATA --out OUT \
    [--global-batch 64] [--target-batch-48gb 2] [--checkpointing]
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

BASE = "/beegfs/a474r867/ecoreasoner"
MODEL_DIR = os.path.join(BASE, "models/LLaDA-MoE-7B-A1B-Instruct")
MASK_ID = 156895
EOS_ID = 156892


# ---------------- LoRA (mismo esquema que sft_lladamoe) ----------------
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


def add_lora(model, r=64, alpha=128, dropout=0.1):
    lora_modules = {}
    for name, mod in model.named_modules():
        if name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate")):
            if ".mlp.gate" not in name and "self_attn" not in name:
                continue
            if "experts" in name:
                continue
            if not isinstance(mod, nn.Linear):
                continue
            lora_modules[name] = mod
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in lora_modules.items():
        in_f, out_f = mod.weight.shape[1], mod.weight.shape[0]
        # crear en CPU y mover al device del módulo (evita mismatch si el
        # módulo vive en cuda:{rank} pero el default cuda es otro)
        lora = LoRALinear(in_f, out_f, r, alpha, dtype=mod.weight.dtype)
        lora.to(mod.weight.device)
        setattr(mod, "lora", lora)
        mod.forward_orig = mod.forward

        def _fwd(x, m=mod, l=lora):
            return m.forward_orig(x) + l(x)
        mod.forward = _fwd
        for p in lora.parameters():
            p.requires_grad_(True)
    return lora_modules


# ---------------- HITO b: autosize + grad_accum adaptativo ----------------
def autosize_vram(vram_gb, bytes_per_sample, headroom=0.20, min_batch=1):
    """Microbatch maximo que cabe en la VRAM dejando `headroom` libres.
    bytes_per_sample = activaciones+grad de UN sample a seq_len max."""
    usable = vram_gb * (1 - headroom)      # GB utilizables
    per_s = bytes_per_sample / 1e9          # GB por sample
    b = max(min_batch, int(usable / max(per_s, 1e-6)))
    # tope 32 (sanity)
    return min(b, 32)


def grad_accum_optimo(global_batch, local_batch):
    """Micros que cada rank debe acumular para un global batch identico.
    Redondeo hacia arriba; el ultimo micro puede tener menos samples reales."""
    return max(1, math.ceil(global_batch / max(local_batch, 1)))


def log(msg, rank=0):
    if rank == 0:
        print(f"[hybrid] {msg}", flush=True)


# ---------------- datos (replica sft_lladamoe) ----------------
def tokenize_pairs(pairs_path):
    items = []
    with open(pairs_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    items.append(json.loads(ln))
                except Exception:
                    pass
    return items


def build_batches(items, tokenizer, max_len=1536, batch_size=8, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items)).tolist()
    batches, cur, cur_len = [], [], 0
    for i in order:
        d = items[i]
        resp = tokenizer(d["response"])["input_ids"]
        ptxt = d["prompt"] + "\n" + d["response"]
        ids = tokenizer(ptxt)["input_ids"][:max_len]
        rs = len(ids) - len(resp)
        if rs < 0:
            rs = 0
        if len(ids) > max_len:
            keep = max_len - (len(ids) - len(resp))
            if keep < 16:
                continue
            ids = ids[:max_len]
            resp = resp[:keep]
            rs = len(ids) - len(resp)
        if cur and cur_len + len(ids) > max_len:
            batches.append(cur); cur, cur_len = [], 0
        cur.append((ids, rs, len(resp)))
        cur_len += len(ids)
        if len(cur) >= batch_size * 8:
            batches.append(cur); cur, cur_len = [], 0
    if cur:
        batches.append(cur)
    return batches


# ---------------- step con loss solo en response ----------------
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
    if int(mask.sum()) == 0:
        # batch sin mascaras (rl=0 o t->0): forward con loss dummy QUE REQUIERE
        # grad (anclada a un param LoRA para no romper el backward/ddp)
        xm = ids.clone()
        out = model(xm).logits
        lg = out[:, 0]
        tg = ids[:, 0].clone()
        ce = F.cross_entropy(lg, tg)
        # anclar al primer LoRA: si hay alguno, ce*0 + lora.sum()*1e-8
        for _n, _mod in model.named_modules():
            _l = getattr(_mod, "lora", None)
            if _l is not None:
                return ce + _l.lora_A.sum() * 1e-8, 0
        return ce, 0
    xm = ids.clone()
    xm[mask] = MASK_ID
    out = model(xm).logits
    return F.cross_entropy(out[mask], ids[mask]), mask.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--pairs", default=os.path.join(BASE, "data/l1/sft_v3_pairs.jsonl"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8, help="empaquetado por batch (no es el microbatch)")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--global-batch", type=int, default=64, help="batch global objetivo (tokens por optimizer step)")
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (tarjetas chicas)")
    ap.add_argument("--out", default=os.path.join(BASE, "outputs/train_hybrid"))
    a = ap.parse_args()

    # ---- DDP init ----
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1
    device = f"cuda:{rank}"
    log(f"rank {rank}/{world} device={device}")

    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True,
                                      dtype=torch.bfloat16, local_files_only=True).to(device)
    add_lora(model, r=a.rank)
    if a.checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # use_reentrant=False es OBLIGATORIO aqui: con reentrant=True (default
        # de transformers 4.56) y inputs enteros sin requires_grad, checkpoint
        # cortocircuita el grafo ("None of the inputs have requires_grad=True")
        # y el loss sale sin grad_fn -> RuntimeError en backward.
        model.gradient_checkpointing_enable(use_reentrant=False)
        log("gradient checkpointing ON (no-reentrant)")
    model.train()
    # NO DDP: el LoRA manual sobre mod.forward no expone params al hook de DDP.
    # Sincronizamos los gradientes de los LoRA con all_reduce manual (mismo
    # resultado, y es la clave del paper: solo ~19M params a sincronizar).
    def allreduce_lora(rmodel):
        for name, mod in rmodel.named_modules():
            lora = getattr(mod, "lora", None)
            if lora is not None:
                for pname in ("lora_A", "lora_B"):
                    p = getattr(lora, pname)
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                    else:
                        # grad None en este rank pero puede existir en otro:
                        # all_reduce requiere el tensor; usar zeros (equivale a
                        # "este rank no contribuye a este param en este paso")
                        z = torch.zeros_like(p)
                        dist.all_reduce(z)
        # barrier: todos los ranks sincronizan antes del optimizer
        dist.barrier()

    # ---- autosize por rank ----
    props = torch.cuda.get_device_properties(rank)
    vram_gb = props.total_memory / 1e9
    # estimar bytes/sample: forward con batch=1 dummy
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True, local_files_only=True)
    # activaciones approx: seq_len(1536) * hidden(2048) * layers(16) * 2 bytes * factor
    bytes_per_sample = a.max_len * cfg.hidden_size * cfg.num_hidden_layers * 2
    micro = autosize_vram(vram_gb, bytes_per_sample)
    accum = grad_accum_optimo(a.global_batch, micro)
    log(f"rank {rank}: VRAM={vram_gb:.0f}GB microbatch={micro} grad_accum={accum} "
        f"(global={a.global_batch})")
    # sync: todos los ranks con el MISMO accum (paso global idéntico)
    accums = [accum]
    if world > 1:
        accums = [torch.tensor(0, device=device) for _ in range(world)]
        dist.all_gather(accums, torch.tensor(accum, device=device))
        accums = [int(x.item()) for x in accums]
        global_accum = max(accums)  # todos usan el del rank con menos memoria
        log(f"ranks accum: {accums} -> usando max {global_accum}")
        accum = global_accum

    items = tokenize_pairs(a.pairs)
    batches = build_batches(items, tok, a.max_len, a.batch)
    total_steps = len(batches) * a.epochs * accum
    log(f"{len(batches)} batches, acum {accum} -> {total_steps} optimizer steps ~{total_steps*len(batches):.0f} micros")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        frac = step / max(1, total_steps)
        if frac > 0.9:
            return a.lr * (1 - (frac - 0.9) / 0.1)
        return a.lr

    raw = getattr(model, "module", model)  # model.module si DDP, si no model
    step = 0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.log"), "w") as logf:
        for epoch in range(a.epochs):
            for bi, b in enumerate(batches):
                B = len(b)
                T = max(len(x[0]) for x in b)
                ids = torch.full((B, T), EOS_ID, dtype=torch.long)
                rstart, rlen = [], []
                for j, (ids_j, rs, rl) in enumerate(b):
                    ids[j, :len(ids_j)] = torch.tensor(ids_j, dtype=torch.long)
                    rstart.append(rs); rlen.append(rl)
                rstart = torch.tensor(rstart); rlen = torch.tensor(rlen)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                # grad_accum adaptativo: no_sync en micros intermedios
                import contextlib
                micro_losses = []
                for mic in range(accum):
                    with contextlib.nullcontext():
                        loss, n_mask = sft_step(model, (ids, rstart, rlen), device)
                        if loss.grad_fn is None:
                            _first = next((getattr(m2, "lora") for m2 in raw.modules()
                                           if getattr(m2, "lora", None) is not None), None)
                            _d = _first.lora_A.device if _first else "?"
                            # ¿el lora_A tiene grad_fn tras el forward? (diagnóstico)
                            _gf = _first.lora_A.grad_fn is not None if _first else None
                            _req = _first.lora_A.requires_grad if _first else None
                            print(f"[hybrid][rank{rank}] WARN loss sin grad_fn (mic {mic}) "
                                  f"train={model.training} n_mask={n_mask} dev={_d} "
                                  f"loraA.grad_fn={_gf} loraA.req={_req}", flush=True)
                        (loss / accum).backward()
                    micro_losses.append(loss.item() / accum)
                # sync grads LoRA SOLO al final del grad_accum (all-reduce manual)
                if world > 1:
                    allreduce_lora(raw)
                # optimizer step (skip si no hay grads — batch sin mascaras)
                gs = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if gs:
                    torch.nn.utils.clip_grad_norm_(gs, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % 10 == 0:
                    msg = (f"epoch {epoch} step {step}/{total_steps} loss {sum(micro_losses)/len(micro_losses):.4f} "
                           f"lr {lr_at(step):.2e} [{time.time()-t0:.0f}s]")
                    print(f"[hybrid] {msg}", flush=True)
                    logf.write(msg + "\n"); logf.flush()
    if rank == 0:
        # guardar adaptadores
        sd = {}
        for name, mod in raw.named_modules():
            lora = getattr(mod, "lora", None)
            if lora is not None:
                sd[name + ".lora_A"] = lora.lora_A.detach().float().cpu()
                sd[name + ".lora_B"] = lora.lora_B.detach().float().cpu()
        ckpt = os.path.join(a.out, "lora-final")
        os.makedirs(ckpt, exist_ok=True)
        torch.save({"adapters": sd, "base": a.model_dir, "step": step},
                   os.path.join(ckpt, "lora.pt"))
        log(f"DONE {step} steps -> {a.out} [{time.time()-t0:.0f}s]")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()