#!/usr/bin/env python3
"""sft_mdlm.py — SFT del MdLMMoE (v2) con receta LLaDA Sec 2.3.

A diferencia del pretrain denoising (enmascara todo el doc), aquí:
  - el PROMPT nunca se enmascara ni entra al loss;
  - la RESPONSE se enmascara con t ~ U[0,1] por doc;
  - CE solo sobre posiciones response enmascaradas (incluye el EOS final,
    para que aprenda a terminar).
Init desde cualquier checkpoint-gN del GA (model.pt o --ema).

Datos: npz de sft_pretok.py {ids, lengths, resp_starts, eos_id}.
Ckpts: output/checkpoint-g{step}/{model.pt,ema_model.pt,optimizer.pt}
mismo formato que train_mdlm_moe_v2 → reusa eval (EVAL_EMA=1) y REPL.

Uso:
  python3 sft_mdlm.py --init runs/g10-ep3-s1/checkpoint-g23000 --ema \
      --data_cache data/sft_ids_v1.npz --output runs/sft-chat-v1 \
      --epochs 3 --batch_size 16 --lr 2e-5
"""
import argparse, importlib.util, json, math, os, random, re, signal, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
ARCH = dict(vocab=126080, hidden=768, layers=12, heads=12,
            ff_mult=4, seq_len=768, n_experts=8, k=1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_class():
    path = REPO / "scripts" / "train_mdlm_moe_v2.py"
    argv_bak = sys.argv
    prev = signal.getsignal(signal.SIGUSR1)
    try:
        sys.argv = [path.name, "--data", "dummy", "--output", "/tmp/dummy_sft"]
        spec = importlib.util.spec_from_file_location("train_mdlm_moe_v2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MdLMMoE
    finally:
        sys.argv = argv_bak
        signal.signal(signal.SIGUSR1, prev)


# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--init", required=True, help="dir checkpoint-gN o model.pt")
p.add_argument("--ema", action="store_true", help="init desde ema_model.pt")
p.add_argument("--data_cache", required=True)
p.add_argument("--output", required=True)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--ema_decay", type=float, default=0.999)
p.add_argument("--seq_len", type=int, default=768)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
ARGS = p.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_ID = ARCH["vocab"]
OUT = Path(ARGS.output); OUT.mkdir(parents=True, exist_ok=True)

glob = {"model": None, "opt": None, "ema": None, "step": 0}


def resolve_init(d):
    d = Path(d)
    if d.is_file():
        return d
    name = "ema_model.pt" if ARGS.ema else "model.pt"
    if (d / name).is_file():
        return d / name
    if ARGS.ema and (d / "model.pt").is_file():
        log("ema_model.pt no existe -> uso model.pt")
        return d / "model.pt"
    sys.exit(f"no {name} en {d}")


def load_docs(npz_path):
    d = np.load(npz_path)
    ids, lens, starts = d["ids"], d["lengths"], d["resp_starts"]
    eos = int(d["eos_id"])
    docs, off = [], 0
    for L, s in zip(lens, starts):
        docs.append((ids[off:off+L], int(s)))
        off += L
    log(f"data: {len(docs)} pares, {off/1e6:.1f}M tok, eos={eos}")
    return docs, eos


def collate(batch, eos):
    """batch de (ids, resp_start) -> tensor padded + máscaras."""
    B = len(batch)
    x = torch.full((B, ARGS.seq_len), eos, dtype=torch.long)
    resp_mask = torch.zeros(B, ARGS.seq_len, dtype=torch.bool)  # posiciones response
    keep = []
    for i, (ids, s) in enumerate(batch):
        L = min(len(ids), ARGS.seq_len)
        x[i, :L] = torch.as_tensor(ids[:L], dtype=torch.long)
        resp_mask[i, s:L] = True
        keep.append(L)
    return x, resp_mask, keep


def apply_mask(x, resp_mask, keep, rng):
    """Enmascara posiciones response con t~U[0,1] por doc. Devuelve xm, target_mask."""
    B, T = x.shape
    xm = x.clone()
    target = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        L = keep[i]
        cand = resp_mask[i, :L]
        m = (torch.rand(L) < rng.random()) & cand
        if not m.any() and cand.any():   # al menos 1
            m[cand.nonzero()[rng.randrange(int(cand.sum()))]] = True
        xm[i, :L][m] = MASK_ID
        target[i, :L] = m
    return xm, target


def save_ckpt(step):
    d = OUT / f"checkpoint-g{step}"
    d.mkdir(exist_ok=True)
    torch.save({"model": glob["model"].state_dict()}, d / "model.pt")
    if glob["ema"] is not None:
        torch.save({"ema_model": glob["ema"]}, d / "ema_model.pt")
    torch.save({"optimizer": glob["opt"].state_dict(), "step": step}, d / "optimizer.pt")
    json.dump({"step": step, "args": vars(ARGS)}, open(OUT / "state.json", "w"), indent=1)
    log(f"checkpoint guardado: {d}")


def on_usr1(sig, frm):
    log("SIGUSR1 -> checkpoint + exit")
    save_ckpt(glob["step"])
    sys.exit(0)


def main():
    signal.signal(signal.SIGUSR1, on_usr1)
    rng = random.Random(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    MdLMMoE = load_model_class()
    model = MdLMMoE(**ARCH).to(DEVICE)
    ck = torch.load(resolve_init(ARGS.init), map_location=DEVICE)
    for key in ("model", "ema_model"):
        if isinstance(ck, dict) and key in ck:
            ck = ck[key]
    model.load_state_dict(ck, strict=True)
    log(f"init: {resolve_init(ARGS.init)} ({model.n_params()/1e6:.0f}M)")

    docs, eos = load_docs(ARGS.data_cache)
    opt = torch.optim.AdamW(model.parameters(), lr=ARGS.lr,
                          weight_decay=ARGS.weight_decay, betas=(0.9, 0.98))
    # EMA residente en GPU (copia CPU por step = cuello D2H de 2.7GB)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    glob.update(model=model, opt=opt, ema=ema)

    step = 0
    n = len(docs)
    for ep in range(ARGS.epochs):
        order = list(range(n)); rng.shuffle(order)
        for b0 in range(0, n - ARGS.batch_size + 1, ARGS.batch_size):
            batch = [docs[i] for i in order[b0:b0 + ARGS.batch_size]]
            x, resp_mask, keep = collate(batch, eos)
            xm, target = apply_mask(x, resp_mask, keep, rng)
            x, xm, target = x.to(DEVICE), xm.to(DEVICE), target.to(DEVICE)
            lr = ARGS.lr * min(1.0, (step + 1) / max(1, ARGS.warmup))
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(xm)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[target.reshape(-1)],
                x.reshape(-1)[target.reshape(-1)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1; glob["step"] = step
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    ema[k].mul_(ARGS.ema_decay).add_(
                        sd[k].detach(), alpha=1 - ARGS.ema_decay)
            if step % 20 == 0:
                log(f"ep{ep+1} step {step} loss {loss.item():.4f} lr {lr:.2e}")
            if step % ARGS.save_every == 0:
                save_ckpt(step)
    save_ckpt(step)
    (OUT / "training_complete.flag").write_text("done\n")
    log(f"SFT DONE steps={step}")


if __name__ == "__main__":
    main()
