#!/usr/bin/env python3
"""train_logicdiff_head.py — cabeza de roles lógicos sobre hidden states congelados.

LogicDiff Fase A (docs/designs/dLLM-rescue-plan.md): un MLP pequeño (~1-5M)
clasifica cada token ENMASCARADO en {FILLER, PREMISE, CONNECTIVE, DERIVED,
CONCLUSION} a partir del hidden state del dLLM. El backbone (checkpoint
v2/v3) queda CONGELADO — solo se entrena la cabeza, así una eventual mejora
en L3 no puede atribuirse a reentrenar el modelo.

Datos: role_train.pt de build_logicdiff_dataset.py
  {ids [N,T] long, roles [N,T] long (-100 = pad), lengths [N]}

El masking de entrenamiento mezcla dos patrones para aproximar el uso real
(en inferencia la cabeza clasifica regiones candidatas completamente
enmascaradas):
  - 50%: máscara aleatoria uniforme t ~ U(--mask-lo, --mask-hi)
  - 50%: bloque contiguo de fracción t (la "región candidata")

Uso:
  python scripts/train_logicdiff_head.py \
      --ckpt runs/f0-span-v3-role/checkpoint-g10000/model.pt \
      --config harness/configs/f0-span-v3-role.yaml \
      --data data/logicdiff/role_train.pt \
      --out runs/logicdiff-head-v3role
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ---- import del modelo v2 (mismo truco que suite_smoke_v2) ----
_trainer_path = Path(__file__).resolve().parents[1] / "scripts" / "train_mdlm_moe_v2.py"
import importlib.util  # noqa: E402

_argv = __import__("sys").argv
import sys
sys.argv = [_trainer_path.name, "--data", "dummy", "--output", "/tmp/dummy_head"]
try:
    _spec = importlib.util.spec_from_file_location("train_mdlm_moe_v2", _trainer_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MdLMMoE = _mod.MdLMMoE
finally:
    sys.argv = _argv

ROLE_NAMES = ["FILLER", "PREMISE", "CONNECTIVE", "DERIVED", "CONCLUSION"]
N_ROLES = len(ROLE_NAMES)


class RoleHead(nn.Module):
    def __init__(self, hidden, head_hidden=1536, n_roles=N_ROLES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, n_roles),
        )
    def forward(self, h):
        return self.net(h)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="role_train.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-hidden", type=int, default=1536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--mask-lo", type=float, default=0.05)
    p.add_argument("--mask-hi", type=float, default=0.95)
    p.add_argument("--block-mask-prob", type=float, default=0.5)
    p.add_argument("--unmasked-aux", type=float, default=0.1,
                   help="fracción de posiciones NO enmascaradas añadidas a la "
                        "loss (peso 0.2) para robustez; 0 = solo masked")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(args, mcfg):
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    model = MdLMMoE(
        vocab=mcfg["vocab"], hidden=mcfg["hidden"], layers=mcfg["layers"],
        heads=mcfg["heads"], ff_mult=mcfg["ff_mult"], seq_len=mcfg["seq_len"],
        n_experts=mcfg["n_experts"], k=mcfg["k"],
        use_rope=mcfg.get("use_rope", False),
        weight_tying=mcfg.get("weight_tying", False),
    ).to(dev)
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError:
        ck = {k[len("module."):]: v for k, v in ck.items()}
        model.load_state_dict(ck, strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model, dev


def sample_masked_positions(n_valid, frac, block, rng, device):
    """Índices de posiciones a enmascarar entre [0, n_valid)."""
    n = max(1, int(frac * n_valid))
    n = min(n, n_valid)
    if block:
        start = rng.randint(0, max(0, n_valid - n))
        return torch.arange(start, start + n, device=device)
    return torch.randperm(n_valid, device=device)[:n]


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")
    def log(m):
        s = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    torch.manual_seed(a.seed); random.seed(a.seed)
    rng = random.Random(a.seed)

    cfg = yaml.safe_load(Path(a.config).read_text())
    mcfg = cfg["model"]
    model, dev = load_model(a, mcfg)
    mask_id = mcfg["vocab"]
    log(f"modelo cargado {a.ckpt} (hidden={mcfg['hidden']}, mask_id={mask_id})")

    data = torch.load(a.data, weights_only=False)
    ids_all, roles_all, lens = data["ids"], data["roles"], data["lengths"]
    N, T = ids_all.shape
    log(f"dataset: {N} docs x {T} tok; stats={data.get('stats')}")

    # split train/val por documento
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed))
    n_val = max(1, int(N * a.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    head = RoleHead(mcfg["hidden"], a.head_hidden).to(dev)
    n_head = sum(p.numel() for p in head.parameters())
    log(f"head: {n_head/1e6:.2f}M params (hidden {mcfg['hidden']} -> "
        f"{a.head_hidden} -> {N_ROLES})")

    # pesos de clase (inverso a frecuencia observada, clamp para no explotar)
    stats = data.get("stats") or {}
    counts = torch.tensor([max(1, stats.get(str(i), stats.get(i, 1)))
                           for i in range(N_ROLES)], dtype=torch.float32)
    cw = (counts.sum() / (N_ROLES * counts)).clamp(max=10.0).to(dev)
    log(f"class weights: { {ROLE_NAMES[i]: round(float(cw[i]),3) for i in range(N_ROLES)} }")

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)

    def set_lr(step):
        if step < a.warmup:
            lr = a.lr * (step + 1) / a.warmup
        else:
            prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr

    def run_batch(bidx, train=True):
        ids = ids_all[bidx].to(dev)            # [B,T]
        roles = roles_all[bidx].to(dev)        # [B,T]
        valid = roles != -100                  # [B,T]
        B = ids.size(0)
        xm = ids.clone()
        target_mask = torch.zeros_like(valid)
        for b in range(B):
            nv = int(valid[b].sum())
            frac = rng.uniform(a.mask_lo, a.mask_hi)
            block = rng.random() < a.block_mask_prob
            pos = sample_masked_positions(nv, frac, block, rng, dev)
            xm[b, pos] = mask_id
            target_mask[b, pos] = True
        # auxiliar: también clasificar una fracción de posiciones visibles
        if a.unmasked_aux > 0:
            for b in range(B):
                nv = int(valid[b].sum())
                free = (~target_mask[b]) & valid[b]
                nfree = int(free.sum())
                k = min(int(a.unmasked_aux * nv), nfree)
                if k > 0:
                    fidx = free.nonzero(as_tuple=False).squeeze(-1)
                    sel = fidx[torch.randperm(nfree, device=dev)[:k]]
                    target_mask[b, sel] = True
        with torch.no_grad():
            _, hidden = model(xm, output_hidden_states=True)
        logits = head(hidden)                  # [B,T,R]
        sel = target_mask & valid
        masked_sel = sel & (xm == mask_id)     # objetivo real: tokens enmascarados
        vis_sel = sel & ~masked_sel            # auxiliar (peso 0.2)
        ce = F.cross_entropy(logits[masked_sel], roles[masked_sel], weight=cw)
        if vis_sel.any():
            ce = ce + 0.2 * F.cross_entropy(logits[vis_sel], roles[vis_sel],
                                           weight=cw)
        acc = (logits[masked_sel].argmax(-1) == roles[masked_sel]).float().mean()
        return ce, acc

    # ---- train ----
    t0 = time.time()
    ntr = len(train_idx)
    for step in range(a.steps):
        bidx = train_idx[torch.randint(0, ntr, (a.batch_size,),
                                       generator=torch.Generator().manual_seed(a.seed + step))]
        loss, acc = run_batch(bidx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        set_lr(step)
        opt.step()
        if step % 50 == 0:
            log(f"step {step} loss {loss.item():.4f} acc {acc.item():.3f}")

    # ---- eval holdout ----
    head.eval()
    conf = torch.zeros(N_ROLES, N_ROLES, dtype=torch.long)
    tot_loss, tot_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_idx), a.batch_size):
            bidx = val_idx[i:i + a.batch_size]
            ids = ids_all[bidx].to(dev); roles = roles_all[bidx].to(dev)
            valid = roles != -100
            xm = ids.clone(); tmask = torch.zeros_like(valid)
            for b in range(ids.size(0)):
                nv = int(valid[b].sum())
                frac = rng.uniform(a.mask_lo, a.mask_hi)
                block = rng.random() < a.block_mask_prob
                pos = sample_masked_positions(nv, frac, block, rng, dev)
                xm[b, pos] = mask_id; tmask[b, pos] = True
            _, hidden = model(xm, output_hidden_states=True)
            logits = head(hidden)
            sel = tmask & valid
            pred = logits[sel].argmax(-1)
            truth = roles[sel]
            tot_loss += F.cross_entropy(logits[sel], truth).item() * int(sel.sum())
            tot_n += int(sel.sum())
            for t_, p_ in zip(truth.tolist(), pred.tolist()):
                conf[t_, p_] += 1
    per_role = {}
    for r in range(N_ROLES):
        n_r = int(conf[r].sum()); ok_r = int(conf[r, r])
        per_role[ROLE_NAMES[r]] = {
            "n": n_r, "acc": round(ok_r / n_r, 4) if n_r else None}
    macro = sum(v["acc"] or 0 for v in per_role.values()) / N_ROLES
    metrics = {
        "ckpt": a.ckpt, "head_params": n_head, "steps": a.steps,
        "val_loss": round(tot_loss / max(tot_n, 1), 4),
        "val_acc_micro": round(float(conf.diag().sum() / conf.sum()), 4),
        "val_acc_macro": round(macro, 4),
        "per_role": per_role,
        "confusion": conf.tolist(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    torch.save({"head": head.state_dict(), "config": {
        "hidden": mcfg["hidden"], "head_hidden": a.head_hidden,
        "n_roles": N_ROLES, "role_names": ROLE_NAMES}},
        out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"VAL loss {metrics['val_loss']} micro {metrics['val_acc_micro']} "
        f"macro {metrics['val_acc_macro']}")
    log(f"per-role: {json.dumps(per_role)}")
    log("COMPLETE")


if __name__ == "__main__":
    main()
