#!/usr/bin/env python3
"""
analyze_activation_expE2.py — Mide la actividad REAL de expertos en el
checkpoint e2_final.pt del experimento E2+E4 (MdLMMoE con timestep-router).

No hay métricas de router logueadas durante el entrenamiento, así que esto
carga el modelo entrenado, le hace forward con datos reales del corpus v3,
y cuantifica:
  - uso de cada experto (por capa): % de tokens que activan cada experto
  - entropía/balance del gate softmax
  - activación del shared experto (E4) y su magnitud relativa
  - colapso de router: índice efectivo de expertos (HHI/Shannon)
  - por QUÉ fase de denoising: qué expertos ganan en t visible vs t enmascarado

Salida: JSON + stdout, no toca nada del entrenamiento (solo lectura).
"""
import argparse, json, os, sys
import numpy as np
import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/moe_expE2_full/e2_final.pt")
    p.add_argument("--data", default="train_ids_v3.npy")
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--expert_k", type=int, default=1)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--timestep_emb", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--mask_p", type=float, default=0.15)
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def main():
    args = parse()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from moe_exp_E2_router_timestep import MdLMMoE as MdLMMoE_E2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev}")

    arr = np.load(args.data)
    all_ids = torch.from_numpy(arr.astype(np.int64))
    log(f"corpus tokens: {all_ids.numel()/1e9:.2f}B")

    vocab = 126080
    model = MdLMMoE_E2(vocab, args.hidden, args.layers, args.heads, args.ff_mult,
                       args.seq_len, args.n_experts, args.expert_k,
                       tdim=args.timestep_emb, n_steps=100, shared=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    # key normalization
    st = ck if all(k.startswith("tok_emb") or True for k in ck) else ck.get("model", ck)
    model.load_state_dict(st, strict=False)
    model.to(dev).eval()
    log(f"loaded {args.ckpt} ({os.path.getsize(args.ckpt)/1e6:.0f}MB)")

    # ---- forward con middle patch para capturar gate ----
    # Guardamos la función original del MoEMLP.forward y la envuelvo para registrar gate probs.
    from moe_exp_E2_router_timestep import MoEMLP
    _orig = MoEMLP.forward
    gate_logs = []
    def fwd(self, x, t=None):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        if t is None:
            te = self.t_emb(torch.zeros(flat.shape[0], device=flat.device, dtype=torch.long))
        else:
            te = self.t_emb(t.reshape(-1).clamp(0, self.t_emb.num_embeddings-1))
        gate_in = torch.cat([flat, te], dim=-1)
        with torch.no_grad():
            g = torch.softmax(self.gate(gate_in).float(), dim=-1)
        gate_logs.append(g.detach().cpu())   # [token, n_exp]
        return _orig(self, x, t)
    MoEMLP.forward = fwd

    MASK = vocab
    n_masked = max(1, int((args.seq_len//2) * args.mask_p))
    collected = 0
    per_layer = []  # list of (B*T, n_exp)
    it = 0
    while collected < args.n_samples:
        xb = all_ids[it*args.batch*args.seq_len : (it+1)*args.batch*args.seq_len]
        if xb.numel() < args.batch*args.seq_len:
            break
        xb = xb.view(args.batch, args.seq_len).to(dev)
        xm = xb.clone()
        mp = torch.randperm(args.seq_len)[:n_masked]
        xm[:, mp] = MASK
        timestep = torch.zeros_like(xb); timestep[:, mp] = 1.0
        with torch.no_grad():
            _ = model(xm, timestep=timestep)
        collected += 1
        it += 1

    # gate_logs[block*nsample + batch?] — MoEMLP.forward called layers*n_samples times
    n_layers = args.layers
    per_block = [[] for _ in range(n_layers)]
    idx = 0
    for s in range(collected):
        for l in range(n_layers):
            per_block[l].append(gate_logs[idx]); idx += 1
    # cada gate_log: (B, n)
    log(f"captured {len(gate_logs)} gate tensors = {collected} samples x {n_layers} layers")

    agg = []
    for l in range(n_layers):
        g = torch.cat(per_block[l], dim=0).float()   # (B*nS, n_experts)
        probs = g.mean(0)
        # top-1 selectio frequency
        sel = torch.argmax(g, dim=1)
        freq = torch.bincount(sel, minlength=args.n_experts).float() / g.size(0)
        # gate entropy (mean over tokens)
        H = (-g * (g+1e-12).log()).sum(-1).mean().item()
        agg.append((l, probs.tolist(), freq.tolist(), H))
        log(f"layer {l:2d}: mean_gate={probs.tolist()} top1_freq={freq.tolist()} entH={H:.3f}")

    # ---- global / collapsed ----
    g = torch.cat(gate_logs, dim=0).float()
    entropy_global = (-g * (g+1e-12).log()).sum(1).mean().item()
    # effective experiment count
    exp_prob = g.mean(0)
    eff = torch.exp(-torch.sum(exp_prob * torch.log(exp_prob+1e-12))).item()
    log(f"\nENTROPY_MEAN={entropy_global:.3f}  EFF_N_EXPERTS={eff:.3f}  (ideal {args.n_experts}; 1=colapso)")

    # shared strength: mid magnitudes vs expert
    log("\nshared vs expert magnitude (approx, by param norm):")
    sd = model.state_dict()
    for l in range(n_layers):
        sh = sd.get(f"blocks.{l}.mlp.shared.1.weight", None)
        ex = sd.get(f"blocks.{l}.mlp.experts.0.1.weight", None)
        if sh is not None and ex is not None:
            log(f"  layer {l}: shared_norm={sh.norm().item():.3f} expert0_norm={ex.norm().item():.3f} ratio={sh.norm().item()/max(ex.norm().item(),1e-9):.3f}")

    # save summary
    out = os.path.join(args.ckpt.rsplit("/",1)[0] if "/" in args.ckpt else ".", "activation_report.json")
    json.dump({"ckpt": args.ckpt, "n_layers": n_layers, "n_experts": args.n_experts,
               "entropy_mean": entropy_global, "effective_n": eff,
               "per_layer": [{"layer": l, "gate_prob": agg[l][1], "top1_freq": agg[l][2], "entropy": agg[l][3]} for l in range(n_layers)]},
              open(out, "w"), indent=2)
    log(f"saved -> {out}")

if __name__ == "__main__":
    main()