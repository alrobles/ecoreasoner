"""chat_proto_v2.py — REPL local para ckpts train_mdlm_moe_v2 (G8+ / G10 / G11).

Igual que chat_proto.py pero importa MdLMMoE de train_mdlm_moe_v2.py
(qkv/out + use_rope/weight_tying — los state_dicts v1/v2 no son
intercambiables). "Charlar" = el modelo continúa el texto que escribes
(prosa científica EN, esqueletos [OBSERVACION]...[CONCLUSION]).

Uso:
  python3 scripts/chat_proto_v2.py \
      --ckpt runs/g10-ep3-s1/checkpoint-g23000 --ema \
      --tokenizer /ruta/al/snapshot/LLaDA-8B-Instruct \
      [--device cuda] [--steps 24] [--max-new 128] [--temp 0.7]

--ckpt acepta: model.pt | dir checkpoint-gN | dir run (toma el gN más alto).
--ema usa ema_model.pt en vez de model.pt (los evals corren sobre EMA).
Comandos REPL: /steps N, /new N, /temp F, /quit.
"""
import argparse
import importlib.util
import random
import re
import signal
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
ARCH = dict(vocab=126080, hidden=768, layers=12, heads=12,
            ff_mult=4, seq_len=768, n_experts=8, k=1)


def detect_arch(sd):
    """Deriva dims del state_dict (G8-G11: 8 experts; v5-1b: 16)."""
    arch = dict(ARCH)
    emb = sd["tok_emb.weight"]
    arch["vocab"], arch["hidden"] = emb.shape[0] - 1, emb.shape[1]
    if "pos.weight" in sd:
        arch["seq_len"] = sd["pos.weight"].shape[0]
    layers = {int(m.group(1)) for k in sd
              if (m := re.search(r"blocks\.(\d+)\.", k))}
    exps = {int(m.group(1)) for k in sd
            if (m := re.search(r"experts\.(\d+)\.", k))}
    if layers:
        arch["layers"] = max(layers) + 1
    if exps:
        arch["n_experts"] = max(exps) + 1
    return arch


def load_model_class():
    """Importa MdLMMoE de train_mdlm_moe_v2.py sin correr main."""
    path = REPO / "scripts" / "train_mdlm_moe_v2.py"
    argv_bak, usr1_bak = sys.argv, signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    try:
        sys.argv = [path.name, "--data", "dummy", "--output", "/tmp/dummy_proto"]
        spec = importlib.util.spec_from_file_location("train_mdlm_moe_v2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MdLMMoE
    finally:
        sys.argv = argv_bak
        signal.signal(signal.SIGUSR1, usr1_bak)


def resolve_ckpt(p, ema=False):
    p = Path(p)
    if p.is_file():
        return p
    name = "ema_model.pt" if ema else "model.pt"
    if (p / name).is_file():
        return p / name
    cands = sorted(p.glob(f"checkpoint-g*/{name}"),
                   key=lambda x: int(re.search(r"g(\d+)", x.parent.name).group(1)))
    if not cands and ema:
        print(f"[proto] sin {name} bajo {p}, cayendo a model.pt")
        return resolve_ckpt(p, ema=False)
    if not cands:
        sys.exit(f"no checkpoint-g*/{name} bajo {p}")
    return cands[-1]


def load_model(ckpt_path, dev):
    MdLMMoE = load_model_class()
    ck = torch.load(ckpt_path, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    if isinstance(ck, dict) and "ema_model" in ck:
        ck = ck["ema_model"]
    arch = detect_arch(ck)
    model = MdLMMoE(**arch).to(dev)
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError:
        if all(k.startswith("module.") for k in ck):
            model.load_state_dict(
                {k[len("module."):]: v for k, v in ck.items()}, strict=True)
        else:
            raise
    model.eval()
    return model, arch


def generate(model, prompt_ids, max_new, steps, temp, rng, mask_id):
    """Mismo denoising que harness/suite_smoke.py::generate (mask-predict)."""
    dev = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + [mask_id] * max_new,
                       dtype=torch.long, device=dev)
    n = len(ids)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(ids.unsqueeze(0)).squeeze(0)
            probs = (logits / max(temp, 1e-6)).softmax(-1)
            still = (ids == mask_id).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                break
            ids[still] = torch.multinomial(probs[still], 1).squeeze(-1)
            remask_n = max(1, int(0.15 * n))
            remask = rng.sample(range(len(prompt_ids), n),
                                min(remask_n, n - len(prompt_ids)))
            if remask:
                ids[torch.tensor(remask, device=ids.device)] = mask_id
    return ids.tolist()


def generate_conf(model, prompt_ids, max_new, steps, temp, rng, mask_id):
    """Denoising por confianza (estilo LLaDA): cada step samplea x0 en TODAS
    las posiciones enmascaradas, toma la prob del token elegido como confianza,
    revela las de MAYOR confianza y remasquea el resto. La fraccion revelada
    crece linealmente hasta 100% en `steps` iteraciones."""
    dev = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + [mask_id] * max_new,
                       dtype=torch.long, device=dev)
    with torch.no_grad():
        for i in range(steps):
            logits = model(ids.unsqueeze(0)).squeeze(0)
            probs = (logits / max(temp, 1e-6)).softmax(-1)
            still = (ids == mask_id).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                break
            samp = torch.multinomial(probs[still], 1).squeeze(-1)
            conf = probs[still].gather(-1, samp[:, None]).squeeze(-1)
            ids[still] = samp
            target = int(round(max_new * (1.0 - (i + 1) / steps)))
            order = torch.argsort(conf)          # ascendente: peor primero
            low = still[order[:min(target, still.numel())]]
            if low.numel():
                ids[low] = mask_id
    return ids.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="model.pt | dir checkpoint-gN | dir run (toma el gN más alto)")
    ap.add_argument("--tokenizer", required=True, help="dir snapshot LLaDA")
    ap.add_argument("--ema", action="store_true",
                    help="usa ema_model.pt (pesos EMA del eval)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", action="append", default=[],
                    help="modo batch: genera para cada prompt y sale (no-REPL)")
    ap.add_argument("--decode", choices=["remask", "conf"], default="conf",
                    help="remask=remask aleatorio 15%%; conf=unmask por confianza")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    dev = torch.device(args.device)
    ckpt = resolve_ckpt(args.ckpt, ema=args.ema)
    print(f"[proto] ckpt={ckpt} dev={dev}")
    model, arch = load_model(ckpt, dev)
    mask_id = arch["vocab"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    print(f"[proto] listo ({model.n_params()/1e6:.0f}M params, "
          f"{arch['n_experts']} experts). "
          f"steps={args.steps} max_new={args.max_new} temp={args.temp}")

    steps, max_new, temp = args.steps, args.max_new, args.temp
    gen = generate_conf if args.decode == "conf" else generate
    if args.prompt:
        for line in args.prompt:
            prompt_ids = tok(line, add_special_tokens=False)["input_ids"]
            prompt_ids = prompt_ids[-(arch["seq_len"] - max_new):]
            out = gen(model, prompt_ids, max_new, steps, temp, rng,
                      mask_id)
            new_ids = [t for t in out[len(prompt_ids):] if t != mask_id]
            print(f"\n=== PROMPT: {line}\n{tok.decode(new_ids)}")
        return
    print("[proto] /steps N /new N /temp F /quit — o escribe texto para continuar")

    while True:
        try:
            line = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split()
            if parts[0] == "/quit":
                break
            if parts[0] == "/steps" and len(parts) > 1:
                steps = int(parts[1])
            elif parts[0] == "/new" and len(parts) > 1:
                max_new = int(parts[1])
            elif parts[0] == "/temp" and len(parts) > 1:
                temp = float(parts[1])
            print(f"[proto] steps={steps} max_new={max_new} temp={temp}")
            continue
        prompt_ids = tok(line, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-(arch["seq_len"] - max_new):]
        out = gen(model, prompt_ids, max_new, steps, temp, rng, mask_id)
        new_ids = [t for t in out[len(prompt_ids):] if t != mask_id]
        print(tok.decode(new_ids))
    print("\n[proto] bye")


if __name__ == "__main__":
    main()
