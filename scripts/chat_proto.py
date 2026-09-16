"""chat_proto.py — REPL local para el prototipo dLLM-MoE (EcoReasoner v1).

Modelo base de difusión enmascarada: NO es chat-instruct; "charlar" = el
modelo continúa el texto que escribes (prosa científica EN). Arquitectura
igual a la cadena v4: vocab 126080 (LLaDA) + MASK en el slot vocab,
hidden 768, 12 capas, 12 heads, 8 expertos top-1, seq_len 768.

Uso:
  python3 scripts/chat_proto.py \
      --ckpt outputs/retrain_v4/checkpoint-g11000/model.pt \
      --tokenizer /ruta/al/snapshot/LLaDA-8B-Instruct \
      [--device cuda] [--steps 24] [--max-new 128] [--temp 0.7]

El ckpt es el state_dict plano guardado por train_mdlm_moe.py
(model.pt dentro de checkpoint-gN). Si se pasa el dir checkpoint-gN,
usa model.pt dentro; si se pasa el OUT dir, toma el checkpoint-gN más
alto. Comandos REPL: /steps N, /new N, /temp F, /quit.
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


def load_model_class():
    """Importa MdLMMoE de train_mdlm_moe.py sin correr main (patrón suite_smoke)."""
    path = REPO / "scripts" / "train_mdlm_moe.py"
    argv_bak, usr1_bak = sys.argv, signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    try:
        sys.argv = [path.name]
        spec = importlib.util.spec_from_file_location("train_mdlm_moe", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MdLMMoE
    finally:
        sys.argv = argv_bak
        signal.signal(signal.SIGUSR1, usr1_bak)


def resolve_ckpt(p):
    p = Path(p)
    if p.is_file():
        return p
    if (p / "model.pt").is_file():
        return p / "model.pt"
    cands = sorted(p.glob("checkpoint-g*/model.pt"),
                   key=lambda x: int(re.search(r"g(\d+)", x.parent.name).group(1)))
    if not cands:
        sys.exit(f"no checkpoint-g*/model.pt bajo {p}")
    return cands[-1]


def load_model(ckpt_path, dev):
    MdLMMoE = load_model_class()
    model = MdLMMoE(**ARCH).to(dev)
    ck = torch.load(ckpt_path, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError:
        if all(k.startswith("module.") for k in ck):
            model.load_state_dict(
                {k[len("module."):]: v for k, v in ck.items()}, strict=True)
        else:
            raise
    model.eval()
    return model


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="model.pt | dir checkpoint-gN | dir OUT (toma el más alto)")
    ap.add_argument("--tokenizer", required=True, help="dir snapshot LLaDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    dev = torch.device(args.device)
    ckpt = resolve_ckpt(args.ckpt)
    print(f"[proto] ckpt={ckpt} dev={dev}")
    model = load_model(ckpt, dev)
    mask_id = ARCH["vocab"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    print(f"[proto] listo ({model.n_params()/1e6:.0f}M params). "
          f"steps={args.steps} max_new={args.max_new} temp={args.temp}")
    print("[proto] /steps N /new N /temp F /quit — o escribe texto para continuar")

    steps, max_new, temp = args.steps, args.max_new, args.temp
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
        prompt_ids = prompt_ids[-(ARCH["seq_len"] - max_new):]
        out = generate(model, prompt_ids, max_new, steps, temp, rng, mask_id)
        new_ids = [t for t in out[len(prompt_ids):] if t != mask_id]
        print(tok.decode(new_ids))
    print("\n[proto] bye")


if __name__ == "__main__":
    main()
