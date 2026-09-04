#!/usr/bin/env python3
"""suite_smoke.py — eval suite Fase 3 F0 para checkpoints dLLM (mdl-moe).

Evalúa un checkpoint SIN tocar el entrenador vivo:

  1. Discriminación inferencial pairwise (ranks, sin LLM-judge):
     lee pairs.jsonl (ctx/ok/bad en ids ya tokenizados, uno por línea)
     y exige que el denoising-loss del candidato correcto sea menor.
  2. Completación generativa (denoising iterativo estilo mask-predict).
  3. Diagnósticos de fluidez: word_ratio, rep4, uniq (de bw3/bw4_span).

Uso:
  python harness/suite_smoke.py \
      --ckpt /beegfs/.../checkpoint-best.pt \
      --config harness/configs/run.yaml \
      --pairs runs/f0/pairs.jsonl \
      --out runs/f0/report.json

Sin --pairs genera pares sintéticos deterministas (solo self-test del harness).
El mismo seed => mismas máscaras y pares => report.json comparable entre runs.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- import del modelo SIN tocar el entrenador vivo ----
# train_mdlm_moe.py ejecuta su argparse a nivel de módulo (--data/--output
# requeridos), así que lo importamos con sys.argv ficticio: las clases quedan
# definidas y main() nunca corre (el guard __main__ existe al final del archivo).
_trainer_path = Path(__file__).resolve().parents[1] / "scripts" / "train_mdlm_moe.py"
import importlib.util  # noqa: E402

_argv = sys.argv
sys.argv = [_trainer_path.name, "--data", "dummy", "--output", "/tmp/dummy"]
try:
    _spec = importlib.util.spec_from_file_location("train_mdlm_moe", _trainer_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MdLMMoE = _mod.MdLMMoE
finally:
    sys.argv = _argv

MASK_ID = None  # default: mcfg["vocab"] (embedding vocab+1, el token MASK es el índice vocab)


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _load_pairs(path, rng, max_ctx, max_cand):
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ctx = rec["ctx"][:max_ctx]
        ok = rec["ok"][:max_cand]
        bad = rec["bad"][:max_cand]
        if len(ctx) >= 2 and len(ok) and len(bad):
            pairs.append((ctx, ok, bad))
    return pairs


def _synth_pairs(rng, n, max_ctx, max_cand):
    """Self-test: ids aleatorios correlacionados -> "ok" siempre es el vecino real."""
    pairs = []
    for _ in range(n):
        base = [rng.randint(2, 30) for _ in range(max_ctx + max_cand + 5)]
        ctx = base[:max_ctx]
        ok = base[max_ctx : max_ctx + max_cand]
        bad = [rng.randint(2, 30) for _ in range(max_cand)]
        pairs.append((ctx, ok, bad))
    return pairs


def denoise_loss(model, seq, mask_p, rng, mask_id):
    """forward con máscara aleatoria determinista -> CE solo en posiciones mask."""
    T = seq.shape[0]
    n = max(1, int(mask_p * T))
    idx = torch.tensor(rng.sample(range(T), n), dtype=torch.long, device=seq.device)
    masked = seq.clone()
    masked[idx] = mask_id
    logits = model(masked.unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits[idx], seq[idx])


def generate(model, prompt_ids, max_new, steps, temp, rng, mask_id, mask_p):
    """denoising iterativo: en cada paso se remascara una fracción y se re-muestrea."""
    ids = torch.tensor(prompt_ids + [mask_id] * max_new, dtype=torch.long)
    n = len(ids)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(ids.unsqueeze(0)).squeeze(0)
            probs = (logits / max(temp, 1e-6)).softmax(-1)
            # muestrear SOLO posiciones aún enmascaradas
            still = (ids == mask_id).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                break
            new = torch.multinomial(probs[still], 1).squeeze(-1)
            ids[still] = new
            # remascara una fracción (mask-predict estándar), salvo el prompt
            remask_n = max(1, int(mask_p * n))
            remask = rng.sample(range(len(prompt_ids), n), min(remask_n, n - len(prompt_ids)))
            if remask:
                ids[torch.tensor(remask, device=ids.device)] = mask_id
    return ids.tolist()


def fluency_from_ids(ids):
    """Diagnósticos sobre ids (no necesita tokenizer): rep4 + uniq."""
    T = len(ids)
    if T < 5:
        return {"rep4": 0.0, "uniq": 0.0}
    grams = [tuple(ids[i : i + 4]) for i in range(T - 3)]
    rep4 = round(1.0 - len(set(grams)) / len(grams), 4)
    uniq = round(len(set(ids)) / T, 4)
    return {"rep4": rep4, "uniq": uniq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None, help="jsonl de pares tokenizados")
    ap.add_argument("--prompt-ids", default="", help="ids separados por coma para completación")
    ap.add_argument("--no-gen", action="store_true", help="saltar generación (eval rápida)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-id", type=int, default=None,
                    help="token MASK; default = vocab del config (126080)")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, ecfg = cfg["model"], cfg["eval"]
    mask_id = args.mask_id if args.mask_id is not None else mcfg["vocab"]
    seed = cfg["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]

    model = MdLMMoE(
        vocab=mcfg["vocab"], hidden=mcfg["hidden"], layers=mcfg["layers"],
        heads=mcfg["heads"], ff_mult=mcfg["ff_mult"], seq_len=mcfg["seq_len"],
        n_experts=mcfg["n_experts"], k=mcfg["k"],
    ).to(dev)
    model.load_state_dict(ck, strict=False)
    model.eval()

    # ---- 1) discriminación pairwise ----
    if args.pairs:
        pairs = _load_pairs(args.pairs, rng, mcfg["seq_len"] // 2, mcfg["seq_len"] // 4)
    else:
        pairs = _synth_pairs(rng, ecfg["n_pairs"], 64, 32)
        print(f"[warn] sin --pairs: usando pares sintéticos (self-test), "
              f"no apto para evaluar el modelo real")
    if not pairs:
        print(f"[fatal] pairs vacío", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    ok_wins, deltas = 0, []
    with torch.no_grad():
        for ctx, ok, bad in pairs:
            seq_ok = torch.tensor(ctx + ok, dtype=torch.long, device=dev)
            seq_bad = torch.tensor(ctx + bad, dtype=torch.long, device=dev)
            l_ok = denoise_loss(model, seq_ok, ecfg["mask_p"], rng, mask_id).item()
            l_bad = denoise_loss(model, seq_bad, ecfg["mask_p"], rng, mask_id).item()
            ok_wins += l_ok < l_bad
            deltas.append(l_bad - l_ok)
    discr = {
        "pairwise_acc": round(ok_wins / len(pairs), 4),
        "n_pairs": len(pairs),
        "mean_delta": round(float(np.mean(deltas)) if deltas else 0.0, 5),
    }

    # ---- 2/3) completación + fluidez ----
    gen = {}
    if not args.no_gen:
        if args.prompt_ids:
            prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
        elif pairs:
            prompt = pairs[0][0][:32]
        else:
            prompt = []
        gen_ids = generate(model, prompt, ecfg["max_new"], ecfg["steps"],
                           ecfg["temp"], rng, mask_id, ecfg["mask_p"])
        n_gen = len(gen_ids) - len(prompt)
        gen = {"completed_len": n_gen, "prompt_len": len(prompt)}
        gen.update(fluency_from_ids(gen_ids[len(prompt):]))

    elapsed = time.time() - t0
    rep = {
        "config_sha256": "",  # lo rellena report.py
        "seed": seed,
        "tag": cfg["out"]["tag"],
        "elapsed_s": round(elapsed, 2),
        "discrimination": discr,
        "generation": gen,
        "device": args.device,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({"tag": rep["tag"], "pairwise_acc": discr["pairwise_acc"],
                      "mean_delta": discr["mean_delta"], "generation": gen}))


if __name__ == "__main__":
    main()