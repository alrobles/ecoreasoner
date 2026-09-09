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


def denoise_loss(model, seq, mask_p, rng, mask_id, use_random_mask=True):
    """forward con máscara aleatoria determinista -> CE solo en posiciones mask.

    Si use_random_mask=False, evalua en las posiciones que ya están enmascaradas
    (útil para scoring de best-of-N de secuencias generadas).
    """
    if use_random_mask:
        T = seq.shape[0]
        n = max(1, int(mask_p * T))
        idx = torch.tensor(rng.sample(range(T), n), dtype=torch.long, device=seq.device)
    else:
        idx = (seq == mask_id).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            idx = torch.tensor([0], device=seq.device, dtype=torch.long)
    masked = seq.clone()
    masked[idx] = mask_id
    logits = model(masked.unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits[idx], seq[idx])


def _generate_one(model, prompt_ids, max_new, steps, temp, gen_seed, mask_id, mask_p, sampling="random"):
    """Genera una sola secuencia con el sampler elegido."""
    torch.manual_seed(gen_seed)
    dev = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + [mask_id] * max_new, dtype=torch.long, device=dev)
    n = len(ids)
    prompt_len = len(prompt_ids)
    with torch.no_grad():
        for step_i in range(steps):
            logits = model(ids.unsqueeze(0)).squeeze(0)
            still = (ids == mask_id).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                break
            if sampling == "low_confidence":
                # Fast-dLLM / LLaDA: desenmascarar primero las posiciones de mayor
                # confianza (argmax), dejando las dudosas para el siguiente paso.
                probs = (logits[still] / max(temp, 1e-6)).softmax(-1)
                pred = probs.argmax(-1)
                conf = probs.max(-1).values
                # por defecto desenmascara un 1/steps de las posiciones restantes
                k = max(1, int(still.numel() * (1.0 / max(steps - step_i, 1))))
                k = min(k, still.numel())
                # Elegir k posiciones por confianza (con temperatura: conf^temp)
                # Si temp bajo -> top-k; si temp alto -> mas diverso
                sel_probs = (conf / max(temp, 1e-6)).softmax(0)
                topk_idx = torch.multinomial(sel_probs, k, replacement=False)
                topk = still[topk_idx]
                # Elegir token: argmax puro si temp bajo, sino samplear
                token_probs = probs[topk_idx]
                if temp < 0.05:
                    ids[topk] = pred[topk_idx]
                else:
                    new = torch.multinomial(token_probs, 1).squeeze(-1)
                    ids[topk] = new
            else:
                # random: samplear todas las posiciones enmascaradas
                probs = (logits[still] / max(temp, 1e-6)).softmax(-1)
                new = torch.multinomial(probs, 1).squeeze(-1)
                ids[still] = new
                # remascara una fracción (mask-predict estándar), salvo el prompt
                remask_n = max(1, int(mask_p * n))
                remask_n = min(remask_n, n - prompt_len)
                if remask_n > 0:
                    # remascar posiciones con menor confianza para inducir revision
                    conf = probs.max(-1).values
                    low_conf = conf.argsort()[:remask_n]
                    remask = still[low_conf]
                    ids[remask] = mask_id
    return ids.tolist()


def generate(model, prompt_ids, max_new, steps, temp, rng, mask_id, mask_p,
             sampling="random", best_of_n=1, rerank=False):
    """Genera (best_of_n veces) y opcionalmente rerankea por denoise-loss."""
    base_seed = rng.randint(0, 2**31 - 1)
    best_ids = None
    best_score = float("inf")
    for i in range(best_of_n):
        gen_seed = base_seed + i
        cand = _generate_one(model, prompt_ids, max_new, steps, temp, gen_seed,
                             mask_id, mask_p, sampling=sampling)
        if not rerank and best_of_n == 1:
            return cand
        # puntuar con denoise-loss en el candidato (usar posiciones generadas)
        seq = torch.tensor(cand, dtype=torch.long, device=next(model.parameters()).device)
        score = denoise_loss(model, seq, mask_p, random.Random(gen_seed),
                             mask_id, use_random_mask=False).item()
        if score < best_score:
            best_score = score
            best_ids = cand
    return best_ids


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
    ap.add_argument("--sampling", default="random", choices=["random", "low_confidence"],
                    help="estrategia de denoising para generación")
    ap.add_argument("--best-of-n", type=int, default=1,
                    help="generar N candidatos y devolver el mejor (ver --rerank)")
    ap.add_argument("--rerank", action="store_true",
                    help="rerankar los best-of-N con denoise-loss")
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
    # STRICT=True (2026-09-08, auditoria 1.6): antes strict=False cargaba un
    # modelo semi-aleatorio en silencio si el ckpt no cuadraba con el config
    # (drift de arquitectura o prefijo "module." DDP) -> acc≈0.5 falso.
    # Ahora un desajuste REAL falla ruidoso; solo se tolera el prefijo "module."
    # (ckpt guardado con el wrapper DDP, caso legitimo).
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError as e:
        if all(k.startswith("module.") for k in ck):
            ck = {k[len("module."):]: v for k, v in ck.items()}
            model.load_state_dict(ck, strict=True)
        else:
            raise RuntimeError(
                f"state_dict desajustado contra el config (strict=True, "
                f"auditoria 1.6): {e}") from e
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
                           ecfg["temp"], rng, mask_id, ecfg["mask_p"],
                           sampling=args.sampling,
                           best_of_n=args.best_of_n,
                           rerank=args.rerank)
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