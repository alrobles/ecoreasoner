#!/usr/bin/env python3
"""suite_smoke_v2.py — eval suite para checkpoints entrenados con train_mdlm_moe_v2.py.

Evalúa un checkpoint v2 (soporta RoPE, weight_tying, packing EOS).

  1. Discriminación inferencial pairwise (ranks, sin LLM-judge):
     lee pairs.jsonl (ctx/ok/bad en ids ya tokenizados, uno por línea)
     y exige que el denoising-loss del candidato correcto sea menor.
  2. Completación generativa (denoising iterativo estilo mask-predict).
  3. Diagnósticos de fluidez: word_ratio, rep4, uniq (de bw3/bw4_span).

Uso simple (un par):
  python harness/suite_smoke_v2.py \
      --ckpt /beegfs/.../checkpoint-best.pt \
      --config harness/configs/run.yaml \
      --pairs runs/f0/pairs.jsonl \
      --out runs/f0/report.suite.json

Uso batería L0-L3:
  python harness/suite_smoke_v2.py \
      --ckpt /beegfs/.../checkpoint-best.pt \
      --config harness/configs/run.yaml \
      --pairs-dir runs/pairs_hard_v2 \
      --out-dir runs/f0/battery

Sin --pairs/--pairs-dir genera pares sintéticos deterministas (self-test del harness).
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- import del modelo v2 ----
_trainer_path = Path(__file__).resolve().parents[1] / "scripts" / "train_mdlm_moe_v2.py"
import importlib.util  # noqa: E402

_argv = sys.argv
sys.argv = [_trainer_path.name, "--data", "dummy", "--output", "/tmp/dummy"]
try:
    _spec = importlib.util.spec_from_file_location("train_mdlm_moe_v2", _trainer_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MdLMMoE = _mod.MdLMMoE
finally:
    sys.argv = _argv


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
    """Self-test: ids aleatorios correlacionados -> 'ok' siempre es el vecino real."""
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
    dev = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + [mask_id] * max_new, dtype=torch.long, device=dev)
    n = len(ids)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(ids.unsqueeze(0)).squeeze(0)
            probs = (logits / max(temp, 1e-6)).softmax(-1)
            still = (ids == mask_id).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                break
            new = torch.multinomial(probs[still], 1).squeeze(-1)
            ids[still] = new
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


def _eval(model, pairs, mcfg, ecfg, mask_id, seed, no_gen, device):
    """Evalúa un conjunto de pares y devuelve el dict de reporte."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    ok_wins, deltas = 0, []
    with torch.no_grad():
        for ctx, ok, bad in pairs:
            seq_ok = torch.tensor(ctx + ok, dtype=torch.long, device=device)
            seq_bad = torch.tensor(ctx + bad, dtype=torch.long, device=device)
            l_ok = denoise_loss(model, seq_ok, ecfg["mask_p"], rng, mask_id).item()
            l_bad = denoise_loss(model, seq_bad, ecfg["mask_p"], rng, mask_id).item()
            ok_wins += l_ok < l_bad
            deltas.append(l_bad - l_ok)
    discr = {
        "pairwise_acc": round(ok_wins / len(pairs), 4),
        "n_pairs": len(pairs),
        "mean_delta": round(float(np.mean(deltas)) if deltas else 0.0, 5),
    }

    gen = {}
    if not no_gen and pairs:
        prompt = pairs[0][0][:32]
        gen_ids = generate(model, prompt, ecfg["max_new"], ecfg["steps"],
                           ecfg["temp"], rng, mask_id, ecfg["mask_p"])
        n_gen = len(gen_ids) - len(prompt)
        gen = {"completed_len": n_gen, "prompt_len": len(prompt)}
        gen.update(fluency_from_ids(gen_ids[len(prompt):]))

    elapsed = time.time() - t0
    return {
        "config_sha256": "",
        "seed": seed,
        "discrimination": discr,
        "generation": gen,
        "elapsed_s": round(elapsed, 2),
        "device": str(device),
    }


def _load_model(args, mcfg):
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]

    use_rope = args.use_rope or mcfg.get("use_rope", False)
    weight_tying = args.weight_tying or mcfg.get("weight_tying", False)
    model = MdLMMoE(
        vocab=mcfg["vocab"], hidden=mcfg["hidden"], layers=mcfg["layers"],
        heads=mcfg["heads"], ff_mult=mcfg["ff_mult"], seq_len=mcfg["seq_len"],
        n_experts=mcfg["n_experts"], k=mcfg["k"],
        use_rope=use_rope, weight_tying=weight_tying,
    ).to(dev)
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
    return model, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="archivo de salida (modo --pairs) o directorio (modo --pairs-dir)")
    ap.add_argument("--out-dir", default=None, dest="out_dir", help="directorio de salida para la batería")
    ap.add_argument("--pairs", default=None, help="jsonl de pares tokenizados")
    ap.add_argument("--pairs-dir", default=None, dest="pairs_dir",
                    help="directorio con pairs_L{0..3}.jsonl")
    ap.add_argument("--prompt-ids", default="", help="ids separados por coma para completación (modo --pairs)")
    ap.add_argument("--no-gen", action="store_true", help="saltar generación (eval rápida)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-id", type=int, default=None,
                    help="token MASK; default = vocab del config")
    ap.add_argument("--use-rope", action="store_true")
    ap.add_argument("--weight-tying", action="store_true")
    args = ap.parse_args()

    if not args.pairs and not args.pairs_dir:
        ap.error("se requiere --pairs o --pairs-dir (o --pairs-dir con self-test)")

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, ecfg = cfg["model"], cfg["eval"]
    mask_id = args.mask_id if args.mask_id is not None else mcfg["vocab"]
    seed = cfg["seed"]
    tag = cfg["out"]["tag"]

    model, dev = _load_model(args, mcfg)

    # ---- modo batería L0-L3 ----
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        out_dir = Path(args.out_dir or args.out)
        if not out_dir:
            ap.error("modo --pairs-dir requiere --out-dir o --out")
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(pairs_dir.glob("pairs_L*.jsonl"))
        if not files:
            print(f"[fatal] sin pairs_L*.jsonl en {pairs_dir}", file=sys.stderr)
            sys.exit(2)

        levels = {}
        max_ctx = mcfg["seq_len"] // 2
        max_cand = mcfg["seq_len"] // 4
        for fp in files:
            lvl = fp.stem.split("_L")[-1]
            if lvl not in ("0", "1", "2", "3"):
                continue
            pairs = _load_pairs(fp, random.Random(seed), max_ctx, max_cand)
            if not pairs:
                print(f"[warn] {fp} vacío", file=sys.stderr)
                continue
            rep = _eval(model, pairs, mcfg, ecfg, mask_id, seed, args.no_gen, dev)
            rep["tag"] = f"{tag}_L{lvl}"
            rep["level"] = f"L{lvl}"
            out_path = out_dir / f"battery_L{lvl}.json"
            out_path.write_text(json.dumps(rep, indent=2))
            levels[lvl] = rep
            print(json.dumps({"level": f"L{lvl}", "pairwise_acc": rep["discrimination"]["pairwise_acc"],
                              "mean_delta": rep["discrimination"]["mean_delta"],
                              "n_pairs": rep["discrimination"]["n_pairs"]}))

        if not levels:
            print("[fatal] ningún nivel evaluable", file=sys.stderr)
            sys.exit(2)

        # resumen para report.py / index
        summary = {
            "tag": tag,
            "seed": seed,
            "levels": {f"L{k}": v for k, v in levels.items()},
            "discrimination": {
                "pairwise_acc": round(
                    sum(v["discrimination"]["pairwise_acc"] for v in levels.values()) / len(levels), 4),
                "n_pairs": sum(v["discrimination"]["n_pairs"] for v in levels.values()),
                "mean_delta": round(
                    sum(v["discrimination"]["mean_delta"] for v in levels.values()) / len(levels), 5),
            },
            "generation": levels[sorted(levels)[0]]["generation"],
            "device": str(dev),
        }
        (out_dir / "battery.json").write_text(json.dumps(summary, indent=2))
        return

    # ---- modo un par ----
    if not args.out:
        ap.error("modo --pairs requiere --out")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_ctx = mcfg["seq_len"] // 2
    max_cand = mcfg["seq_len"] // 4
    if args.pairs:
        pairs = _load_pairs(args.pairs, rng, max_ctx, max_cand)
    else:
        pairs = _synth_pairs(rng, ecfg["n_pairs"], max_ctx, max_cand)
        print(f"[warn] sin --pairs: usando pares sintéticos (self-test)")
    if not pairs:
        print("[fatal] pairs vacío", file=sys.stderr)
        sys.exit(2)

    # compatibilidad: --prompt-ids genera con prompt propio
    prompt = None
    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()]

    rep = _eval(model, pairs, mcfg, ecfg, mask_id, seed, args.no_gen, dev)
    rep["tag"] = tag
    if prompt is not None and not args.no_gen:
        gen_ids = generate(model, prompt, ecfg["max_new"], ecfg["steps"],
                           ecfg["temp"], rng, mask_id, ecfg["mask_p"])
        n_gen = len(gen_ids) - len(prompt)
        rep["generation"] = {"completed_len": n_gen, "prompt_len": len(prompt)}
        rep["generation"].update(fluency_from_ids(gen_ids[len(prompt):]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({"tag": rep["tag"], "pairwise_acc": rep["discrimination"]["pairwise_acc"],
                      "mean_delta": rep["discrimination"]["mean_delta"],
                      "generation": rep["generation"]}))


if __name__ == "__main__":
    main()
