#!/usr/bin/env python3
"""smoke_cont_bw3.py — DIAGNOSTICO de CONTINUACION para bw3.

Pregunta: ¿el modelo sabe CONTINUAR texto real (contexto fijo) o su basura es
solo de la generacion en frio (todo-mascara, arranque desde cero)?

Metodo: se toman N documentos REALES del corpus v7, se tokenizan, y se generan
n_gen tokens de continuacion con el mismo denoising iterativo LLaDA de
smoke_l1.py (contexto fijo + mascara en los tokens nuevos). Se barre
ctx_len x temp x steps, incluyendo ctx_len=0 (control = generacion en frio,
lo que ya hacia el smoke normal).

Metricas vs baseline de TEXTO REAL del mismo corpus (tramo equivalente):
  - word_ratio : fraccion de tokens que decodifican a palabra (empieza con letra)
  - rep4       : fraccion de 4-gramas de tokens repetidos (degeneracion)
  - uniq_ratio : tokens unicos / total (riqueza de vocabulario)
Si con contexto la generacion se acerca al baseline real -> el problema es el
arranque en frio (sampler/arq). Si sigue siendo basura -> el modelo no tiene
sintaxis y el objetivo de entrenamiento no la ensena.

Las clases del modelo se importan de smoke_l1.py (misma copia embebida del
trainer; main() esta protegido, importar no ejecuta argparse).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_l1 import load_model, gen, TOK_PATH

BASE = "/beegfs/a474r867/ecoreasoner"


def word_ratio(ids, tok):
    if not ids:
        return 0.0
    n_words = 0
    for i in ids:
        s = tok.decode([int(i)], skip_special_tokens=True).strip()
        if s and s[0].isalpha():
            n_words += 1
    return n_words / len(ids)


def rep4(ids):
    if len(ids) < 5:
        return 0.0
    seen, reps = set(), 0
    for i in range(len(ids) - 3):
        g = tuple(ids[i:i + 4])
        if g in seen:
            reps += 1
        seen.add(g)
    return reps / (len(ids) - 3)


def uniq_ratio(ids):
    return len(set(ids)) / max(1, len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BASE + "/outputs/bw3/checkpoint-g77000")
    ap.add_argument("--corpus", default=BASE + "/data/curate/train_corpus_v7_curated.jsonl")
    ap.add_argument("--n-docs", type=int, default=6)
    ap.add_argument("--ctx-lens", default="0,64,200,400")
    ap.add_argument("--n-gen", type=int, default=180)
    ap.add_argument("--temps", default="0.7,0.9")
    ap.add_argument("--steps", default="64,128")
    ap.add_argument("--out", default=BASE + "/data/l1/smoke_cont_bw3.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = load_model(a.ckpt, device)
    print(f"[cont] modelo cargado {a.ckpt} ({time.time()-t0:.0f}s) device={device}",
          flush=True)

    # contextos reales: primeros n_docs con >= 500 tokens
    ctxs = []
    with open(a.corpus, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            ids = tok.encode(d.get("text", ""))[:600]
            if len(ids) < 500:
                continue
            ctxs.append(ids)
            if len(ctxs) >= a.n_docs:
                break
    if not ctxs:
        sys.exit("ERROR: no hay docs con >=500 tokens en el corpus")
    print(f"[cont] {len(ctxs)} contextos reales (>=500 tok)", flush=True)

    ctx_lens = [int(x) for x in a.ctx_lens.split(",")]
    temps = [float(x) for x in a.temps.split(",")]
    stepss = [int(x) for x in a.steps.split(",")]

    report = {"ckpt": a.ckpt, "n_docs": len(ctxs), "ctx_lens": ctx_lens,
              "temps": temps, "steps": stepss, "n_gen": a.n_gen,
              "baselines_real": {}, "cfgs": {}, "samples": []}
    # baselines sobre texto real (tramo de 180 tok tras el contexto)
    for ids in ctxs:
        seg = ids[400:580]
        report["baselines_real"].setdefault("word_ratio", []).append(word_ratio(seg, tok))
        report["baselines_real"].setdefault("rep4", []).append(rep4(seg))
        report["baselines_real"].setdefault("uniq_ratio", []).append(uniq_ratio(seg))
    for k in report["baselines_real"]:
        report["baselines_real"][k] = round(float(np.mean(report["baselines_real"][k])), 3)

    t1 = time.time()
    for cl in ctx_lens:
        for temp in temps:
            for steps in stepss:
                key = f"ctx{cl}_t{temp}_s{steps}"
                row = {"word_ratio": [], "rep4": [], "uniq_ratio": [], "json_found": 0}
                for i, ids in enumerate(ctxs):
                    ctx = ids[:cl] if cl > 0 else []
                    gen_ids = gen(m, ctx, n_gen=a.n_gen, steps=steps, temp=temp,
                                  device=device)
                    row["word_ratio"].append(word_ratio(gen_ids, tok))
                    row["rep4"].append(rep4(gen_ids))
                    row["uniq_ratio"].append(uniq_ratio(gen_ids))
                    out = tok.decode(gen_ids, skip_special_tokens=True)
                    if "{" in out:
                        row["json_found"] += 1
                    if i == 0:
                        ctx_txt = tok.decode(ctx, skip_special_tokens=True) if cl else "(frio)"
                        report["samples"].append({
                            "cfg": key, "ctx_snip": ctx_txt[-160:],
                            "gen_snip": out[:240]})
                for k in ("word_ratio", "rep4", "uniq_ratio"):
                    row[k] = round(float(np.mean(row[k])), 3)
                row["elapsed_s"] = round(time.time() - t1, 0)
                report["cfgs"][key] = row
                print(f"[cont] {key}: word={row['word_ratio']} rep4={row['rep4']} "
                      f"uniq={row['uniq_ratio']} json={row['json_found']} "
                      f"[{row['elapsed_s']} s]", flush=True)

    report["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[cont] DONE -> {a.out} [{report['elapsed_s']} s]", flush=True)


if __name__ == "__main__":
    main()