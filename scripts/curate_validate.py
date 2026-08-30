#!/usr/bin/env python3
"""
curate_validate.py — VALIDA el pipeline de curacion por conectividad en una
muestra, para calibrar umbrales antes de correr sobre los 2.2M docs.

Etapas (muestra):
  1. Cargar N docs (v6 + phys) por dominio (estratificado).
  2. Embeddings BGE-small-en-v1.5 (dim 384, normalizados).
  3. kNN (k=20) con faiss/sklearn: similitud media a vecinos por doc.
  4. Reportar la DISTRIBUCION de conectividad (percentiles por dominio) ->
     define el umbral top-% para la seleccion final.
  5. Probar dedup semantico (cos > 0.95): cuantos pares duplicados detecta.

Salida (consola + json): data/curate/validate_report.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np

if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: curate_validate.py SOLO via Slurm.")

def load_sample(v6_path, phys_path, n_per_domain, seed=42):
    """Carga n docs por dominio de v6 + phys (estratificado). Devuelve lista
    de {text, domain, source}."""
    rng = random.Random(seed)
    per_domain = defaultdict(list)
    # v6
    if o:=os.path.exists(v6_path):
        with open(v6_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln=ln.strip()
                if not ln: continue
                try: d=json.loads(ln)
                except Exception: continue
                dom = d.get("domain") or "?"
                per_domain[dom].append((d.get("text") or "")[:1500])
                if len(per_domain[dom]) > n_per_domain*2: break
    # phys
    if os.path.exists(phys_path):
        with open(phys_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln=ln.strip()
                if not ln: continue
                try: d=json.loads(ln)
                except Exception: continue
                dom = d.get("domain") or "?"
                per_domain[dom].append((d.get("text") or "")[:1500])
    # muestra n por dominio
    out=[]
    for dom, texts in per_domain.items():
        if len(texts) > n_per_domain:
            texts = rng.sample(texts, n_per_domain)
        out.extend({"text": t, "domain": dom, "src": "v6" if dom not in
                    {"phys-astro","phys-cond","phys-hep","phys-physics","phys-quant","phys-grqc","phys-mathph","phys-nucl","phys-nlin"} else "phys"} for t in texts)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v6.jsonl")
    ap.add_argument("--phys", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_phys.jsonl")
    ap.add_argument("--n-per-domain", type=int, default=5000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/curate/validate_report.json")
    a = ap.parse_args()

    t0=time.time()
    sample = load_sample(a.v6, a.phys, a.n_per_domain)
    print(f"[curate-validate] muestra: {len(sample)} docs "
          f"(dominios: {len(set(d['domain'] for d in sample))}) [{time.time()-t0:.0f}s]", flush=True)

    # 1) embeddings
    from sentence_transformers import SentenceTransformer
    os.environ.setdefault("HF_HOME", "/beegfs/a474r867/hf-cache")
    t1=time.time()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [d["text"] for d in sample]
    vecs = model.encode(texts, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    print(f"[curate-validate] embeddings {vecs.shape} [{time.time()-t1:.0f}s]", flush=True)
    vecs = vecs.astype(np.float32)

    # 2) kNN con sklearn (BruteForce es O(n²) pero para 60K docs ok)
    from sklearn.neighbors import NearestNeighbors
    t2=time.time()
    nn = NearestNeighbors(n_neighbors=min(a.k+1, len(vecs)), metric="cosine").fit(vecs)
    dist, idx = nn.kneighbors(vecs, return_distance=True)  # dist=cosine dist (0=igual, 1=opuesto)
    sim = 1.0 - dist  # similitud coseno a vecinos
    sim_avg = sim[:, 1:].mean(axis=1)  # excluye el propio (vecino 0 = self)
    print(f"[curate-validate] kNN k={a.k} [{time.time()-t2:.0f}s]", flush=True)

    # 3) distribucion de conectividad por dominio
    dom_sim = defaultdict(list)
    for d, s in zip(sample, sim_avg):
        dom_sim[d["domain"]].append(float(s))
    print("\n== conectividad media por dominio (percentiles) ==", flush=True)
    perc_data = {}
    for dom, vals in sorted(dom_sim.items(), key=lambda x: -np.median(x[1])):
        vals = np.array(vals)
        p = np.percentile(vals, [10,25,50,75,90])
        print(f"  {dom:<12} n={len(vals):>5} mediana={np.median(vals):.3f} "
              f"p10={p[0]:.3f} p25={p[1]:.3f} p75={p[3]:.3f} p90={p[4]:.3f}", flush=True)
        perc_data[dom] = {"n": int(len(vals)), "median": float(np.median(vals)),
                          "p10": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
                          "p75": float(p[3]), "p90": float(p[4])}

    # 4) dedup semantico: pares con cos > 0.95
    t3=time.time()
    n_pairs, n_dup = 0, 0
    # muestrear 5K pares aleatorios por documento para estimar dup rate
    rng = random.Random(42)
    idx_arr = np.arange(len(vecs))
    # para velocidad: comprobar solo vecinos cercanos (idx de kNN ya los da)
    est_dup = 0
    for i in range(min(5000, len(vecs))):
        near = idx[i][1:]
        for j in near[:5]:
            if sim[i, list(idx[i]).index(j)] > 0.95:
                est_dup += 1
    dup_rate = est_dup / (min(5000, len(vecs)) * 5)
    print(f"\n[dedup-semantico] tasa pares con cos>0.95 (vecinos cercanos): "
          f"{dup_rate:.4f} ({est_dup} de {min(5000,len(vecs))*5}) [{time.time()-t3:.0f}s]", flush=True)

    # 5) estimar tokens a distintas fracciones
    all_sim = np.array(sim_avg)
    toks_by_dom = defaultdict(list)
    for d, s in zip(sample, sim_avg):
        toks_by_dom[d["domain"]].append((s, len(d["text"])//4))
    est = {}
    for frac in (0.3, 0.5, 0.7):
        tot = 0
        for dom, items in toks_by_dom.items():
            items.sort(key=lambda x: -x[0])
            n_keep = max(1, int(len(items)*frac))
            tot += sum(t for _, t in items[:n_keep])
        est[f"top{int(frac*100)}%"] = tot
    print("\n== estimacion tokens retenidos por fraccion (muestra) ==", flush=True)
    for k, v in est.items():
        print(f"  {k}: ~{v/1e6:.1f}M tok", flush=True)

    report = {"sample_n": len(sample), "k": a.k,
              "per_domain": perc_data, "dup_rate_cos95": float(dup_rate),
              "est_tokens_by_frac": est, "vecs_shape": list(vecs.shape),
              "elapsed_s": round(time.time()-t0,1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[curate-validate] DONE -> {a.out} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()