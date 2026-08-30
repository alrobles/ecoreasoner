#!/usr/bin/env python3
"""
curate_final.py — CURA el corpus completo (v6 + phys) por CONECTIVIDAD CONCEPTUAL.

Pipeline (GPU para embeddings; CPU para kNN):
  1. Leer todos los docs de v6 (2M) + phys (204K) con su {text~1400, domain}.
  2. Embeddings BGE-small-en-v1.5 (dim 384) en GPU (lotes grandes, fp32).
  3. kNN (k=20) + similitud media por doc -> puntaje de CONECTIVIDAD.
  4. Dedup semantico (cos>umbral) -> descartar duplicados semanticos.
  5. Seleccion top-FRAC% por dominio (balanceado) por conectividad.
  6. Escribir train_corpus_v7_curated.jsonl (subconjunto curado) + report.

PARAMETROS (calibrados por curate_validate):
  --k 20  --dup-cos 0.95  --frac 0.5
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np

if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: curate_final.py SOLO via Slurm.")

CTX = 1400  # chars de texto para embedder (como validate)

def gen_docs(v6, phys):
    """Streaming: yield {text, domain, src}."""
    for path, src in ((v6, "v6"), (phys, "phys")):
        if not os.path.exists(path): continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try: d = json.loads(ln)
                except Exception: continue
                txt = (d.get("text") or "")[:CTX]
                if len(txt) < 200: continue
                yield {"text": txt, "domain": d.get("domain") or "?", "src": src,
                       "pmid": d.get("pmid") or d.get("arxiv_id") or d.get("pmcid") or ""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v6.jsonl")
    ap.add_argument("--phys", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_phys.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/curate/train_corpus_v7_curated.jsonl")
    ap.add_argument("--report", default="/beegfs/a474r867/ecoreasoner/data/curate/curate_final_report.json")
    ap.add_argument("--vecs", default="/beegfs/a474r867/ecoreasoner/data/curate/vectors_full.npy")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dup-cos", type=float, default=0.95)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()

    t0 = time.time()
    docs = list(gen_docs(a.v6, a.phys))
    print(f"[curate] docs totales: {len(docs)} [{time.time()-t0:.0f}s]", flush=True)
    print("dominios:", dict(Counter(d["domain"] for d in docs)), flush=True)

    # 1) embeddings en GPU
    os.environ.setdefault("HF_HOME", "/beegfs/a474r867/hf-cache")
    from sentence_transformers import SentenceTransformer
    t1 = time.time()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    texts = [d["text"] for d in docs]
    vecs = model.encode(texts, batch_size=a.batch, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"[curate] embeddings {vecs.shape} [{time.time()-t1:.0f}s] -> {a.vecs}", flush=True)
    np.save(a.vecs, vecs)

    # 2) kNN + conectividad
    t2 = time.time()
    from sklearn.neighbors import NearestNeighbors
    kk = min(a.k + 1, len(vecs))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine", n_jobs=-1).fit(vecs)
    dist, idx = nn.kneighbors(vecs, return_distance=True)
    sim = 1.0 - dist
    con = sim[:, 1:].mean(axis=1) if kk > 1 else np.zeros(len(vecs))
    print(f"[curate] kNN k={a.k} conectividad media={con.mean():.3f} [{time.time()-t2:.0f}s]", flush=True)

    # 3) dedup semantico (solo hacia atras, cos>dup_cos)
    t3 = time.time()
    keep = np.ones(len(vecs), bool)
    seen = set()
    # por dominio para no cruzar dominios al dedup (mas rapido y correcto)
    by_dom = defaultdict(list)
    for i, d in enumerate(docs):
        by_dom[d["domain"]].append(i)
    n_dup = 0
    for dom, ids in by_dom.items():
        if len(ids) < 2: continue
        sub = vecs[ids]
        s_nn = NearestNeighbors(n_neighbors=min(11, len(ids)), metric="cosine", n_jobs=-1).fit(sub)
        s_dist, s_idx = s_nn.kneighbors(sub, return_distance=True)
        for r, ids_r in enumerate(ids):
            for c in range(1, min(11, len(ids))):
                if 1.0 - s_dist[r, c] > a.dup_cos and keep[ids_r] and keep[s_idx[r, c]]:
                    keep[s_idx[r, c]] = False  # descarta el segundo
                    n_dup += 1
    print(f"[curate] dedup semantico cos>{a.dup_cos}: {n_dup} duplicados [{time.time()-t3:.0f}s]", flush=True)

    # 4) seleccion top-frac% por dominio (conectividad)
    t4 = time.time()
    n_tot = 0; n_keep_tot = 0; tok_keep = 0; tok_tot = 0
    sel_idx = []
    per_dom = defaultdict(list)
    for i, d in enumerate(docs):
        if not keep[i]: continue
        per_dom[d["domain"]].append(i)
    dom_stats = {}
    for dom, ids in per_dom.items():
        ids.sort(key=lambda i: -con[i])
        n_keep = max(1, int(len(ids) * a.frac))
        sel = ids[:n_keep]
        sel_idx.extend(sel)
        dom_stats[dom] = {"total": len(ids), "kept": len(sel), "frac": round(len(sel)/max(1,len(ids)),3)}
        n_tot += len(ids); n_keep_tot += len(sel)
        tok_tot += sum(len(docs[i]["text"])//4 for i in ids)
        tok_keep += sum(len(docs[i]["text"])//4 for i in sel)
    print(f"[curate] seleccion: {n_keep_tot}/{n_tot} docs ({n_keep_tot/max(1,n_tot):.0%}), "
          f"tokens {tok_keep/1e6:.0f}M de {tok_tot/1e6:.0f}M [{time.time()-t4:.0f}s]", flush=True)
    print("por dominio:", dict(dom_stats)[:300] if len(str(dom_stats))<300 else "…", flush=True)

    # 5) escribir corpus curado
    t5 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(42)
    rng.shuffle(sel_idx)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in sel_idx:
            d = docs[i]
            rec = {"text": d["text"], "domain": d["domain"], "source": d["src"],
                   "pmid": d["pmid"]}
            # preservar todos los campos originales posibles si el doc vino completo
            f.write(json.dumps({**rec}, ensure_ascii=False) + "\n")
    report = {"n_total": len(docs), "n_kept": n_keep_tot, "frac": a.frac,
              "k": a.k, "dup_cos": a.dup_cos, "n_dup_sem": n_dup,
              "tok_total": tok_tot, "tok_kept": tok_keep,
              "domains": dom_stats, "vecs": a.vecs,
              "elapsed_s": round(time.time()-t0,1)}
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[curate] DONE {n_keep_tot} docs -> {a.out} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()