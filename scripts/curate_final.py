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
    """Streaming: yield {text (truncado CTX), full_len, domain, src, line}.
    line = indice DENTRO del archivo (el enumerate reinicia por archivo);
    src distingue archivo. Bug 2026-08-29: usar un indice global plano en la
    escritura hacia colisionar los indices de phys con las primeras lineas de
    v6 (corpus mezclado)."""
    for path, src in ((v6, "v6"), (phys, "phys")):
        if not os.path.exists(path): continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, ln in enumerate(f):
                ln = ln.strip()
                if not ln: continue
                try: d = json.loads(ln)
                except Exception: continue
                raw = d.get("text") or ""
                txt = raw[:CTX]
                if len(txt) < 200: continue
                yield {"text": txt, "full_len": len(raw),
                       "domain": d.get("domain") or "?", "src": src,
                       "line": i,
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
    ap.add_argument("--only-embed", action="store_true",
                    help="solo embeddear y guardar vectors (primera fase, GPU)")
    ap.add_argument("--from-vecs", action="store_true",
                    help="cargar vectors guardados y hacer kNN+dedup+seleccion (segunda fase, CPU)")
    a = ap.parse_args()

    t0 = time.time()

    if a.from_vecs:
        # FASE 2: leer vectors precomputados + docs (solo metadata/text corto)
        if not os.path.exists(a.vecs):
            sys.exit(f"ERROR: no existe {a.vecs} (corre primero la fase de embed)")
        vecs = np.load(a.vecs).astype(np.float32)
        docs = list(gen_docs(a.v6, a.phys))
        print(f"[curate] fase2: cargo {len(docs)} docs + vectors {vecs.shape} [{time.time()-t0:.0f}s]", flush=True)
    else:
        # FASE 1: embeddear
        docs = list(gen_docs(a.v6, a.phys))
        print(f"[curate] docs totales: {len(docs)} [{time.time()-t0:.0f}s]", flush=True)
        print("dominios:", dict(Counter(d["domain"] for d in docs)), flush=True)
        os.environ.setdefault("HF_HOME", "/beegfs/a474r867/hf-cache")
        from sentence_transformers import SentenceTransformer
        t1 = time.time()
        model = SentenceTransformer("BAAI/bge-small-en-v1.5",
                                    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        texts = [d["text"] for d in docs]
        vecs = model.encode(texts, batch_size=a.batch, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        print(f"[curate] embeddings {vecs.shape} [{time.time()-t1:.0f}s] -> {a.vecs}", flush=True)
        os.makedirs(os.path.dirname(a.vecs) or ".", exist_ok=True)
        np.save(a.vecs, vecs)
        print(f"[curate] vectors guardados OK ({os.path.getsize(a.vecs)/1e9:.1f}GB)", flush=True)
        if a.only_embed:
            print("FASE1_DONE", flush=True)
            return

    # 2) kNN + conectividad (faiss HNSW aproximado: O(n^2) de sklearn es inviable
    # para 2.2M docs — matado 2026-08-29 tras 30 min solo en el fit)
    t2 = time.time()
    import faiss
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)          # vecs normalizados: IP = coseno
    index.hnsw.efConstruction = 200
    index.add(vecs)
    kk = min(a.k + 1, len(vecs))
    sim, idx = index.search(vecs, kk)            # sim = coseno (normalizados)
    con = sim[:, 1:].mean(axis=1) if kk > 1 else np.zeros(len(vecs))
    print(f"[curate] kNN HNSW k={a.k} conectividad media={con.mean():.3f} "
          f"[{time.time()-t2:.0f}s]", flush=True)

    # 3) dedup semantico (cos>dup_cos) — recorrer vecinos cercanos de cada doc
    t3 = time.time()
    keep = np.ones(len(vecs), bool)
    n_dup = 0
    for i in range(len(vecs)):
        if not keep[i]:
            continue
        for j in range(1, kk):
            jj = int(idx[i, j])
            if jj == i or not keep[jj]:
                continue
            if sim[i, j] > a.dup_cos:
                keep[jj] = False          # descarta el que aparece como vecino
                n_dup += 1
    print(f"[curate] dedup semantico cos>{a.dup_cos}: {n_dup} duplicados "
          f"[{time.time()-t3:.0f}s]", flush=True)

    # 4) seleccion por MASA DE TOKENS (no por docs): ordenar por conectividad
    #    desc dentro de cada dominio y acumular hasta frac% de los TOKENS del
    #    dominio. 2026-08-29: top-50% de docs retenia 94% de tokens (los docs
    #    conectados son los largos) — no reducia el corpus.
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
        tot_tok_dom = sum(docs[i]["full_len"] // 4 for i in ids)
        acc = 0
        sel = []
        for i in ids:
            acc += docs[i]["full_len"] // 4
            sel.append(i)
            if acc >= tot_tok_dom * a.frac:
                break
        if not sel:
            sel = [ids[0]]
        sel_idx.extend(sel)
        dom_stats[dom] = {"total": len(ids), "kept": len(sel),
                          "frac_docs": round(len(sel) / max(1, len(ids)), 3),
                          "frac_tokens": round(acc / max(1, tot_tok_dom), 3)}
        n_tot += len(ids); n_keep_tot += len(sel)
        tok_tot += tot_tok_dom
        tok_keep += acc
    # guardar conectividad para iteraciones futuras (evita re-kNN)
    np.save(os.path.join(os.path.dirname(a.vecs) or ".", "con_full.npy"), con)
    print(f"[curate] seleccion por tokens: {n_keep_tot}/{n_tot} docs, "
          f"tokens {tok_keep/1e9:.2f}B de {tok_tot/1e9:.2f}B [{time.time()-t4:.0f}s]", flush=True)

    # 5) escribir corpus curado — STREAMING desde los jsonl ORIGINALES (texto
    #    COMPLETO, no el truncado a CTX del embedder — bug 2026-08-29: el corpus
    #    quedo con 1400 chars/doc y solo 369M tok en vez de ~2B)
    t5 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # sets de lineas por archivo (src): los indices de gen_docs son GLOBALES
    # (posicion en docs), pero las lineas del jsonl reinician por archivo.
    sel_by_src = {"v6": set(), "phys": set()}
    for i in sel_idx:
        sel_by_src.setdefault(docs[i]["src"], set()).add(docs[i]["line"])
    tok_keep_real = 0
    n_out = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for path, src in ((a.v6, "v6"), (a.phys, "phys")):
            if not os.path.exists(path):
                continue
            want = sel_by_src.get(src, set())
            if not want:
                continue
            with open(path, encoding="utf-8", errors="ignore") as g:
                for i, ln in enumerate(g):
                    if i not in want:
                        continue
                    ln = ln.strip()
                    if not ln:
                        continue
                    f.write(ln + "\n")
                    tok_keep_real += len(ln) // 4
                    n_out += 1
    print(f"[curate] escrito: {n_out} docs completos, ~{tok_keep_real/1e6:.0f}M tok "
          f"[{time.time()-t5:.0f}s]", flush=True)
    report = {"n_total": len(docs), "n_kept": n_keep_tot, "frac": a.frac,
              "k": a.k, "dup_cos": a.dup_cos, "n_dup_sem": n_dup,
              "tok_total": tok_tot, "tok_kept": tok_keep,
              "tok_kept_real": tok_keep_real, "n_out": n_out,
              "domains": dom_stats, "vecs": a.vecs,
              "elapsed_s": round(time.time()-t0, 1)}
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[curate] DONE {n_keep_tot} docs -> {a.out} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()