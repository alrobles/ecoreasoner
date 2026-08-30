#!/usr/bin/env python3
"""
build_arxiv_select.py — Selecciona ~300K papers de arXiv (fisica general) balanceados,
desde el metadata OAI snapshot (jackkuo/arXiv-metadata-oai-snapshot, JSONL 4.58GB).

Grupos de categorias (raiz de la taxonomia arXiv):
  physics.* | astro-ph.* | cond-mat.* | hep-th/hep-ph/hep-ex/hep-lat | quant-ph |
  gr-qc | nucl-th/nucl-ex | nlin.* | math-ph

Salida: data/arxiv/_arxiv_selected.jsonl
  {id, title, abstract, license, year, categories, domain}
  domain = "phys-<raiz>" (phys-physics, phys-astro, phys-cond, phys-hep,
           phys-quant, phys-grqc, phys-nucl, phys-nlin, phys-mathph)

Ejecutar SOLO via Slurm (guarda SLURM_JOB_ID, igual que build_v5/v6).
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict

if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: build_arxiv_select.py SOLO via Slurm (SLURM_JOB_ID ausente).")

# raiz -> prefijos de categoria arXiv
ROOTS = {
    "phys-physics": ["physics."],
    "phys-astro":   ["astro-ph."],
    "phys-cond":    ["cond-mat."],
    "phys-hep":     ["hep-th", "hep-ph", "hep-ex", "hep-lat"],
    "phys-quant":   ["quant-ph"],
    "phys-grqc":    ["gr-qc"],
    "phys-nucl":    ["nucl-th", "nucl-ex"],
    "phys-nlin":    ["nlin."],
    "phys-mathph":  ["math-ph"],
}

def root_of(categories: str):
    for cat in [c.strip() for c in (categories or "").split()]:
        for dom, pref in ROOTS.items():
            if any(cat.startswith(p) for p in pref):
                return dom
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="/beegfs/a474r867/ecoreasoner/data/arxiv/arxiv-metadata-oai-snapshot.json")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/arxiv/_arxiv_selected.jsonl")
    ap.add_argument("--target", type=int, default=300000)
    ap.add_argument("--min-abstract", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    by_root = defaultdict(list)
    n_total = n_elig = 0
    print(f"[arxiv-select] leyendo {a.meta} ...", flush=True)
    with open(a.meta, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: d = json.loads(ln)
            except Exception: continue
            n_total += 1
            root = root_of(d.get("categories", ""))
            if root is None: continue
            if not d.get("abstract") or len(d.get("abstract","")) < a.min_abstract: continue
            lic = d.get("license") or "arxiv-license"
            n_elig += 1
            # year = primer version (created: "Mon, 2 Apr 2007 19:18:42 GMT")
            yr = None
            vers = d.get("versions") or []
            if vers and isinstance(vers[0], dict):
                m = re.search(r"\b(19|20)\d{2}\b", vers[0].get("created",""))
                if m: yr = m.group(0)
            if yr is None:
                ud = d.get("update_date","")
                if len(ud) >= 4 and ud[:4].isdigit(): yr = ud[:4]
            by_root[root].append({
                "id": d.get("id",""),
                "title": d.get("title","").replace("\n", " ").strip(),
                "abstract": d.get("abstract",""),
                "license": lic,
                "year": yr,
                "categories": d.get("categories",""),
                "domain": root,
            })
    print(f"[arxiv-select] total={n_total} elegibles={n_elig} " 
          f"por_grupo={ {k: len(v) for k,v in sorted(by_root.items())} } "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # balancear: cuota proporcional a disponibilidad, tope por grupo
    random.seed(a.seed)
    out = []
    for root, items in sorted(by_root.items()):
        # cuota: reparto proporcional de a.target entre grupos (min 5% , max 40%)
        frac = len(items) / max(1, n_elig)
        quota = max(8000, min(int(a.target * frac), a.target // 3, len(items)))
        sel = random.sample(items, quota)
        out.extend(sel)
        print(f"  {root}: disp={len(items)} quota={quota}", flush=True)

    # rebalanceo global si nos pasamos -> recorte proporcional
    if len(out) > a.target:
        random.shuffle(out)
        out = out[:a.target]
    print(f"[arxiv-select] seleccion final: {len(out)} papers", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    doms = Counter(d["domain"] for d in out)
    yrs = Counter(str(d["year"]) for d in out)
    lic = Counter(d["license"] for d in out)
    print(f"[arxiv-select] DONE -> {a.out} [{time.time()-t0:.0f}s]", flush=True)
    print("dominios:", dict(doms), flush=True)
    print("years:", dict(sorted(yrs.items())), flush=True)
    print("licencias:", dict(lic), flush=True)

if __name__ == "__main__":
    main()