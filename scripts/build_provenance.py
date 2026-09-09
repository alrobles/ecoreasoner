#!/usr/bin/env python3
"""build_provenance.py — genera el manifiesto de procedencia del corpus de entrenamiento.

Para cada doc de train_corpus_v3.jsonl, asocia licencia por fuente:
  - pmc-full / pubmed-abstract con PMCID: pmc_license_map (por pmid)
  - pubmed-abstract sin PMCID: resolver pmid->pmcid via PMC-ids.csv.gz, o marcar
    no-pmc (investigacion/fair-use, sin redistribucion)
  - ecoevorxiv-pdf: ecoevorxiv_license_map (por article_id, match por titulo)

Ademas: sha256 del texto, tamano, clasificacion open-ready vs solo-entrenamiento.
Output: provenance_manifest.jsonl (una fila por doc).

Clasificacion de licencia:
  open  = redistribuible con atribucion (CC0, CC BY, CC BY-SA, dominio publico)
  train = solo-entrenamiento no-comercial (TDM, CC BY-NC*, CC BY-ND*, all-rights)
  unk   = sin licencia determinada (marcar como solo-entrenamiento por defecto)
"""
import json, hashlib, sys, time
from pathlib import Path

BASE = Path("/beegfs/a474r867/ecoreasoner/data")
OPEN_CODES = {"CC0", "CC BY", "CC BY-SA", "Public Domain", "PD"}
# PMC license_code -> clasificacion
def classify_pmc(code):
    code = (code or "").strip()
    if code in OPEN_CODES:
        return "open"
    # TDM, CC BY-NC*, CC BY-ND*, anything else -> solo entrenamiento
    return "train"
def classify_osf(name):
    name = (name or "").lower()
    if not name:
        return "train"
    # compartir texto completo permitido? ND (no-deriv) y NC (no-comercial) restringen redistribucion
    if "-nc" in name or "noncommercial" in name:
        return "train"
    if "-nd" in name or "no derivative" in name:
        return "train"
    # CC-BY puro o CC-BY-SA (share-alike permitido con misma licencia) -> abierto
    return "open"

t0 = time.time()
# 1) load pmc license by pmid
pmc_by_pmid = {}
for l in open(BASE / "pmc_license_map.jsonl"):
    d = json.loads(l)
    p = d.get("pmid")
    if p: pmc_by_pmid[str(p)] = d
print(f"[{time.strftime('%H:%M:%S')}] pmc_license_map por pmid: {len(pmc_by_pmid)}", flush=True)

# 2) ecoevorxiv license by article_id
eev_by_aid = {}
for l in open(BASE / "ecoevorxiv_license_map.jsonl"):
    d = json.loads(l)
    eev_by_aid[str(d.get("article_id"))] = d
print(f"[{time.strftime('%H:%M:%S')}] ecoevorxiv_license_map por article_id: {len(eev_by_aid)}", flush=True)

# 3) estado de conteo
stats = {"total": 0, "open": 0, "train": 0, "unk": 0}
by_src = {}
out = BASE / "provenance_manifest.jsonl"

with open(BASE / "train_corpus_v3.jsonl") as fin, open(out, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        pmid = str(d.get("pmid") or "")
        src = d.get("source")
        lic_code = None; lic_name = None; lic_src = None; pmcid = None
        # --- PMC (abstracts y fulltext) ---
        if src in ("pmc-full", "pubmed-abstract"):
            pmc = pmc_by_pmid.get(pmid)
            if pmc:
                lic_code = pmc.get("license_code")
                pmcid = pmc.get("pmcid")
                lic_src = "pmc"
        # --- EcoEvoRxiv ---
        elif src in ("ecoevorxiv-pdf",):
            # en v3 el article_id va incrustado en pmid como "EEV-<aid>"
            eev_aid = d.get("article_id") or pmid
            if eev_aid and str(eev_aid).startswith("EEV-"):
                eev_aid = str(eev_aid)[4:]
            eev = eev_by_aid.get(str(eev_aid or ""))
            if eev and eev.get("license_name"):
                lic_name = eev.get("license_name")
                lic_src = "osf-ecoevorxiv"
        # --- clasificar ---
        if lic_src == "pmc" and lic_code:
            cls = classify_pmc(lic_code)
        elif lic_src == "osf-ecoevorxiv" and lic_name:
            cls = classify_osf(lic_name)
        else:
            cls = "unk"
        stats[cls] += 1
        stats["total"] += 1
        by_src[src] = by_src.get(src, 0) + 1
        # hash
        h = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
        rec = {
            "doc_id": f"{src}:{pmid or (d.get('article_id') or 'x')}:{h[:12]}",
            "pmid": int(pmid) if pmid.isdigit() else None,
            "pmcid": pmcid,
            "article_id": d.get("article_id"),
            "source": src,
            "license_code": lic_code,
            "license_name": lic_name,
            "license_source": lic_src,
            "classification": cls,
            "text_chars": len(d["text"]),
            "text_sha256": h,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if stats["total"] % 200000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {stats['total']} docs", flush=True)

el = time.time() - t0
print(f"DONE {stats['total']} docs in {el/60:.1f}min -> {out}")
print("clasificacion:", {k: stats[k] for k in ("open", "train", "unk")})
print("por fuente:", by_src)