#!/usr/bin/env python3
"""annotate_pubtator3.py — FASE 3: anotar especies con PubTator3 (NCBI) por pmid.

Usa la API real (verificada 2026-09-09):
  GET https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=A,B,C

Extrae las anotaciones de tipo 'Species' (el tagger AIONER/GNormPlus de NCBI,
normalizadas a NCBI Taxonomy) con:
  - text:  la mención en el texto
  - id:    el NCBITaxonId (identifier) — la normalización de NCBI
  - passage: title/abstract

Así se obtienen especies REALES del texto con su taxonomía NCBI estándar,
sin ruido de regex (los 'western blot' no se anotan como especie por AIONER).

Uso:
  python3 scripts/annotate_pubtator3.py --pmids-file /tmp/pmids.txt \
      --out /tmp/annot_pt3.jsonl [--limit 100]
"""
import argparse, json, re, sys, time, urllib.request, urllib.error
from collections import Counter
from pathlib import Path

API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
RATE = 0.4  # respetar límite NCBI (3 req/s) — espaciado prudente

def fetch(pmids):
    url = f"{API}?pmids={','.join(map(str, pmids))}"
    req = urllib.request.Request(url, headers={"User-Agent": "ecoreasoner-fase3/1.0 (contact: alrobles)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())

def extract_species(doc):
    """Extrae las especies anotadas de un doc PubTator3 biocjson."""
    pmid = doc.get("id")
    title = ""
    species = Counter()  # (ncbi_id|name) -> count
    for passage in doc.get("passages", []):
        ptype = (passage.get("infons") or {}).get("type", "")
        if ptype == "title" and not title:
            title = passage.get("text", "")[:300]
        for ann in passage.get("annotations", []):
            inf = (ann.get("infons") or {})
            typ = str(inf.get("type", ""))
            if typ.lower() == "species":
                text = ann.get("text", "")
                ncid = inf.get("identifier") or ""
                # identifier en biocjson es el NCBITaxonId (número) o vacío
                name = inf.get("name", "") or (text if not ncid else ncid)
                key = f"{ncid}|{name or text}"
                species[key] += 1
    out = []
    for key, cnt in species.most_common():
        ncid, name = key.split("|", 1)
        out.append({"ncbi_id": ncid, "name": name, "count": cnt})
    return {"pmid": pmid, "title": title, "species": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=40)
    a = ap.parse_args()
    pmids = [int(l.strip()) for l in open(a.pmids_file) if l.strip().isdigit()][: a.limit]
    print(f"pmids a anotar: {len(pmids)}")
    n_ok = n_err = 0
    with open(a.out, "w") as fo:
        for i in range(0, len(pmids), a.batch):
            batch = pmids[i : i + a.batch]
            try:
                d = fetch(batch)
                for doc in d.get("PubTator3", []):
                    rec = extract_species(doc)
                    if rec["species"]:
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_ok += 1
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
                n_err += 1
                print(f"[warn] batch {i}: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(RATE)
    print(f"docs con especies: {n_ok} | batch err: {n_err} -> {a.out}")

if __name__ == "__main__":
    sys.exit(main())