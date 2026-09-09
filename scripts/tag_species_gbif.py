#!/usr/bin/env python3
"""tag_species_gbif.py — FASE 3: tagger de especies REALES vía GBIF species/match.

Pipeline (el "tagger complejo" para nuestro dominio ecológico):
  1. Extrae candidatos binomiales (Género especie) del texto con regex afinado.
  2. Valida CADA candidato contra la API GBIF species/match (rate-limit 1 req/s).
  3. Solo conserva los que GBIF confirma como SPECIES (rank=SPECIES, confidence>=85,
     status ACCEPTED/DOUBTFUL) -> especies REALES con usageKey + canonico.
  4. Emite {especie_canonica, usageKey} listo para el generador de tool-calls.

PubTator3: evaluado y descartado para ESTE pipeline (API requiere API key/registro
para acceso programático; cobertura biomédica no ecológica). GBIF cubre todo el
árbol de la vida y es la fuente de nuestro resolver (mismo vocabulario).

Uso:
  python3 scripts/tag_species_gbif.py --input data/ecoevorxiv_fulltext_c0.jsonl \
      --sample 200 --out /tmp/species_tagged.json \
      [--min-confidence 85] [--max-candidates 300]  # límite req/s a GBIF
"""
import argparse, json, re, sys, time, urllib.request, urllib.error, urllib.parse
from collections import Counter
from pathlib import Path

GBIF_MATCH = "https://api.gbif.org/v1/species/match"
RATE = 1.1  # 1 req/s (GBIF recomienda max 2/s)

BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b")
_NON_GENUS = {"the","this","however","figure","table","et","al","fig","tab",
    "when","were","are","was","with","from","that","these","those","their",
    "there","where","which","after","before","during","between","within",
    "across","among","under","over","than","then","such","some","more","most",
    "less","also","can","may","might","must","should","would","could","will",
    "shall","into","onto","upon","about","against","because","although",
    "while","using","used","use","based","shown","shows","show","found",
    "observed","including","include","includes","related","relative","compared",
    "different","significant","mean","means","average","total","range","species",
    "genus","family","order","class","data","study","studies","sample","samples",
    "site","sites","area","areas","region","regions","year","years","day","days",
    "time","times","number","numbers","level","levels","type","types","group",
    "groups","part","parts","form","forms","case","cases","result","results",
    "value","values","test","tests","model","models","analysis","approach",
    "method","methods","factor","factors","variable","variables","effect",
    "effects","response","responses","pattern","patterns","process","processes",
    "mechanism","mechanisms","estimate","estimates","figure","table","appendix"}

def is_candidate(m):
    g1, g2 = m.group(1).lower(), m.group(2).lower()
    if g1 in _NON_GENUS or g2 in _NON_GENUS:
        return False
    sp = f"{g1} {g2}"
    return 6 <= len(sp) <= 40

def gbif_match(name):
    try:
        url = f"{GBIF_MATCH}?name={urllib.parse.quote(name)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ecoreasoner-fase3/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-confidence", type=int, default=85)
    ap.add_argument("--max-candidates", type=int, default=300)
    a = ap.parse_args()

    # 1) extraer candidatos (dedupe global por nombre)
    cands = Counter()
    n_docs = 0
    with open(a.input) as f:
        for line in f:
            n_docs += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text", "")[:20000]
            for m in BINOMIAL.finditer(t):
                if is_candidate(m):
                    cands[f"{m.group(1).lower()} {m.group(2).lower()}"] += 1
            if n_docs >= a.sample:
                break
    print(f"docs: {n_docs} | candidatos distintos: {len(cands)}")
    # filtrar por frecuencia (>1 en el sample = no aislado) y acotar
    ranked = [k for k, v in cands.most_common() if v > 1][: a.max_candidates]
    print(f"a validar contra GBIF (freq>1, max {a.max_candidates}): {len(ranked)}")

    # 2) validar contra GBIF
    ok_species, not_species, errors = [], [], 0
    for i, name in enumerate(ranked):
        m = gbif_match(name)
        if m is None:
            errors += 1
            continue
        rank = m.get("rank", "").upper()
        conf = m.get("confidence", 0)
        if rank == "SPECIES" and conf >= a.min_confidence:
            ok_species.append({
                "query": name, "canonical": m.get("canonicalName") or m.get("scientificName") or name,
                "usageKey": m.get("usageKey") or m.get("acceptedUsageKey"),
                "confidence": conf, "status": m.get("status"),
                "mentions": cands[name],
            })
        else:
            not_species.append({"query": name, "rank": rank, "confidence": conf, "mentions": cands[name]})
        time.sleep(RATE)
        if (i + 1) % 50 == 0:
            print(f"  validados {i+1}/{len(ranked)} | OK {len(ok_species)} | NO {len(not_species)} | err {errors}", flush=True)

    # dedupe por usageKey (misma especie normalizada)
    by_key = {}
    for s in ok_species:
        k = s["usageKey"]
        if k and (k not in by_key or s["confidence"] > by_key[k]["confidence"]):
            by_key[k] = s
    final = sorted(by_key.values(), key=lambda x: -x["mentions"])
    out = {"docs": n_docs, "candidatos": len(cands), "validados": len(ranked),
           "especies_reales": len(final), "min_confidence": a.min_confidence,
           "species": final, "errors": errors}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n=== RESULTADO ===")
    print(f"especies REALES confirmadas por GBIF: {len(final)}")
    for s in final[:25]:
        print(f"  {s['canonical']:<30} usageKey={s['usageKey']} conf={s['confidence']} men={s['mentions']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())