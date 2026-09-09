#!/usr/bin/env python3
"""tag_entities_fulltext.py — FASE 3: curar las 1.32M especies candidatas del
full-text completo (97K docs PMC) validando contra GBIF species/match.

Reutiliza la función gbif_match de tag_species_gbif.py (mismo rate-limit).
Filtro previo con stoplist inglesa amplia derivada EMPÍRICAMENTE del top de
frecuencias del propio corpus (frases "for example" / "error bars" / "article
version" no son binomiales) para no quemar la cuota de GBIF en frases.

Uso:
  python3 scripts/tag_entities_fulltext.py --cands docs/results/entities_fulltext_corpus.json \
      --out docs/results/species_ecofulltext_tagged.json [--max-candidates 600] [--min-freq 3]
"""
import argparse, json, sys, time, urllib.request, urllib.parse
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tag_species_gbif import gbif_match

RATE = 1.1  # 1 req/s (GBIF)

# palabras que abren frases no-taxonómicas del PMC (empírico del corpus + inglés común)
STOP_WORDS = set("""
for this however article additional electronic center taken first last statistical
material our publication care all western error ethics research food only scale print
health science america africa europe during using used use based shown shows show found
observed including include includes related compared different significant mean means
average total range data study studies sample samples site sites area areas year years
day days time times number numbers level levels type types group groups part parts form
forms case cases result results value values test tests model models analysis approach
method methods factor factors variable variables effect effects response responses
pattern patterns process processes mechanism mechanisms estimate estimates figure table
appendix supplementary content copyright journal information received accepted published
online available access obtained performed conducted carried undertaken employed applied
developed created designed described presented reported analysed analyzed examined
investigated explored assessed evaluated estimated calculated measured compared assessed
increased decreased reduced enhanced improved showed shown whereas although because while
when where which those these their there here both each every other many much more most
less also can may might must should would could will shall into onto upon about against
between within across among under over than then such some what who whom whose how why
new old good bad large small high low great little long short same different second
third above below near far left right bottom top within without after before during
between among related concerning regarding toward following given shown listed below
above indicates suggest suggests demonstrating demonstrates demonstrating
""".split())

# predecir ruido: si la palabra 1 o 2 está en stopwords -> descartar
def passes(sp):
    g1, g2 = sp.split()
    if g1 in STOP_WORDS or g2 in STOP_WORDS:
        return False
    if len(g1) < 2 or len(g1) > 20 or len(g2) < 3 or len(g2) > 20:
        return False
    if any(ch.isdigit() for ch in sp):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="entities_fulltext_corpus.json (species_top)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=600)
    a = ap.parse_args()

    d = json.load(open(a.cands))
    freq = dict(d["species_top"])
    ranked = []
    for sp, n in sorted(freq.items(), key=lambda kv: -kv[1]):
        if n < a.min_freq:
            break
        if " " not in sp or not passes(sp):
            continue
        ranked.append((sp, n))
        if len(ranked) >= a.max_candidates:
            break
    print(f"candidatas validadas contra GBIF (freq>={a.min_freq}, stop-filtered): {len(ranked)}",
          flush=True)

    ok_species, not_species, errors = [], [], 0
    for i, (name, n) in enumerate(ranked):
        m = gbif_match(name)
        if m is None:
            errors += 1
            continue
        rank = m.get("rank", "").upper()
        conf = m.get("confidence", 0)
        if rank == "SPECIES" and conf >= 85:
            ok_species.append({
                "query": name, "canonical": m.get("canonicalName") or m.get("scientificName") or name,
                "usageKey": m.get("usageKey") or m.get("acceptedUsageKey"),
                "confidence": conf, "status": m.get("status"),
                "mentions": n,
            })
        else:
            not_species.append({"query": name, "rank": rank, "confidence": conf, "mentions": n})
        time.sleep(RATE)
        if (i + 1) % 50 == 0:
            print(f"  validados {i+1}/{len(ranked)} | OK {len(ok_species)} | NO {len(not_species)} | err {errors}", flush=True)

    by_key = {}
    for s in ok_species:
        k = s["usageKey"]
        if k and (k not in by_key or s["confidence"] > by_key[k]["confidence"]):
            by_key[k] = s
    final = sorted(by_key.values(), key=lambda x: -x["mentions"])
    # filtro de ruido NO ecológico (modelos lab / patógenos humanos / vectores)
    DROP = {"homo sapiens", "drosophila melanogaster", "poecilia reticulata",
            "escherichia coli", "zika virus", "aedes aegypti", "neisseria gonorrhoeae"}
    final = [s for s in final if s["canonical"].lower() not in DROP]
    out = {"docs": d["docs_leidos"], "candidatos": len(freq), "validados": len(ranked),
           "especies_reales": len(final), "min_freq": a.min_freq,
           "min_confidence": 85, "species": final, "errors": errors,
           "drop_no_ecologico": sorted(DROP - {""})}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n=== RESULTADO === especies REALES confirmadas por GBIF: {len(final)}")
    for s in final[:30]:
        print(f"  {s['canonical']:<32} usageKey={s['usageKey']} conf={s['confidence']} men={s['mentions']}")
    print("drop:", out["drop_no_ecologico"])
    return 0

if __name__ == "__main__":
    sys.exit(main())