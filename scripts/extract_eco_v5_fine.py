#!/usr/bin/env python3
"""extract_eco_v5_fine.py — FASE 3: subcorpus ECO GENUINO por domain_fine del v5.

El hallazgo 2026-09-09: los dominios GRUESOS del v7 (12 ILIKE) no separan eco de
biomed (el 63% "eco" sigue siendo patógenos clínicos). La etiqueta FINA del v5
(fine_label_v5: MeSH + texto, ~40 dominios, ver skill) sí separa. Este script
extrae solo los dominios finos ecológicos de campo para la minería de
tool-calls (especies/regiones reales).

Dominios finos ECO (excluye metagenomics/microbiology/general/biomed):
sdm, community_ecology, population_ecology, conservation, climate_ecology,
landscape_ecology, macroecology, evolutionary, phylogeography, plant_biology,
marine, soil_ecology, paleoecology, animal_behavior, ecoevo,
disease_ecology, ecotoxicology.

Uso: python3 scripts/extract_eco_v5_fine.py --in train_corpus_v5.jsonl --out eco_v5_fine.jsonl
"""
import argparse, json, sys

ECO_FINE = {
    "sdm", "community_ecology", "population_ecology", "conservation",
    "climate_ecology", "landscape_ecology", "macroecology", "evolutionary",
    "phylogeography", "plant_biology", "marine", "soil_ecology",
    "paleoecology", "animal_behavior", "ecoevo", "disease_ecology",
    "ecotoxicology",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = kept = 0
    dist = {}
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            df = d.get("domain_fine")
            if df in ECO_FINE:
                fo.write(json.dumps({"text": d.get("text", ""), "pmid": d.get("pmid"),
                                     "domain_fine": df}, ensure_ascii=False) + "\n")
                dist[df] = dist.get(df, 0) + 1
                kept += 1
            if n % 300000 == 0:
                print(f"  {n} leidos, {kept} eco-fino", flush=True)
    print(f"DONE: {n} leidos, {kept} eco-fino ({(100*kept/max(n,1)):.2f}%)")
    for k in sorted(dist, key=lambda x: -dist[x]):
        print(f"  {dist[k]:7d}  {k}")
    return 0

if __name__ == "__main__":
    sys.exit(main())