#!/usr/bin/env python3
"""build_toolcalls_500.py — FASE 3: genera el dataset de 500 tool-calls desde
una lista de especies REALES y regiones reales.

Fuentes de especies (todas ya validadas como reales, con proveniencia):
  1. GBIF-tagged de ecoevorxiv (docs/results/species_tagged_ecoevorxiv.json):
     27 especies confirmadas por GBIF species/match (conf>=97).
  2. Gold limpio de literatura (data/l1/toolcalls_lit_gold.jsonl): las
     ~76 especies únicas del gold validado de Devin.
  3. (En Fase 3 real) + especies del corpus ecológico.

Regiones reales: lista curada (coinciden con las ya usadas en el gold válido).

El generador produce pares {prompt, gold, source} con variedad:
  - distribucion de tools: ~55% gbif, ~25% bioclim, ~12% maxent, resto iucn/srtm/inat
  - especies/regiones muestreadas sin repeticion inmediata
  - prompts naturales en espanol (plantillas variadas, no 1 sola)
  - source: nota de la especie (proveniencia)

Output: data/l1/toolcalls_fase3_500.jsonl
Uso:
  python3 scripts/build_toolcalls_500.py --out data/l1/toolcalls_fase3_500.jsonl --seed 7331
"""
import argparse, json, random, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

# regiones reales (las mismas del gold valido + variedad + las del subcorpus eco-fino)
REGIONS = [
    "neotropico", "paleartico", "norteamerica", "sudamerica", "centroamerica",
    "mesoamerica", "amazonia", "andes", "patagonia", "caribe", "mediterraneo",
    "sahara", "sahel", "africa oriental", "africa occidental", "africa subsahariana",
    "madagascar", "india", "sudeste asiatico", "himalaya", "asia oriental",
    "australia", "oceania", "artico", "peninsula de yucatan", "mexico", "brazil",
    "colombia", "europa", "alpes", "balcanes", "canarias", "galapagos",
    "gran barrera de coral", "tundra", "boreal", "bosque tropical", "sabana",
    # + regiones top del subcorpus eco-fino (entities_eco_fine_v5.json, 2026-09-09)
    "ibera", "neotropical", "tropical forest", "caribbean", "southeast asia",
    "west africa", "andina", "bosque boreal", "north america", "south america",
    "east asia", "mediterranean", "pampas",
]

# capas para maxent_train (reales del stack HPC)
LAYERS = ["bio1,bio12,bio19", "bio2,bio7,bio15", "bio1,bio4,bio12,bio15,bio17",
          "bio5,bio6,bio13,bio14", "bio8,bio9,bio18,bio19", "chelsa_19",
          "current_bio1_19", "future_2050_bio1_19"]

def load_species():
    """Unifica especies reales con fuente. Devuelve lista de dicts {name, source}."""
    sps = {}
    # 1) gold devin (data/l1/toolcalls_lit_gold.jsonl)
    gf = ROOT / "data/l1/toolcalls_lit_gold.jsonl"
    if gf.exists():
        for line in open(gf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            for g in (r.get("gold") or []):
                sp = (g.get("args") or {}).get("species")
                if sp:
                    sps[sp.lower()] = {"name": sp, "source": "gold-devin"}
    # 2) GBIF-tagged ecoevorxiv (excluir Homo sapiens, Zika, modelos lab)
    tj = ROOT / "docs/results/species_tagged_ecoevorxiv.json"
    EXCL = {"homo sapiens", "zika virus", "drosophila melanogaster", "poecilia reticulata"}
    if tj.exists():
        d = json.load(open(tj))
        for s in d.get("species", []):
            n = s.get("canonical", "").lower()
            if n not in EXCL:
                sps[n] = {"name": s.get("canonical"), "source": f"gbif-ecoevorxiv (usageKey {s.get('usageKey')})"}
    # 3) GBIF-tagged subcorpus ECO-FINO v5 (species_eco_fine_tagged.json, 2026-09-09).
    #    Excluir patogenos clinicos y modelos de laboratorio (aunque GBIF los valide
    #    como SPECIES) — no son objetivo de tool-calls ecologicas de campo.
    ef = ROOT / "docs/results/species_eco_fine_tagged.json"
    EXCL_CLIN = {
        "staphylococcus aureus", "pseudomonas aeruginosa", "saccharomyces cerevisiae",
        "mycobacterium tuberculosis", "klebsiella pneumoniae", "candida albicans",
        "bacillus subtilis", "caenorhabditis elegans", "acinetobacter baumannii",
        "streptococcus pneumoniae", "salmonella enterica", "helicobacter pylori",
        "enterococcus faecalis", "listeria monocytogenes", "plasmodium falciparum",
        "danio rerio", "staphylococcus epidermidis", "enterococcus faecium",
        "vibrio cholerae", "toxoplasma gondii", "haemophilus influenzae",
        "streptococcus mutans", "streptococcus pyogenes",
        "mus musculus", "porphyromonas gingivalis", "bacillus cereus",
        "chlamydia trachomatis", "aedes albopictus", "anopheles gambiae",
        "clostridium difficile", "borrelia burgdorferi", "cryptococcus neoformans",
        "trypanosoma cruzi", "clostridioides difficile", "campylobacter jejuni",
        "neisseria meningitidis", "aspergillus niger", "legionella pneumophila",
        "pseudomonas putida", "mycoplasma pneumoniae", "fusobacterium nucleatum",
        "schizosaccharomyces pombe", "botrytis cinerea", "salmonella typhimurium",
        "streptococcus agalactiae", "lactococcus lactis", "pseudomonas fluorescens",
        "agrobacterium tumefaciens", "clostridium perfringens", "bacteroides fragilis",
        "serratia marcescens", "enterobacter cloacae", "pan troglodytes",
        "rattus norvegicus", "arabidopsis thaliana", "aspergillus fumigatus",
        "xenopus laevis",
    }
    if ef.exists():
        d = json.load(open(ef))
        for s in d.get("species", []):
            n = s.get("canonical", "").lower()
            if n not in EXCL and n not in EXCL_CLIN:
                sps[n] = {"name": s.get("canonical"),
                          "source": f"gbif-eco-fino-v5 (usageKey {s.get('usageKey')})"}
    return list(sps.values())

# plantillas de prompt por tool (variadas, naturales, en espanol; solo llamada directa)
TEMPLATES = {
    "gbif_occurrence": [
        "Busca registros de presencia de la especie {species} en {region} para un modelo de nicho.",
        "Descarga observaciones de {species} en {region} para analizar su distribucion.",
        "Consulta los registros de ocurrencia de {species} en la region {region}.",
        "Obten los datos de presencia de {species} en {region} desde GBIF.",
        "Localiza registros de {species} en {region} para estudiar su area de distribucion.",
    ],
    "bioclim_download": [
        "Descarga las capas bioclimaticas para la region {region} en el anio {year}.",
        "Obtiene datos climaticos (CHELSA/ERA5) de {region} para {year}.",
        "Descarga las variables bioclimaticas de {region} del periodo {year}.",
        "Consigue las capas de clima (bio01-bio19) para {region}, anio {year}.",
    ],
    "maxent_train": [
        "Entrena un modelo de nicho (MaxEnt) para {species} usando las capas {layers}.",
        "Crea un modelo de distribucion de {species} con las variables {layers}.",
        "Entrena MaxEnt para {species} con las capas {layers} y datos de presencia.",
        "Modela el nicho de {species} con las variables {layers}.",
    ],
    "iucn_status": [
        "Consulta el estado de conservacion de {species} en la Lista Roja de la UICN.",
        "Busca la categoria de amenaza de {species} segun la IUCN.",
    ],
    "srtm_elevation": [
        "Descarga la capa de elevacion SRTM para la region {region} a {res} de resolucion.",
        "Obtiene los datos de elevacion (DEM) de {region} con resolucion {res}.",
    ],
    "inaturalist_occurrence": [
        "Busca observaciones de {species} en iNaturalist para la region {region}.",
        "Descarga avistamientos de {species} en {region} desde iNaturalist.",
    ],
    "try_traits": [
        "Consulta los rasgos funcionales de {species} en la base de datos TRY.",
        "Busca los rasgos morfologicos de {species} en TRY.",
    ],
    "ncbi_taxonomy": [
        "Consulta la taxonomia de {species} en NCBI Taxonomy.",
        "Busca la clasificacion taxonomica de {species} en NCBI.",
    ],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/l1/toolcalls_fase3_500.jsonl"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7331)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    sps = load_species()
    if len(sps) < 20:
        print(f"[fatal] solo {len(sps)} especies (necesito >=20)", file=sys.stderr)
        sys.exit(2)
    print(f"especies disponibles: {len(sps)}")

    # distribucion de tools (~55/25/12/8)
    tools = (["gbif_occurrence"] * 275 + ["bioclim_download"] * 125 +
             ["maxent_train"] * 60 + ["iucn_status", "srtm_elevation",
             "inaturalist_occurrence", "try_traits", "ncbi_taxonomy"] * 8)
    rng.shuffle(tools)
    tools = tools[: a.n]

    # muestreo pareado de especies (rotar sin repeticion inmediata)
    sp_cycle = [s["name"] for s in sps]
    rng.shuffle(sp_cycle)
    rows = []
    sp_used = Counter()
    di = 0
    for ti, tool in enumerate(tools):
        # elegir especie: rotatoria con repeticion minima
        sp = sp_cycle[di % len(sp_cycle)]
        di += 1
        sp_used[sp] += 1
        region = rng.choice(REGIONS)
        year = str(rng.choice([2010, 2015, 2018, 2020, 2021, 2022, 2023]))
        res = rng.choice(["30m", "90m"])
        layers = rng.choice(LAYERS)
        # build prompt
        if tool == "gbif_occurrence":
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp, region=region)
            args = {"species": sp, "region": region}
        elif tool == "bioclim_download":
            prompt = rng.choice(TEMPLATES[tool]).format(region=region, year=year)
            args = {"region": region, "year": year}
        elif tool == "maxent_train":
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp, layers=layers)
            args = {"species": sp, "layers": layers}
        elif tool == "iucn_status":
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp)
            args = {"species": sp}
        elif tool == "srtm_elevation":
            prompt = rng.choice(TEMPLATES[tool]).format(region=region, res=res)
            args = {"region": region, "resolution": res}
        elif tool == "inaturalist_occurrence":
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp, region=region)
            args = {"species": sp, "region": region}
        elif tool == "try_traits":
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp)
            args = {"species": sp}
        else:  # ncbi_taxonomy
            prompt = rng.choice(TEMPLATES[tool]).format(species=sp)
            args = {"species": sp}
        src = next((s["source"] for s in sps if s["name"].lower() == sp.lower()), "especie real")
        rows.append({"prompt": prompt, "gold": [{"tool": tool, "args": args}],
                     "source": f"Fase3-500 | {src} | region={region}"})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # reporte
    dist = Counter(r["gold"][0]["tool"] for r in rows)
    print(f"generadas: {len(rows)} -> {a.out}")
    print("dist: dict(dist)=", dict(dist))
    print("especies usadas:", len(sp_used))
    return 0

if __name__ == "__main__":
    sys.exit(main())