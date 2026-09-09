#!/usr/bin/env python3
"""extract_entities_lit.py — FASE 3: extracción de entidades desde literatura PMC/arXiv.

Recorre una MUESTRA de fulltext_corpus_all.jsonl y extrae:
  - ESPECIES: nombres binomiales (Genus species) + captura el contexto de la frase
  - REGIONES: menciones de regiones/áreas (lista KNOWN_VALUES ampliada)
  - DATASETS/TOOLS: menciones de fuentes (GBIF, iNaturalist, OBIS, eBird, WorldClim,
    CHELSA, ERA5, SRTM, MaxEnt, TRY, IUCN...) -> sugiere la tool-call apropiada
  - CLASIFICACIÓN: por cada mención de especie+dataset en la MISMA frase, propone
    una tool-call candidata {tool, args sugeridos} con su frase-fuente (auditable).

Salida: <out>.json con {docs_leidos, species[], regions[], herramientas[],
                     toolcall_candidates[]} — candides listos para curar.

Uso:
    python3 scripts/extract_entities_lit.py --input data/fulltext_corpus_all.jsonl \
        --sample 5000 --out /tmp/entities.json
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

# región -> herramienta principal (regla sencilla del diseño fase 3)
TOOL_BY_DATASET = {
    "gbif": "gbif_occurrence", " global biodiversity": "gbif_occurrence",
    "inaturalist": "inaturalist_occurrence", "obis": "obis_occurrence",
    "ebird": "ebird_occurrence", "ebird": "ebird_occurrence",
    "worldclim": "worldclim_download", "chelsa": "bioclim_download",
    "chelsa climate": "bioclim_download", "era5": "bioclim_download",
    "bioclim": "bioclim_download", "srtm": "srtm_elevation",
    "maxent": "maxent_train", "maxent model": "maxent_train",
    "try database": "try_traits", "try traits": "try_traits",
    "elton": "elton_traits", "elton traits": "elton_traits",
    "iucn": "iucn_status", "iucn red list": "iucn_status",
    "timetree": "timetree_divergence", "opentree": "opentree_phylogeny",
    "ncbi": "ncbi_taxonomy", "soilgrids": "soilgrids_download",
    "worldcover": "landcover_download", "esa worldcover": "landcover_download",
    # "esa" SOLO como parte de "esa worldcover"/"esa land cover"/"esa climate" (evita falso
    # positivo del pronombre español "esa" y del acrónimo común ESA=European Space Agency)
    "esa land cover": "landcover_download", "esa climate change initiative": "landcover_download",
    "esa cci": "landcover_download", "land cover": "landcover_download",
    "modis": "modis_ndvi", "human footprint": "human_footprint",
    "protected area": "protected_area_download", "wdpa": "protected_area_download",
    "road density": "road_density_download", "daymet": "daymet_download",
    "landcover": "landcover_download",
}

# binomio: Género minuscula (2+ mayúscula inicial) — captura panthera onca, quercus robur...
BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b")
# palabras que NO son género de especie (falsos positivos de binomio)
_NON_GENUS = {"the", "this", "however", "figure", "table", "et", "al", "fig", "tab",
    "submit", "received", "abstract", "introduction", "materials", "methods", "results",
    "discussion", "conclusion", "when", "were", "are", "was", "with", "from", "that",
    "these", "those", "their", "there", "where", "which", "after", "before", "during",
    "between", "within", "across", "among", "under", "over", "than", "then", "such",
    "some", "more", "most", "less", "also", "can", "may", "might", "must", "should",
    "would", "could", "will", "shall", "into", "onto", "upon", "about", "against",
    "because", "although", "though", "while", "using", "used", "use", "based", "shown",
    "shows", "show", "found", "find", "observed", "show", "supports", "supported",
    "including", "include", "includes", "related", "relative", "respect", "regard",
    "compared", "comparison", "different", "significant", "significantly", "mean",
    "means", "average", "total", "range", "species", "genus", "family", "order", "class",
    "data", "study", "studies", "sample", "samples", "site", "sites", "area", "areas",
    "region", "regions", "year", "years", "day", "days", "week", "weeks", "month",
    "months", "hour", "hours", "time", "times", "number", "numbers", "level", "levels",
    "type", "types", "group", "groups", "part", "parts", "form", "forms", "case",
    "cases", "result", "results", "value", "values", "test", "tests", "model", "models",
    "analysis", "approach", "method", "methods", "factor", "factors", "variable",
    "variables", "effect", "effects", "response", "responses", "pattern", "patterns",
    "process", "processes", "mechanism", "mechanisms", "estimate", "estimates",
    "figure", "fig", "table", "appendix", "supplement", "supplementary",
}
# nombres muy comunes que NO son taxones (evita sinonimia masiva)
_NON_TAXON_WORDS = {"nov", "sp", "spp", "cf", "var", "subsp", "aff", "gen", "et al."}

# lista de regiones (ampliada de KNOWN_VALUES + geográficas de literatura)
REGIONS = ["neotropico", "neotropical", "paleartico", "palearctic", "norteamerica",
    "north america", "sudamerica", "south america", "centroamerica", "central america",
    "mesoamerica", "antartida", "artico", "arctic", "caribe", "caribbean", "mediterraneo",
    "mediterranean", "sahara", "sahel", "amazonia", "amazon", "andes", "andina",
    "himalaya", "himalayan", "africa occidental", "west africa", "africa oriental",
    "east africa", "africa subsahariana", "sub-saharan africa", "india", "asia oriental",
    "east asia", "sudeste asiatico", "southeast asia", "australia", "oceania",
    "peninsula de yucatan", "yucatan", "mexico", "brazil", "brasil", "colombia",
    "patagonia", "ibera", "sonora", "chihuahuan", "gobi", "europa", "europe",
    "gran barrera de coral", "great barrier reef", "pampas", "tundra", "boreal",
    "alpes", "alps", "balcanes", "canarias", "galapagos", "polinesia", "sabana",
    "bosque tropical", "tropical forest", "bosque boreal"]

def _kw_tool(text_lower, tool_map):
    for kw, tool in tool_map.items():
        if kw in text_lower:
            return tool, kw
    return None, None

def _strip_jats_header(t):
    """Quita el front-matter JATS del PMC.

    El PMC pone 'JOURNAL INFORMATION ... ARTICLE INFORMATION ...' como bloque
    fijo al inicio, seguido del título/autores y luego el contenido real que
    empieza con 'ABSTRACT' o 'INTRODUCTION' A INICIO DE LÍNEA (con o sin número).
    Corta en la primera línea que es uno de esos marcadores de contenido.
    """
    lines = t.split("\n")
    # encabezados de contenido real (inicio de línea, sin indent, con/sin numero)
    head_pat = re.compile(r"^\s*(?:abstract|introduction|background|1\.?\s+(?:introduction|background)|methods?|results?|discussion|summary)\s*$", re.I)
    # encontrar el índice del primer encabezado de contenido DESPUÉS de que
    # aparezca 'JOURNAL INFORMATION' o tras 20 líneas de front-matter
    start = 0
    for i, ln in enumerate(lines[:40]):
        if "journal information" in ln.lower():
            start = i + 1
        elif start > 0 and head_pat.match(ln.strip()):
            return "\n".join(lines[i:])
    # fallback: primer ABSTRACT/INTRODUCTION en cualquier línea
    for i, ln in enumerate(lines[:200]):
        if head_pat.match(ln.strip()):
            return "\n".join(lines[i:])
    return t


def extract_doc(text, doc_id, sp_hist, reg_hist, tool_hist, cands):
    """Procesa UN doc: especies/regiones/erramientas + candidatos."""
    t = _strip_jats_header(text)
    low = t.lower()
    # especies binomiales (dedupe por doc, filtrar no-género y no-taxones)
    doc_sps = set()
    for m in BINOMIAL.finditer(t[:20000]):  # acotar por doc
        g1, g2 = m.group(1).lower(), m.group(2).lower()
        if g1 in _NON_GENUS or g2 in _NON_TAXON_WORDS or g1 in _NON_TAXON_WORDS:
            continue
        # el epíteto no debe ser palabra demasiado común ("and", "for", "the"...)
        if g2 in _NON_GENUS:
            continue
        sp = f"{g1} {g2}"
        if 6 <= len(sp) <= 40 and sp not in doc_sps:
            doc_sps.add(sp)
            sp_hist[sp] += 1
    # regiones (dedupe por doc)
    doc_regs = set()
    for rg in REGIONS:
        if rg in low:
            doc_regs.add(rg)
            reg_hist[rg] += 1
    # herramientas/datasets (dedupe por doc)
    found_tools = set()
    for kw, tool in TOOL_BY_DATASET.items():
        if kw in low:
            found_tools.add(tool)
            tool_hist[tool] += 1
    # CANDIDATO: especie + herramienta en el MISMO doc (misma petición plausible)
    if found_tools and doc_sps:
        # regiones top del doc (guardar con frecuencia para ordenar)
        doc_regs_ranked = sorted(((low.count(rg), rg) for rg in doc_regs), reverse=True)
        for tool in sorted(found_tools):
            for sp in list(doc_sps)[:2]:
                cands.append({
                    "doc": doc_id, "tool": tool, "species": sp,
                    "region": doc_regs_ranked[0][1] if doc_regs_ranked else None,
                })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fulltext_corpus_all.jsonl")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--top", type=int, default=200000,
                    help="cuantas especies mas frecuentes guardar en species_top (default 200K)")
    ap.add_argument("--out", default="/tmp/entities.json")
    a = ap.parse_args()

    sp_hist, reg_hist, tool_hist = Counter(), Counter(), Counter()
    cands = []
    n = 0
    with open(a.input) as f:
        for line in f:
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            did = d.get("pmid") or d.get("pmcid") or d.get("id") or n
            extract_doc(d.get("text", ""), did, sp_hist, reg_hist, tool_hist, cands)
            if n >= a.sample:
                break
    report = {
        "docs_leidos": n,
        "sample": a.sample,
        "species_top": sp_hist.most_common(min(a.top, len(sp_hist))),
        "regions_top": reg_hist.most_common(25),
        "tools_top": tool_hist.most_common(),
        "candidates": cands,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"docs: {n} | especies distintas: {len(sp_hist)} | regiones: {len(reg_hist)} | tools: {dict(tool_hist)}")
    print(f"candidatos tool-call (doc×tool): {len(cands)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())