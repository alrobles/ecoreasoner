#!/usr/bin/env python3
"""
fine_label_v5.py — Etiqueta fina del corpus v5 usando MeSH de PubMed (NIH).

Une mesh_terms por pmid (100% de los docs v5 tienen pmid) y deriva domain_fine
con taxonomia fina de ~40 dominios (eco + biomed + generales), con regla de
senal relativa: los marcadores clinicos fuertes (Humans/Female/Male/Adult/
Disease/Patient...) pesan hacia biomedicina; la senal ecologica domina solo
con >=2 hits y >= que la biomedica (validado 2026-08-28: falsos positivos
-> 3.7% de eco con marcadores clinicos).

JOIN via duckdb+Parquet (rapido, sin indice SQLite de 20-40 min):
  /beegfs/a474r867/litdump/pubmed/parsed/parquet/ (30.8M filas, columna mesh)
Fallback a sqlite si el parquet no esta.

Salida: reescribe train_corpus_v5.jsonl anadiendo por doc:
  - "mesh_terms": lista de terminos MeSH
  - "domain_fine": etiqueta fina derivada de MeSH
  - conserva "domain" (grueso 12 ILIKE) como referencia
NO cambia "text" ni el orden => train_ids_v5.npy Sigue siendo valido.

Uso (Slurm CPU, no login):
  python3 fine_label_v5.py --input data/train_corpus_v5.jsonl \
      --db /home/a474r867/work/pubmed/index/pubmed_fts.db \
      --out data/train_corpus_v5.jsonl --report data/v5_fine_report.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

# Guarda anti-login-node (regla de labor)
if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: fine_label_v5.py SOLO via Slurm (SLURM_JOB_ID ausente).")

# ── Taxonomia fina: ECO + BIOMED + GENERAL ──
# Patrones = terminos MeSH reales (o raices seguras), NO palabras sueltas ambiguas.
ECO = {
    "sdm": ["species distribution", "ecological niche", "habitat suitability", "maxent",
            "ecotype", "species range", "climatic niche"],
    "community_ecology": ["community ecology", "species interaction", "food web", "trophic",
            "symbiosis", "predation", "mutualism", "herbivory", "dominance (ecology)"],
    "population_ecology": ["population dynamics", "population density", "metapopulation",
            "life history", "population growth", "population structure", "birth rate"],
    "conservation": ["conservation of natural resources", "endangered species", "extinction",
            "protected area", "wildlife management", "species protection", "habitat conservation",
            "restoration ecology"],
    "climate_ecology": ["climate change", "global warming", "greenhouse effect", "sea level rise",
            "drought", "carbon cycle", "ocean warming"],
    "landscape_ecology": ["landscape ecology", "habitat fragmentation", "land use",
            "spatial analysis", "remote sensing", "corridor"],
    "macroecology": ["macroecolog", "biogeograph", "species richness", "latitudinal gradient",
            "range size", "abundance"],
    "evolutionary": ["evolution, molecular", "evolution, biological", "natural selection",
            "adaptation, biological", "speciation", "genetic drift", "fitness",
            "reproductive isolation", "selection, genetic"],
    "phylogeography": ["phylogeograph", "molecular evolution", "haplotypes", "gene flow",
            "genetic variation", "mitochondrial dna", "microsatellite"],
    "metagenomics": ["metagenom", "microbiota", "microbiome", "16s", "community composition"],
    "microbiology": ["bacteria (unicellular)", "archaea", "viruses (unicellular)",
            "fungi (unicellular)", "microbial", "bioremediation", "microbial community"],
    "plant_biology": ["plants (botany)", "botany", "crops", "forests", "trees", "pollination",
            "vegetation", "angiosperms", "photosynthesis"],
    "marine": ["marine biology", "oceans", "coral reefs", "fisheries", "aquatic organisms",
            "plankton", "estuaries"],
    "soil_ecology": ["soil", "rhizosphere", "mycorrhizae", "decomposition", "nutrient cycling",
            "sediments", "biogeochemistry", "nitrogen fixation", "soil microbiology"],
    "paleoecology": ["paleontology", "fossils", "paleoecology", "quaternary", "paleoclimatology"],
    "animal_behavior": ["animal behavior", "migration (animal)", "foraging", "territoriality",
            "social behavior", "nesting behavior", "predator-prey dynamics", "home range"],
    "ecoevo": ["phenotypic plasticity", "local adaptation", "trait evolution", "coevolution",
            "ecological genetics", "quantitative trait", "genotype-environment"],
    "disease_ecology": ["disease ecology", "zoonoses", "host-pathogen", "wildlife disease",
            "vector-borne"],
    "ecotoxicology": ["ecotoxicolog", "pollutant", "contaminant", "heavy metal", "pesticide",
            "bioaccumulation", "water quality"],
}
BIOMED = {
    "cardiovascular": ["cardiovascular diseases", "heart diseases", "myocardial infarction",
            "hypertension", "coronary", "arrhythmia", "atheroscleros"],
    "oncology": ["neoplasms", "cancer", "carcinoma", "tumor", "metastasis", "oncogenes"],
    "infectious": ["infection", "hiv", "hepatitis", "tuberculosis", "influenza", "sepsis",
            "pathogen", "vaccines", "antibiotic"],
    "immunology": ["immune system", "immunity", "antibody", "antigen", "cytokine",
            "inflammation", "immune response", "lymphocytes", "autoimmune"],
    "neurology": ["brain", "neurons", "neurolog", "stroke", "alzheimer", "parkinson",
            "seizure", "nervous system"],
    "endocrinology": ["hormones", "insulin", "thyroid", "endocrine", "diabetes", "cortisol"],
    "respiratory": ["lung", "pulmonary", "respirat", "asthma", "copd"],
    "renal": ["kidney", "renal", "nephro", "urinary"],
    "public_health": ["public health", "epidemiology", "health services", "health care",
            "maternal", "women's health", "child health", "health policy", "health education"],
    "pharmacology": ["drug therapy", "pharmacolog", "dosage", "adverse effects",
            "clinical trial", "pharmaceutical"],
    "medical_genetics": ["genetic diseases", "genetic predisposition", "mutation", "hereditary",
            "genetic testing", "variant", "genetic counseling"],
    "psychiatry": ["depression", "anxiety", "mental disorders", "psychiatric",
            "stress, psychological"],
    "nutrition": ["diet", "nutrition", "obesity", "metabolic", "body weight", "energy intake"],
    "surgery": ["surgical", "surgery", "operative", "postoperative"],
    "geriatrics": ["aged", "elderly", "geriatr"],
    "pediatrics": ["pediatric", "child development", "infant, newborn", "adolescent health"],
}
GENERAL = {
    "molecular_biology": ["dna", "rna", "protein", "gene expression", "molecular",
            "polymerase chain reaction", "nucleotide", "transcription", "translation"],
    "cell_biology": ["cell", "cellular", "apoptosis", "organelle", "membrane", "cytoplasm"],
    "genetics_general": ["genes", "chromosome", "alleles", "inheritance", "genotype",
            "phenotype", "heredity", "genetic phenomena"],
    "biochemistry": ["enzyme", "metabolism", "biochemical", "atp", "amino acid", "lipid"],
    "methods_stats": ["methods", "statistical", "algorithms", "models", "mathematical",
            "data analysis", "software", "machine learning", "computational"],
}
CLINICAL_MARKERS = ["humans", "female", "male", "adult", "clinical", "patient",
                    "disease", "therapeutic", "drug therapy", "treatment outcome", "health"]

def fine_domain(mesh_terms: str):
    """Deriva dominio fino de los terminos MeSH con regla de senal relativa."""
    return _classify(mesh_terms)

def fine_domain_text(text: str):
    """Fallback: dominio fino por patrones sobre el texto (para docs sin MeSH)."""
    return _classify(text)

def _classify(txt: str):
    """Nucleo: puntua ECO/BIOMED/GENERAL sobre un string (MeSH o texto)."""
    if not txt:
        return None
    t = txt.lower()
    eco_best, eco_score = None, 0
    for dom, pats in ECO.items():
        s = sum(1 for p in pats if p in t)
        if s > eco_score:
            eco_best, eco_score = dom, s
    bm_best, bm_score = None, 0
    for dom, pats in {**BIOMED, **GENERAL}.items():
        s = sum(1 for p in pats if p in t)
        if s > bm_score:
            bm_best, bm_score = dom, s
    clin = sum(1 for m in CLINICAL_MARKERS if m in t)
    # eco domina solo con >=2 hits y >= que biomed (o clinica debil)
    if eco_score >= 2 and (eco_score >= bm_score or clin < 3):
        return eco_best
    if bm_score >= 1:
        return bm_best
    if eco_score >= 1:
        return eco_best
    return "other"

def load_mesh_map_duckdb(pmids, parquet_glob, pmcid_csv=None):
    """JOIN por pmid via duckdb sobre Parquet (rapido, sin indice SQLite).
    pmid en el parquet es VARCHAR -> pasar strings.
    pmcid_csv: PMC-ids.csv.gz (PMID<->PMCID). Los docs pmc-v4 tienen
    "pmid"=PMCID (PMCxxxx); hay que traducir a PMID real via el CSV para
    encontrar su MeSH en el parquet PubMed.
    OPTIMIZADO: UN solo escaneo de todo el parquet (SELECT pmid, mesh),
    filtrando en Python con el set (los IN de 200K params escaneaban 10x)."""
    import duckdb
    con = duckdb.connect()
    # 1) traducir PMCID->PMID si hay CSV y hay pmids con prefijo PMC
    pmid_to_real = {}     # pmid en corpus -> pmid real en parquet
    pm_list = list(pmids)
    plain = [p for p in pm_list if not str(p).startswith("PMC")]
    pmc = [p for p in pm_list if str(p).startswith("PMC")]
    for p in plain:
        pmid_to_real[p] = p
    if pmc and pmcid_csv and os.path.exists(pmcid_csv):
        import gzip, csv
        with gzip.open(pmcid_csv, "rt", errors="ignore") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            # localizar columnas PMCID y PMID por nombre
            try:
                i_pmc = header.index("PMCID"); i_pmid = header.index("PMID")
            except (ValueError, AttributeError):
                i_pmc, i_pmid = 0, 1
            want = set(pmc)
            for row in rd:
                if len(row) <= max(i_pmc, i_pmid): continue
                pmc_id = (row[i_pmc] or "").strip()
                pmid_r = (row[i_pmid] or "").strip()
                if pmc_id in want and pmid_r:
                    pmid_to_real[pmc_id] = pmid_r
        print(f"[fine_label] PMCIDs traducidos: {len(pmid_to_real)} (de {len(pmc)} docs PMC)", flush=True)
    # 2) UN solo escaneo: leer todo pmid+mesh del parquet, filtrar en Python
    real_set = set(pmid_to_real.values())
    q = (f"SELECT pmid, mesh FROM read_parquet('{parquet_glob}') "
         f"WHERE mesh IS NOT NULL AND mesh != ''")
    rows = con.execute(q).fetchall()
    con.close()
    found = {str(p): mesh for p, mesh in rows if str(p) in real_set}
    print(f"[fine_label] escaneo parquet: {len(rows)} filas con mesh, "
          f"{len(found)} en nuestro set", flush=True)
    # re-mapear clave: corpus-pmid -> mesh (via pmcid->pmid)
    out = {}
    for corpus_p, real_p in pmid_to_real.items():
        if real_p in found:
            out[corpus_p] = found[real_p]
    return out

def load_mesh_map_sqlite(pmids, db_path):
    """Fallback: join por pmid via sqlite (crea indice si no existe)."""
    import sqlite3
    con = sqlite3.connect(db_path, timeout=300)
    con.row_factory = sqlite3.Row
    con.execute("CREATE INDEX IF NOT EXISTS idx_articles_pmid ON articles(pmid)")
    con.commit()
    pm_int = sorted(int(p) for p in pmids if p.isdigit())
    m = {}
    if pm_int:
        lo, hi = pm_int[0], pm_int[-1]
        rows = con.execute(
            "SELECT pmid, mesh_terms FROM articles "
            "WHERE pmid BETWEEN ? AND ? AND mesh_terms IS NOT NULL AND mesh_terms != ''",
            (lo, hi)).fetchall()
        for r in rows:
            m[str(r["pmid"])] = r["mesh_terms"]
    con.close()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl")
    ap.add_argument("--db", default="/home/a474r867/work/pubmed/index/pubmed_fts.db")
    ap.add_argument("--parquet", default="/beegfs/a474r867/litdump/pubmed/parsed/parquet/year=*/*.parquet")
    ap.add_argument("--pmcid-csv", default="/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl")
    ap.add_argument("--report", default="/beegfs/a474r867/ecoreasoner/data/v5_fine_report.json")
    a = ap.parse_args()

    t0 = time.time()
    # 1) pase 1: pmids del corpus
    all_pmids = set()
    n_total = 0
    for line in open(a.input, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        n_total += 1
        p = str(d.get("pmid", ""))
        if p: all_pmids.add(p)
    print(f"[fine_label] pase 1: {n_total} docs, {len(all_pmids)} pmids", flush=True)

    # 2) join MeSH (duckdb/parquet primero, fallback sqlite)
    mesh_map = {}
    parquet_base = a.parquet.split("year=")[0]
    if os.path.isdir(parquet_base) and any(Path(parquet_base).glob("year=*")):
        try:
            mesh_map = load_mesh_map_duckdb(all_pmids, a.parquet, a.pmcid_csv)
            print(f"[fine_label] pase 2: duckdb/parquet -> {len(mesh_map)} pmids con MeSH", flush=True)
        except Exception as e:
            print(f"[fine_label] duckdb fallo ({e}); fallback sqlite", flush=True)
    if not mesh_map:
        mesh_map = load_mesh_map_sqlite(all_pmids, a.db)
        print(f"[fine_label] pase 2: sqlite -> {len(mesh_map)} pmids con MeSH", flush=True)

    # 3) streaming: reescribir con mesh + domain_fine
    stats = Counter()
    n_with = 0
    out_tmp = a.out + ".tmp"
    with open(a.input, encoding="utf-8") as fin, open(out_tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            pmid = str(d.get("pmid", ""))
            mt = mesh_map.get(pmid, "")
            if mt:
                n_with += 1
                df = fine_domain(mt)
                d["mesh_terms"] = [x.strip() for x in mt.split(";") if x.strip()]
                d["domain_fine"] = df
                d["fine_source"] = "mesh"
                stats[df] += 1
            else:
                # sin MeSH (PMCs no-MEDLINE): clasificacion fina por texto
                df = fine_domain_text(d.get("text", ""))
                d["mesh_terms"] = []
                d["domain_fine"] = df or d.get("domain", "other")
                d["fine_source"] = "text"
                stats[d["domain_fine"]] += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(out_tmp, a.out)

    report = {
        "version": 5, "fine_labels": True, "taxonomy_v": 2,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "docs_total": n_total, "docs_with_mesh": n_with,
        "mesh_coverage_pct": round(100 * n_with / max(n_total, 1), 2),
        "domain_fine": dict(stats.most_common()),
        "fine_source": {"mesh": n_with, "text": n_total - n_with},
        "input": a.input, "output": a.out,
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[fine_label] DONE {n_total} docs, {n_with} con MeSH "
          f"({100*n_with/max(n_total,1):.1f}%), {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({"domain_fine": report["domain_fine"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()