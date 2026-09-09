#!/usr/bin/env python3
"""gen_toolcalls_evolucion.py — 120 pares prompt→gold para tools de evolución.

Tools:
  - timetree_divergence:    consultar edad de divergencia de un taxón en TimeTree
  - opentree_phylogeny:     descargar filogenia de Open Tree of Life
  - ncbi_taxonomy:          consultar taxonomía en NCBI

Salida:
  - data/l1/toolcalls_lit_evolucion.jsonl
  - añade a data/l1/toolcalls_lit_gold.jsonl (opcional)
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "gen_toolcalls_lit.py"
_spec = importlib.util.spec_from_file_location("gen_toolcalls_lit", _module_path)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

SPECIES = _gen.SPECIES

# Taxa reales de literatura evolutiva: especies, géneros, familias, órdenes, clados.
# Proveniencia: papers de filogeografía, filogenia molecular, divergence-time
# estimation, biogeografía histórica y evolución de rasgos.
TAXA = [
    # mamíferos - felinos
    ("Panthera", "género de felinos", "Mammalia"),
    ("Felidae", "familia de felinos", "Mammalia"),
    ("Panthera onca", "jaguar", "Mammalia"),
    ("Panthera leo", "león", "Mammalia"),
    ("Puma concolor", "puma", "Mammalia"),
    ("Acinonyx jubatus", "guepardo", "Mammalia"),
    ("Lynx pardinus", "lince ibérico", "Mammalia"),
    # cánidos
    ("Canidae", "familia de cánidos", "Mammalia"),
    ("Canis lupus", "lobo gris", "Mammalia"),
    ("Lycaon pictus", "perro salvaje africano", "Mammalia"),
    ("Vulpes vulpes", "zorro rojo", "Mammalia"),
    # úrsidos
    ("Ursidae", "familia de osos", "Mammalia"),
    ("Ursus arctos", "oso pardo", "Mammalia"),
    ("Ursus maritimus", "oso polar", "Mammalia"),
    ("Ailuropoda melanoleuca", "oso panda", "Mammalia"),
    # cérvidos / bóvidos
    ("Cervidae", "familia de ciervos", "Mammalia"),
    ("Rangifer tarandus", "reno", "Mammalia"),
    ("Bovidae", "familia de bóvidos", "Mammalia"),
    ("Bison bison", "bisonte americano", "Mammalia"),
    # primates
    ("Hominidae", "grandes simios", "Mammalia"),
    ("Homo sapiens", "humano", "Mammalia"),
    ("Pan troglodytes", "chimpancé", "Mammalia"),
    ("Gorilla beringei", "gorila", "Mammalia"),
    ("Pongo abelii", "orangután de sumatra", "Mammalia"),
    ("Macaca fascicularis", "mono de cola larga", "Mammalia"),
    ("Cercopithecidae", "monos del viejo mundo", "Mammalia"),
    # proboscídeos
    ("Elephantidae", "elefantes", "Mammalia"),
    ("Loxodonta africana", "elefante africano", "Mammalia"),
    ("Elephas maximus", "elefante asiático", "Mammalia"),
    # cetáceos
    ("Cetacea", "ballenas y delfines", "Mammalia"),
    ("Delphinidae", "delfines", "Mammalia"),
    ("Tursiops truncatus", "delfín mular", "Mammalia"),
    ("Balaenoptera musculus", "ballena azul", "Mammalia"),
    ("Physeter macrocephalus", "cachalote", "Mammalia"),
    # ungulados / perisodáctilos
    ("Equidae", "caballos", "Mammalia"),
    ("Equus ferus", "caballo", "Mammalia"),
    ("Equus grevyi", "cebra de grevy", "Mammalia"),
    ("Rhinocerotidae", "rinocerontes", "Mammalia"),
    ("Diceros bicornis", "rinoceronte negro", "Mammalia"),
    # roedores / lagomorfos
    ("Rodentia", "roedores", "Mammalia"),
    ("Sciuridae", "ardillas", "Mammalia"),
    ("Marmota flaviventris", "marmota", "Mammalia"),
    # marsupiales
    ("Didelphidae", "zarigüeyas", "Mammalia"),
    ("Macropodidae", "canguros", "Mammalia"),
    # aves
    ("Aves", "aves", "Aves"),
    ("Passeriformes", "paseriformes", "Aves"),
    ("Centrocercus urophasianus", "gallo de las artemisa", "Aves"),
    ("Gymnogyps californianus", "cóndor californiano", "Aves"),
    ("Psittaciformes", "loros", "Aves"),
    ("Anodorhynchus hyacinthinus", "guacamayo azul", "Aves"),
    ("Ara macao", "guacamayo escarlata", "Aves"),
    ("Falconiformes", "falconiformes", "Aves"),
    ("Anseriformes", "anseriformes", "Aves"),
    ("Spheniscidae", "pingüinos", "Aves"),
    ("Spheniscus demersus", "pingüino africano", "Aves"),
    # reptiles
    ("Squamata", "escamosos", "Reptilia"),
    ("Serpentes", "serpientes", "Reptilia"),
    ("Boa constrictor", "boa", "Reptilia"),
    ("Python bivittatus", "pitón birmana", "Reptilia"),
    ("Crotalus atrox", "víbora de cascabel", "Reptilia"),
    ("Testudines", "tortugas", "Reptilia"),
    ("Chelonia mydas", "tortuga verde", "Reptilia"),
    ("Crocodylia", "cocodrilos", "Reptilia"),
    ("Varanus komodoensis", "dragón de komodo", "Reptilia"),
    # anfibios
    ("Anura", "ranas", "Amphibia"),
    ("Dendrobates tinctorius", "rana venenosa", "Amphibia"),
    ("Ambystoma mexicanum", "ajolote", "Amphibia"),
    ("Caudata", "salamandras", "Amphibia"),
    # peces
    ("Teleostei", "teleósteos", "Actinopterygii"),
    ("Cyprinidae", "carpas", "Actinopterygii"),
    ("Salmonidae", "salmónidos", "Actinopterygii"),
    ("Oncorhynchus mykiss", "trucha arcoíris", "Actinopterygii"),
    ("Carcharodon carcharias", "tiburón blanco", "Chondrichthyes"),
    ("Sphyrna lewini", "tiburón martillo", "Chondrichthyes"),
    # insectos
    ("Lepidoptera", "mariposas", "Insecta"),
    ("Danaus plexippus", "monarca", "Insecta"),
    ("Heliconius erato", "mariposa heliconius", "Insecta"),
    ("Hymenoptera", "abejas y hormigas", "Insecta"),
    ("Apidae", "abejas", "Insecta"),
    ("Coleoptera", "escarabajos", "Insecta"),
    ("Diptera", "moscas y mosquitos", "Insecta"),
    ("Aedes aegypti", "mosquito", "Insecta"),
    ("Anopheles gambiae", "anófeles", "Insecta"),
    ("Orthoptera", "saltamontes", "Insecta"),
    # plantas
    ("Angiosperms", "angiospermas", "Plantae"),
    ("Fagaceae", "faigáceas", "Plantae"),
    ("Quercus", "robles", "Plantae"),
    ("Quercus robur", "roble", "Plantae"),
    ("Quercus suber", "alcornoque", "Plantae"),
    ("Pinaceae", "pináceas", "Plantae"),
    ("Pinus sylvestris", "pino silvestre", "Plantae"),
    ("Poaceae", "gramíneas", "Plantae"),
    ("Zea mays", "maíz", "Plantae"),
    ("Oryza sativa", "arroz", "Plantae"),
    ("Fabaceae", "leguminosas", "Plantae"),
    ("Asteraceae", "compuestas", "Plantae"),
    ("Orchidaceae", "orquídeas", "Plantae"),
    ("Rosaceae", "rosáceas", "Plantae"),
    # grupos mayores
    ("Mammalia", "mamíferos", "Chordata"),
    ("Chordata", "cordados", "Animalia"),
    ("Metazoa", "metazoos", "Animalia"),
    ("Arthropoda", "artrópodos", "Animalia"),
    ("Mollusca", "moluscos", "Animalia"),
    ("Echinodermata", "equinodermos", "Animalia"),
    ("Gymnosperms", "gimnospermas", "Plantae"),
    ("Bryophyta", "musgos", "Plantae"),
    ("Fungi", "hongos", "Fungi"),
    ("Ascomycota", "ascomicetes", "Fungi"),
    ("Basidiomycota", "basidiomicetes", "Fungi"),
]

# Regiones asociadas a estudios de evolución / filogeografía
REGIONS = [
    "neotropico", "paleartico", "afrotropico", "indomalayo", "australasia",
    "nearctico", "amazonia", "patagonia", "andina", "mesoamerica",
    "caribbean", "mediterraneo", "alpes", "sahara", "africa oriental",
    "sudeste asiatico", "himalaya", "india", "china", "indonesia",
    "sumatra", "borneo", "australia", "madagascar", "sudafrica",
    "kenia", "etiopia", "congo", "gabon", "brasil", "colombia",
    "peru", "chile", "argentina", "mexico", "espana", "europa",
    "norteamerica", "canada", "artico", "antartida", "groenlandia",
    "pacifico", "atlantico", "oceano indico", "gran barrera de coral",
    "galapagos", "hawai", "nueva zelanda", "falklands", "islas canarias",
    "mojave", "sonora", "desierto de gobi", "siberia", "escandinavia",
]

PROMPT_TEMPLATES = {
    "timetree_divergence": [
        "Consulta la edad de divergencia de {taxon} en TimeTree.",
        "¿Cuándo divergió {taxon} según TimeTree?",
        "Obtén el tiempo de divergencia de {taxon} en TimeTree.",
        "Busca la edad de divergencia de {taxon} en la base de datos TimeTree.",
        "Consulta en TimeTree el tiempo de divergencia estimado para {taxon}.",
        "Necesito el dato de divergencia de {taxon} de TimeTree.",
        "¿Qué edad de divergencia tiene {taxon} en TimeTree?",
        "Extrae de TimeTree el tiempo de divergencia del clado {taxon}.",
    ],
    "opentree_phylogeny": [
        "Descarga el árbol filogenético de {taxon} desde Open Tree of Life.",
        "Obtén la filogenia de {taxon} de Open Tree of Life.",
        "Busca en OpenTree el árbol filogenético del clado {taxon}.",
        "Descarga la filogenia de {taxon} desde OTL (Open Tree of Life).",
        "Necesito el árbol de {taxon} de Open Tree of Life.",
        "Consulta la topología filogenética de {taxon} en OpenTree.",
        "Obtén el árbol de {taxon} del Open Tree of Life.",
    ],
    "ncbi_taxonomy": [
        "Consulta la clasificación taxonómica de {species} en NCBI.",
        "Obtén la taxonomía de {species} desde NCBI Taxonomy.",
        "Busca la clasificación de {species} en NCBI.",
        "Necesito la taxonomía de {species} del NCBI.",
        "Consulta la línea taxonómica de {species} en NCBI.",
        "Descarga la clasificación de {species} de NCBI Taxonomy.",
        "Obtén los rangos taxonómicos de {species} en NCBI.",
        "Revisa la taxonomía de {species} en NCBI.",
    ],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _prompt_contains(prompt: str, value: str) -> bool:
    np = _norm(prompt)
    parts = [p for p in _norm(value).split() if len(p) > 2]
    for p in parts:
        if p in np:
            return True
    if _norm(value) in np:
        return True
    return False


def _consistent(prompt: str, tool: str, args: dict) -> bool:
    if tool in ("timetree_divergence", "opentree_phylogeny"):
        if not _prompt_contains(prompt, args["taxon"]):
            return False
    if tool == "ncbi_taxonomy":
        if not _prompt_contains(prompt, args["species"]):
            return False
    if "region" in args:
        if not _prompt_contains(prompt, args["region"]):
            return False
    return True


def _verify_text(toolcall: dict) -> dict:
    text = json.dumps([toolcall], ensure_ascii=False)
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import verify_toolcall as vt
    return vt.verify(text)


def _make_gold(tool: str, args: dict) -> dict:
    return {"tool": tool, "args": args}


def generate(n_timetree=40, n_opentree=40, n_ncbi=40, seed=7333, verbose=True):
    rng = random.Random(seed)
    records = []
    _used_set = set()

    def register(tool, key1, key2=None):
        _used_set.add((tool, _norm(key1), _norm(key2 or "")))

    def is_used(tool, key1, key2=None):
        return (tool, _norm(key1), _norm(key2 or "")) in _used_set

    # timetree_divergence
    while len([r for r in records if r["gold"][0]["tool"] == "timetree_divergence"]) < n_timetree:
        taxon, group, _ = rng.choice(TAXA)
        if is_used("timetree_divergence", taxon):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["timetree_divergence"])
        prompt = tpl.format(taxon=taxon)
        args = {"taxon": taxon}
        if not _consistent(prompt, "timetree_divergence", args):
            continue
        tc = _make_gold("timetree_divergence", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("timetree_divergence", taxon)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: divergencia temporal de {group} ({taxon})",
        })

    # opentree_phylogeny: región es opcional; generamos sin ella para
    # evitar inconsistencia prompt↔args (la filogenia es global).
    while len([r for r in records if r["gold"][0]["tool"] == "opentree_phylogeny"]) < n_opentree:
        taxon, group, _ = rng.choice(TAXA)
        if is_used("opentree_phylogeny", taxon):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["opentree_phylogeny"])
        prompt = tpl.format(taxon=taxon)
        args = {"taxon": taxon}
        if not _consistent(prompt, "opentree_phylogeny", args):
            continue
        tc = _make_gold("opentree_phylogeny", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("opentree_phylogeny", taxon)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: filogenia de {group} ({taxon})",
        })

    # ncbi_taxonomy
    while len([r for r in records if r["gold"][0]["tool"] == "ncbi_taxonomy"]) < n_ncbi:
        species, group, _ = rng.choice(SPECIES)
        if is_used("ncbi_taxonomy", species):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["ncbi_taxonomy"])
        prompt = tpl.format(species=species)
        args = {"species": species}
        if not _consistent(prompt, "ncbi_taxonomy", args):
            continue
        tc = _make_gold("ncbi_taxonomy", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("ncbi_taxonomy", species)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: taxonomía de {group} ({species})",
        })

    if verbose:
        print(f"Generados (evolución): {len(records)} pares")
        by_tool = Counter(r["gold"][0]["tool"] for r in records)
        for t, c in sorted(by_tool.items()):
            print(f"  {t}: {c}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-timetree", type=int, default=40)
    ap.add_argument("--n-opentree", type=int, default=40)
    ap.add_argument("--n-ncbi", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7333)
    ap.add_argument("--gold", default="data/l1/toolcalls_lit_gold.jsonl")
    ap.add_argument("--evol", default="data/l1/toolcalls_lit_evolucion.jsonl")
    ap.add_argument("--no-append-gold", action="store_true")
    args = ap.parse_args()

    records = generate(args.n_timetree, args.n_opentree, args.n_ncbi, args.seed, verbose=True)

    ok = 0
    for rec in records:
        tc = rec["gold"][0]
        v = _verify_text(tc)
        if v["ok"]:
            ok += 1
    if ok != len(records):
        print(f"[ERROR] verify: {ok}/{len(records)} ok")
        sys.exit(2)
    print(f"[OK] verify (evolución): {ok}/{len(records)} pasan")

    out_dir = Path(args.evol).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.evol, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {args.evol}")

    if not args.no_append_gold:
        existing = []
        if Path(args.gold).exists():
            with open(args.gold, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
        existing_prompts = {r["prompt"] for r in existing}
        new_records = [r for r in records if r["prompt"] not in existing_prompts]
        combined = existing + new_records
        with open(args.gold, "w", encoding="utf-8") as f:
            for rec in combined:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[append] {args.gold}: {len(existing)} + {len(new_records)} = {len(combined)} pares")


if __name__ == "__main__":
    main()
