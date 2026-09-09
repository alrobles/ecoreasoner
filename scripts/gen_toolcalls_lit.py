#!/usr/bin/env python3
"""gen_toolcalls_lit.py — genera pares prompt→tool-call (gold) basados en
literatura ecológica real.

Uso:
  python3 scripts/gen_toolcalls_lit.py

Salida:
  - data/l1/toolcalls_lit_gold.jsonl
  - data/l1/REPORT-SOURCE.txt

Cada par es validado con scripts/verify_toolcall.py y auto-chequeado por
consistencia prompt↔args.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ---------------- listas curadas: especies y regiones plausibles ----------------
# Cada tupla es (especie, grupo, [regiones_caracteristicas]).
# Las regiones preferidas se combinan con el catálogo global.
SPECIES = [
    # mamíferos
    ("Panthera onca", "felino neotropical", ["neotropico", "amazon", "peninsula de yucatan", "pantanal", "mesoamerica", "brazil"]),
    ("Puma concolor", "felino", ["neotropico", "patagonia", "andina", "mesoamerica"]),
    ("Panthera leo", "felino africano", ["africa subsahariana", "africa oriental", "norte de africa"]),
    ("Panthera tigris", "felino", ["india", "sudeste asiatico", "china"]),
    ("Acinonyx jubatus", "felino", ["africa oriental", "africa subsahariana", "iran"]),
    ("Lynx pardinus", "felino", ["espana", "ibera"]),
    ("Lynx canadensis", "felino", ["canada"]),
    ("Canis lupus", "cánido", ["espana", "europa", "alpes", "norteamerica"]),
    ("Lycaon pictus", "cánido", ["africa oriental", "africa subsahariana"]),
    ("Ursus arctos", "oso", ["paleartico", "espana", "europa", "alpes"]),
    ("Ursus maritimus", "oso polar", ["artico", "groenlandia"]),
    ("Ursus americanus", "oso negro", ["norteamerica", "canada"]),
    ("Rangifer tarandus", "cérvido", ["artico", "tundra", "escandinavia", "canada"]),
    ("Loxodonta africana", "elefante", ["africa oriental", "africa subsahariana"]),
    ("Elephas maximus", "elefante asiático", ["sudeste asiatico", "india"]),
    ("Bison bison", "bovino", ["norteamerica"]),
    ("Bos primigenius taurus", "bovino", ["europa", "mediterraneo"]),
    ("Camelus dromedarius", "camelido", ["sahara", "norte de africa", "arabia saudita"]),
    ("Giraffa camelopardalis", "jirafa", ["africa oriental", "africa subsahariana"]),
    ("Equus grevyi", "cebra", ["africa oriental", "etiopia"]),
    ("Diceros bicornis", "rinoceronte", ["africa oriental", "sudafrica"]),
    ("Tapirus terrestris", "tapir", ["amazon", "neotropico", "brazil"]),
    ("Bradypus variegatus", "perezoso", ["neotropico", "costa rica", "amazon"]),
    ("Myrmecophaga tridactyla", "oso hormiguero", ["neotropico", "pantanal", "amazon"]),
    ("Hippopotamus amphibius", "hipopótamo", ["africa oriental", "africa subsahariana"]),
    ("Gorilla beringei", "primate", ["africa central", "africa oriental"]),
    ("Pan troglodytes", "primate", ["africa central", "africa occidental"]),
    ("Mandrillus sphinx", "primate", ["africa central"]),
    ("Macaca fascicularis", "primate", ["sudeste asiatico"]),
    ("Pongo abelii", "orangután", ["sumatra"]),
    ("Hylobates lar", "gibón", ["sudeste asiatico"]),
    ("Crocuta crocuta", "hiena", ["africa subsahariana"]),
    ("Gulo gulo", "glutón", ["escandinavia", "canada", "taiga"]),
    ("Lontra canadensis", "nutria", ["norteamerica", "canada"]),
    ("Marmota flaviventris", "marmota", ["norteamerica", "western us"]),
    ("Leptailurus serval", "serval", ["africa subsahariana"]),
    ("Genetta genetta", "ginetta", ["espana", "africa del norte", "mediterraneo"]),
    ("Vulpes vulpes", "zorro", ["espana", "europa", "paleartico"]),
    # aves
    ("Centrocercus urophasianus", "gallo de las artemisa", ["great basin", "western us"]),
    ("Gymnogyps californianus", "cóndor californiano", ["california", "western us"]),
    ("Spheniscus demersus", "pingüino africano", ["sudafrica"]),
    ("Aptenodytes patagonicus", "pingüino", ["patagonia", "sub-antarctica"]),
    ("Thalassarche melanophris", "albatros", ["falklands", "patagonia"]),
    ("Diomedea exulans", "albatros viajero", ["southern ocean"]),
    ("Phoebastria nigripes", "albatros patinegro", ["pacifico norte", "hawai"]),
    ("Anodorhynchus hyacinthinus", "guacamayo azul", ["pantanal", "cerrado"]),
    ("Ara macao", "guacamayo rojo", ["neotropico", "amazon"]),
    ("Amazona auropalliata", "loro", ["mesoamerica", "yucatan"]),
    ("Cyanocorax yncas", "urraca verde", ["andina", "colombia"]),
    ("Ramphastos sulfuratus", "tucán", ["neotropico", "amazon"]),
    ("Procnias tricarunculatus", "campanero tricarunculado", ["costa rica", "neotropico"]),
    ("Campephilus principalis", "picamaderos", ["southeastern us"]),
    ("Picoides borealis", "carpintero", ["southeastern us"]),
    # anfibios / reptiles
    ("Ambystoma mexicanum", "ajolote", ["mexico", "lago de xochimilco"]),
    ("Ambystoma tigrinum", "salamandra tigre", ["norteamerica"]),
    ("Dendrobates tinctorius", "rana venenosa", ["amazon", "guayanas"]),
    ("Oophaga pumilio", "rana fresera", ["costa rica", "panama"]),
    ("Agalychnis callidryas", "rana de ojos rojos", ["neotropico", "mesoamerica"]),
    ("Rana temporaria", "rana bermeja", ["europa", "paleartico"]),
    ("Salamandra salamandra", "salamandra", ["europa", "paleartico"]),
    ("Pleurodeles waltl", "gallipato", ["espana", "portugal"]),
    ("Alytes obstetricans", "sapo partero", ["ibera", "pirineos"]),
    ("Chelonia mydas", "tortuga verde", ["caribbean", "pacifico", "gran barrera de coral"]),
    ("Caretta caretta", "tortuga boba", ["mediterraneo", "caribbean"]),
    ("Crotalus atrox", "víbora de cascabel", ["sonora", "chihuahuan desert", "desierto de sonora"]),
    ("Boa constrictor", "boa", ["neotropico", "amazon"]),
    ("Python bivittatus", "pitón birmana", ["sudeste asiatico"]),
    ("Varanus komodoensis", "dragón de komodo", ["indonesia"]),
    # peces
    ("Oncorhynchus mykiss", "trucha arcoíris", ["norteamerica", "patagonia"]),
    ("Salmo trutta", "trucha común", ["europa", "patagonia"]),
    ("Gadus morhua", "bacalao", ["atlantico norte", "artico"]),
    ("Thunnus thynnus", "atún rojo", ["mediterraneo", "atlantico"]),
    ("Thunnus albacares", "rabil", ["pacifico", "atlantico"]),
    ("Engraulis ringens", "anchoveta", ["humboldt current", "peru", "chile"]),
    ("Sardinops sagax", "sardina", ["humboldt current", "california current"]),
    ("Trachurus murphyi", "jurel", ["humboldt current"]),
    ("Carcharodon carcharias", "tiburón blanco", ["california", "sudafrica", "australia"]),
    ("Sphyrna lewini", "tiburón martillo", ["pacifico este", "galapagos"]),
    ("Hippocampus guttulatus", "caballito de mar", ["mediterraneo", "atlantico este"]),
    # insectos / vectores
    ("Danaus plexippus", "mariposa monarca", ["norteamerica", "mexico", "neotropico"]),
    ("Aedes aegypti", "mosquito", ["neotropico", "tropicos", "africa"]),
    ("Aedes albopictus", "mosquito tigre", ["tropicos", "subtropicos", "sudeste asiatico"]),
    ("Anopheles gambiae", "anófeles", ["africa subsahariana"]),
    ("Culex pipiens", "mosquito", ["cosmopolita", "europa"]),
    ("Bombyx mori", "gusano de seda", ["asia"]),
    ("Heliconius erato", "mariposa", ["neotropico", "amazon"]),
    ("Morpho helenor", "mariposa morfo", ["amazon", "neotropico"]),
    ("Ornithoptera priamus", "mariposa ave", ["australia", "sudeste asiatico"]),
    ("Spodoptera frugiperda", "cogollero", ["america", "neotropico"]),
    ("Leptinotarsa decemlineata", "escarabajo de la papa", ["norteamerica", "europa"]),
    ("Diabrotica virgifera", "barrenador", ["norteamerica"]),
    ("Anoplophora glabripennis", "escarabajo asiático", ["asia", "norteamerica"]),
    # plantas
    ("Pinus sylvestris", "pino silvestre", ["europa", "paleartico"]),
    ("Pinus strobus", "pino blanco", ["norteamerica oriental"]),
    ("Pinus halepensis", "pino carrasco", ["mediterraneo"]),
    ("Quercus robur", "roble", ["europa", "paleartico"]),
    ("Quercus ilex", "encina", ["mediterraneo", "espana"]),
    ("Quercus suber", "alcornoque", ["mediterraneo occidental", "espana", "portugal"]),
    ("Fagus sylvatica", "haya", ["europa", "paleartico"]),
    ("Abies alba", "abeto blanco", ["europa central", "alpes"]),
    ("Picea abies", "picea", ["europa boreal", "alpes"]),
    ("Olea europaea", "olivo", ["mediterraneo"]),
    ("Triticum aestivum", "trigo", ["europa", "mediterraneo"]),
    ("Zea mays", "maíz", ["centroamerica"]),
    ("Oryza sativa", "arroz", ["asia", "sudeste asiatico"]),
    ("Coffea arabica", "café", ["africa oriental", "etiopia"]),
    ("Theobroma cacao", "cacao", ["neotropico", "amazon"]),
    ("Eucalyptus globulus", "eucalipto", ["australia", "mediterraneo"]),
    ("Acacia tortilis", "acacia", ["africa", "sahara"]),
    ("Prosopis glandulosa", "mezquite", ["sonora", "chihuahuan desert"]),
    ("Larrea tridentata", "gobernadora", ["sonora", "mojave", "desierto de sonora"]),
    ("Carnegiea gigantea", "saguaro", ["sonora", "desierto de sonora"]),
    ("Yucca brevifolia", "joshua", ["mojave"]),
    ("Tillandsia usneoides", "musgo español", ["southeastern us", "neotropico"]),
    ("Rhizophora mangle", "mangle rojo", ["caribbean", "neotropico"]),
    ("Avicennia germinans", "mangle negro", ["caribbean"]),
    ("Posidonia oceanica", "posidonia", ["mediterraneo"]),
    ("Zostera marina", "zostera", ["atlantico norte", "pacifico norte"]),
    ("Acropora palmata", "coral", ["caribbean", "gran barrera de coral"]),
    ("Acropora cervicornis", "coral", ["caribbean"]),
    ("Amanita muscaria", "amanita", ["holartico", "boreal"]),
]

GLOBAL_REGIONS = [
    "neotropico", "paleartico", "afrotropico", "indomalayo", "australasia",
    "antartico", "oceanico", "nearctico", "amazonia", "peninsula de yucatan",
    "mesoamerica", "patagonia", "andina", "caribbean", "gran barrera de coral",
    "mediterraneo", "alpes", "sahara", "africa oriental", "africa subsahariana",
    "sudeste asiatico", "himalaya", "sonora", "chihuahuan desert", "mojave",
    "great basin", "california", "western us", "eastern us", "southeastern us",
    "pacifico norte", "humboldt current", "atlantico norte", "mar baltico",
    "groenlandia", "artico", "tundra", "taiga", "india", "china", "indonesia",
    "sumatra", "borneo", "nueva guinea", "australia", "nueva zelanda", "falklands",
    "galapagos", "madagascar", "sudafrica", "namibia", "botswana", "kenia",
    "etiopia", "congo", "gabon", "costa rica", "panama", "brasil", "colombia",
    "peru", "chile", "argentina", "ecuador", "venezuela", "mexico", "espana",
    "portugal", "francia", "italia", "grecia", "alemania", "polonia", "noruega",
    "suecia", "finlandia", "reino unido", "irlanda", "ucrania", "rusia", "siberia",
    "turquia", "iran", "arabia saudita", "yemen", "oman", "pakistan", "bangladesh",
    "myanmar", "tailandia", "vietnam", "malasia", "filipinas", "japon", "corea",
    "mongolia", "egipto", "marruecos", "argelia", "tunez", "libia", "norte de africa",
    "europa", "ibera", "pirineos", "lago de xochimilco", "pantanal", "cerrado",
    "pampas", "desierto de sonora", "hawai", "pacifico", "atlantico", "indico",
    "southern ocean", "sub-antarctica", "canada", "estados unidos", "norteamerica",
    "africa del norte", "australia", "holartico",
]

YEARS = [str(y) for y in range(2015, 2024)]

# frases de capas literarias/estándar de MaxEnt
LAYERS = [
    "bioclim_19", "bioclim", "current", "yucatan_bioclim", "future", "bioclim_2050",
    "chelsa_1981-2010", "wc2.1_30s_bio", "bioclim_1970_2000",
]

PROMPT_TEMPLATES = {
    "gbif_occurrence": [
        "Busca registros de presencia de {species} en {region} para analizar su distribución.",
        "Descarga observaciones de {species} en la región {region}.",
        "Obtén datos de ocurrencia de {species} en {region}.",
        "Consulta la presencia de {species} en {region} para modelar su nicho.",
        "Localiza registros de {species} en {region}.",
        "Recupera registros de presencia de {species} en {region}.",
        "Busca en GBIF ejemplares de {species} observados en {region}.",
        "Extrae datos de presencia de {species} en {region}.",
    ],
    "bioclim_download": [
        "Descarga las capas bioclimáticas de {region} para el año {year}.",
        "Baja las variables bioclimáticas de {region} ({year}).",
        "Obtén los datos bioclimáticos de {region} correspondientes a {year}.",
        "Descarga los datos de clima de {region} para {year}.",
        "Necesito las capas bioclimáticas de {region} del año {year}.",
        "Trae las variables de WorldClim/CHELSA para {region} en {year}.",
    ],
    "maxent_train": [
        "Entrena un modelo MaxEnt para {species} usando las capas {layers}.",
        "Ajusta un modelo de nicho con MaxEnt para {species} con capas {layers}.",
        "Ejecuta MaxEnt sobre {species} usando las variables {layers}.",
        "Entrena MaxEnt para {species} con las capas predictoras {layers}.",
    ],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _split_region(region: str) -> list:
    """Devuelve fragmentos de una región para buscar en el prompt."""
    parts = _norm(region).split()
    # también aceptar raíces de una sola palabra (california, humboldt...)
    return parts


def _prompt_contains(prompt: str, value: str) -> bool:
    """Comprueba si el prompt contiene palabras clave de value (tolerante)."""
    np = _norm(prompt)
    # palabras significativas de value (evitar tokens cortos demasiado generales)
    parts = _split_region(value)
    for p in parts:
        if len(p) <= 2:
            continue
        if p in np:
            return True
    # también buscar el nombre completo
    if _norm(value) in np:
        return True
    return False


def _verify_text(toolcall: dict) -> dict:
    """Convierte un gold al formato que acepta verify_toolcall.py y valida."""
    text = json.dumps([toolcall], ensure_ascii=False)
    # verify_toolcall.py --text lanza un subproceso; para velocidad importamos
    root = Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    import verify_toolcall as vt
    return vt.verify(text)


def _verify_subprocess(toolcall: dict) -> dict:
    """Alternativa: llamar a verify_toolcall.py vía subprocess (más lenta)."""
    text = json.dumps([toolcall], ensure_ascii=False)
    res = subprocess.run(
        ["python3", "scripts/verify_toolcall.py", "--text", text],
        capture_output=True, text=True,
    )
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": res.stderr or res.stdout}


def _make_gold(tool: str, args: dict) -> dict:
    return {"tool": tool, "args": args}


def _consistent(prompt: str, tool: str, args: dict) -> bool:
    """Auto-check: el prompt menciona los args principales."""
    if tool in ("gbif_occurrence", "maxent_train"):
        species = args.get("species")
        if species and not _prompt_contains(prompt, species):
            return False
    if "region" in args:
        if not _prompt_contains(prompt, args["region"]):
            return False
    if "year" in args:
        if args["year"] not in prompt:
            return False
    if "layers" in args:
        layers = args["layers"]
        # buscar raíces clave: bioclim, current, future, rcp, chelsa, worldclim
        lnorm = _norm(layers)
        found = False
        for tok in ["bioclim", "current", "future", "rcp", "chelsa", "worldclim", "yucatan"]:
            if tok in lnorm and tok in _norm(prompt):
                found = True
                break
        if not found:
            # si el prompt dice "capas X" de alguna forma, aceptamos
            if "capas" in _norm(prompt) and len(lnorm) > 1:
                found = True
        if not found:
            return False
    return True


def generate(n_gbif=84, n_bioclim=24, n_maxent=12, seed=7331, verbose=True):
    rng = random.Random(seed)
    records = []
    used_combos = set()

    def add_combo(species, region, tool):
        used_combos.add((tool, _norm(species), _norm(region)))

    def combo_used(species, region, tool):
        return (tool, _norm(species), _norm(region)) in used_combos

    def choose_region(regions):
        pref = [r for r in regions if _norm(r) in {_norm(g) for g in GLOBAL_REGIONS}]
        if not pref:
            pref = regions
        return rng.choice(pref + regions)  # más variedad, con peso a preferidas

    # ---- gbif_occurrence ----
    attempts = 0
    while len([r for r in records if r["gold"][0]["tool"] == "gbif_occurrence"]) < n_gbif and attempts < 5000:
        attempts += 1
        species, group, regions = rng.choice(SPECIES)
        region = choose_region(regions)
        if combo_used(species, region, "gbif_occurrence"):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["gbif_occurrence"])
        prompt = tpl.format(species=species, region=region)
        args = {"species": species, "region": region}
        if not _consistent(prompt, "gbif_occurrence", args):
            continue
        tc = _make_gold("gbif_occurrence", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        add_combo(species, region, "gbif_occurrence")
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: {group} ({species} - distribución / nicho)",
        })

    # ---- bioclim_download ----
    attempts = 0
    while len([r for r in records if r["gold"][0]["tool"] == "bioclim_download"]) < n_bioclim and attempts < 5000:
        attempts += 1
        region = rng.choice(GLOBAL_REGIONS)
        year = rng.choice(YEARS)
        if combo_used(year, region, "bioclim_download"):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["bioclim_download"])
        prompt = tpl.format(region=region, year=year)
        args = {"region": region, "year": year}
        if not _consistent(prompt, "bioclim_download", args):
            continue
        tc = _make_gold("bioclim_download", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        add_combo(year, region, "bioclim_download")
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: datos bioclimáticos para {region} ({year})",
        })

    # ---- maxent_train ----
    attempts = 0
    while len([r for r in records if r["gold"][0]["tool"] == "maxent_train"]) < n_maxent and attempts < 5000:
        attempts += 1
        species, group, _ = rng.choice(SPECIES)
        layers = rng.choice(LAYERS)
        if combo_used(species, layers, "maxent_train"):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["maxent_train"])
        prompt = tpl.format(species=species, layers=layers)
        args = {"species": species, "layers": layers}
        if not _consistent(prompt, "maxent_train", args):
            continue
        tc = _make_gold("maxent_train", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        add_combo(species, layers, "maxent_train")
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: nicho ecológico de {group} con {layers}",
        })

    if verbose:
        print(f"Generados: {len(records)} pares")
        by_tool = Counter(r["gold"][0]["tool"] for r in records)
        for t, c in sorted(by_tool.items()):
            print(f"  {t}: {c}")
    return records


def validate_all(records, use_subprocess=False):
    verify_fn = _verify_subprocess if use_subprocess else _verify_text
    ok = 0
    failures = []
    for i, rec in enumerate(records):
        tc = rec["gold"][0]
        res = verify_fn(tc)
        if not res.get("ok"):
            failures.append((i, rec, res))
        else:
            ok += 1
    return ok, failures


def consistency_all(records):
    failures = []
    for i, rec in enumerate(records):
        tc = rec["gold"][0]
        if not _consistent(rec["prompt"], tc["tool"], tc["args"]):
            failures.append((i, rec))
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gbif", type=int, default=84)
    ap.add_argument("--n-bioclim", type=int, default=24)
    ap.add_argument("--n-maxent", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--out", default="data/l1/toolcalls_lit_gold.jsonl")
    ap.add_argument("--report", default="data/l1/REPORT-SOURCE.txt")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--subprocess-verify", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate(args.n_gbif, args.n_bioclim, args.n_maxent, args.seed, verbose=True)

    # Auto-checks
    consistency_failures = consistency_all(records)
    if consistency_failures:
        print(f"[ERROR] {len(consistency_failures)} pares fallan consistencia prompt↔args")
        for i, rec in consistency_failures[:5]:
            print(f"  {i}: {rec['prompt'][:80]} ... {rec['gold'][0]}")
        sys.exit(2)

    if not args.no_verify:
        ok, failures = validate_all(records, use_subprocess=args.subprocess_verify)
        if ok != len(records):
            print(f"[ERROR] verify_toolcall: {ok}/{len(records)} ok")
            for i, rec, res in failures[:5]:
                print(f"  {i}: {res.get('error')}")
            sys.exit(2)
        print(f"[OK] verify_toolcall: {ok}/{len(records)} pasan")

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Report
    by_tool = Counter(r["gold"][0]["tool"] for r in records)
    report = f"""toolcalls_lit_gold — REPORT-SOURCE
================================
Fecha: 2026-09-09
Pares totales: {len(records)}
Distribución:
"""
    for t, c in sorted(by_tool.items()):
        pct = c / len(records) * 100
        report += f"  {t}: {c} ({pct:.1f}%)\n"
    report += f"""
Seed: {args.seed}
Archivo: {args.out}
Validación:
  - Consistencia prompt↔args: {len(records) - len(consistency_failures)}/{len(records)}
  - verify_toolcall.py: {len(records) if not args.no_verify else 'omitido (modo --no-verify)'}

Notas:
- Especies y regiones reales de literatura ecológica.
- Prompts son llamadas directas a 3 herramientas (gbif_occurrence,
  bioclim_download, maxent_train).
- Sin análisis compuestos, sin scripts/HPC/PubMed/GitHub/APIs de terceros.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"[write] {args.out}")
    print(f"[write] {args.report}")


if __name__ == "__main__":
    main()
