#!/usr/bin/env python3
"""gen_toolcalls_pilot4.py — genera 60 pares adicionales para el piloto de
4 nuevas tools: iucn_status, srtm_elevation, inaturalist_occurrence, try_traits.

Las nuevas muestras se añaden a data/l1/toolcalls_lit_gold.jsonl y se
escriben también en data/l1/toolcalls_lit_pilot4.jsonl para trazabilidad.

Uso:
  python3 scripts/gen_toolcalls_pilot4.py
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Reutilizar listas curadas de gen_toolcalls_lit.py
_module_path = Path(__file__).resolve().parent / "gen_toolcalls_lit.py"
_spec = importlib.util.spec_from_file_location("gen_toolcalls_lit", _module_path)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

SPECIES = _gen.SPECIES
GLOBAL_REGIONS = _gen.GLOBAL_REGIONS

RESOLUTIONS = ["30m", "90m", "1km", "250m"]

TRAIT_NAMES = [
    "leaf_area", "specific_leaf_area", "leaf_nitrogen", "seed_mass",
    "plant_height", "wood_density", "stem_specific_density", "leaf_lifespan",
    "photosynthetic_rate", "rooting_depth", "flower_length", "fruit_mass",
    "dispersal_mode", "growth_form", "leaf_phenology", "drought_tolerance",
    "fire_tolerance", "shade_tolerance", "water_use_efficiency",
    "body_mass", "metabolic_rate", "home_range_size", "litter_size",
    "longevity", "diet_breadth", "activity_pattern", "habitat_breadth",
    "thermal_tolerance", "salinity_tolerance",
]

PROMPT_TEMPLATES = {
    "iucn_status": [
        "Consulta el estado de conservación de {species} en la Lista Roja de la UICN.",
        "Busca la categoría de amenaza de {species} según la UICN.",
        "Obtén el status de la Lista Roja de la UICN para {species}.",
        "¿Cuál es el riesgo de extinción de {species} en la UICN?",
        "Revisa el listado de la UICN para {species}.",
        "Consulta la categoría de conservación de {species} en la Lista Roja.",
    ],
    "srtm_elevation": [
        "Descarga las capas de elevación SRTM para {region} a {resolution} de resolución.",
        "Obtén el modelo digital de elevación SRTM de {region} ({resolution}).",
        "Baja los datos de elevación SRTM para {region} con resolución {resolution}.",
        "Necesito el DEM SRTM de {region} a {resolution}.",
        "Descarga el SRTM de {region} a resolución {resolution} para análisis topográfico.",
    ],
    "inaturalist_occurrence": [
        "Busca observaciones de {species} en iNaturalist para {region}.",
        "Obtén registros ciudadanos de {species} en {region} desde iNaturalist.",
        "Consulta observaciones de {species} en iNaturalist en la región {region}.",
        "Localiza avistamientos de {species} en {region} en iNaturalist.",
        "Busca en iNaturalist observaciones de {species} en {region}.",
        "Descarga observaciones de ciudadanos de {species} para {region}.",
    ],
    "try_traits": [
        "Consulta los rasgos funcionales de {species} en TRY ({trait}).",
        "Busca en TRY el rasgo {trait} para {species}.",
        "Obtén rasgos de {species} de la base de datos TRY ({trait}).",
        "Descarga rasgos funcionales de {species} de TRY para {trait}.",
        "Consulta en TRY el rasgo {trait} de {species}.",
        "Busca rasgos de {species} en la base de datos TRY ({trait}).",
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
    if "species" in args:
        if not _prompt_contains(prompt, args["species"]):
            return False
    if "region" in args:
        if not _prompt_contains(prompt, args["region"]):
            return False
    if "year" in args:
        if args["year"] not in prompt:
            return False
    if "resolution" in args:
        if args["resolution"] not in prompt:
            return False
    if "trait" in args:
        if not _prompt_contains(prompt, args["trait"]):
            return False
    if "layers" in args:
        lnorm = _norm(args["layers"])
        found = any(t in lnorm and t in _norm(prompt) for t in ["bioclim", "current", "future", "rcp", "chelsa", "worldclim", "yucatan"])
        if not found and ("capas" in _norm(prompt) and len(lnorm) > 1):
            found = True
        if not found:
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


def choose_region(rng, preferred=None):
    if preferred:
        return rng.choice(preferred + GLOBAL_REGIONS)
    return rng.choice(GLOBAL_REGIONS)


def generate(n_per_tool=15, seed=7332, verbose=True):
    rng = random.Random(seed)
    records = []
    _used_set = set()

    def register(tool, key1, key2=None):
        _used_set.add((tool, _norm(key1), _norm(key2 or "")))

    def is_used(tool, key1, key2=None):
        return (tool, _norm(key1), _norm(key2 or "")) in _used_set

    # iucn_status: solo especie (region opcional)
    while len([r for r in records if r["gold"][0]["tool"] == "iucn_status"]) < n_per_tool:
        species, group, _ = rng.choice(SPECIES)
        if is_used("iucn_status", species, ""):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["iucn_status"])
        prompt = tpl.format(species=species)
        args = {"species": species}
        tc = _make_gold("iucn_status", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("iucn_status", species)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: categoría de amenaza UICN de {group}",
        })

    # srtm_elevation: region + resolution
    while len([r for r in records if r["gold"][0]["tool"] == "srtm_elevation"]) < n_per_tool:
        region = rng.choice(GLOBAL_REGIONS)
        resolution = rng.choice(RESOLUTIONS)
        if is_used("srtm_elevation", region, resolution):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["srtm_elevation"])
        prompt = tpl.format(region=region, resolution=resolution)
        args = {"region": region, "resolution": resolution}
        if not _consistent(prompt, "srtm_elevation", args):
            continue
        tc = _make_gold("srtm_elevation", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("srtm_elevation", region, resolution)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: datos SRTM para {region} ({resolution})",
        })

    # inaturalist_occurrence: species + region
    while len([r for r in records if r["gold"][0]["tool"] == "inaturalist_occurrence"]) < n_per_tool:
        species, group, regions = rng.choice(SPECIES)
        region = choose_region(rng, regions)
        if is_used("inaturalist_occurrence", species, region):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["inaturalist_occurrence"])
        prompt = tpl.format(species=species, region=region)
        args = {"species": species, "region": region}
        if not _consistent(prompt, "inaturalist_occurrence", args):
            continue
        tc = _make_gold("inaturalist_occurrence", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("inaturalist_occurrence", species, region)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: observaciones ciudadanas de {group} ({species}) en {region}",
        })

    # try_traits: species + trait
    while len([r for r in records if r["gold"][0]["tool"] == "try_traits"]) < n_per_tool:
        species, group, _ = rng.choice(SPECIES)
        trait = rng.choice(TRAIT_NAMES)
        if is_used("try_traits", species, trait):
            continue
        tpl = rng.choice(PROMPT_TEMPLATES["try_traits"])
        prompt = tpl.format(species=species, trait=trait)
        args = {"species": species, "trait": trait}
        if not _consistent(prompt, "try_traits", args):
            continue
        tc = _make_gold("try_traits", args)
        v = _verify_text(tc)
        if not v["ok"]:
            continue
        register("try_traits", species, trait)
        records.append({
            "prompt": prompt,
            "gold": [tc],
            "source": f"literatura: rasgos funcionales de {group} ({species}) - {trait}",
        })

    if verbose:
        print(f"Generados (pilot4): {len(records)} pares")
        by_tool = Counter(r["gold"][0]["tool"] for r in records)
        for t, c in sorted(by_tool.items()):
            print(f"  {t}: {c}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tool", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7332)
    ap.add_argument("--gold", default="data/l1/toolcalls_lit_gold.jsonl")
    ap.add_argument("--pilot", default="data/l1/toolcalls_lit_pilot4.jsonl")
    ap.add_argument("--no-append-gold", action="store_true")
    args = ap.parse_args()

    records = generate(args.n_per_tool, args.seed, verbose=True)

    # Validación
    ok = 0
    for rec in records:
        tc = rec["gold"][0]
        v = _verify_text(tc)
        if v["ok"]:
            ok += 1
    if ok != len(records):
        print(f"[ERROR] verify: {ok}/{len(records)} ok")
        sys.exit(2)
    print(f"[OK] verify (pilot4): {ok}/{len(records)} pasan")

    out_dir = Path(args.pilot).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Escribir archivo del piloto
    with open(args.pilot, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {args.pilot}")

    # Añadir al gold maestro
    if not args.no_append_gold:
        existing = []
        if Path(args.gold).exists():
            with open(args.gold, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
        # evitar duplicados exactos por prompt
        existing_prompts = {r["prompt"] for r in existing}
        new_records = [r for r in records if r["prompt"] not in existing_prompts]
        combined = existing + new_records
        with open(args.gold, "w", encoding="utf-8") as f:
            for rec in combined:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[append] {args.gold}: {len(existing)} + {len(new_records)} = {len(combined)} pares")


if __name__ == "__main__":
    main()
