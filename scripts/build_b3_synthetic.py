#!/usr/bin/env python3
"""build_b3_synthetic.py — capa sintetica deductiva estilo FLD (Fase B3, 2026-09-12).

Genera docs de esqueleto con cadenas premisa->derivada->conclusion sobre
relaciones ecologicas. 70% VALIDOS (etapas consistentes: la PREDICCION se
sigue de la HIPOTESIS y la CONCLUSION de la EVIDENCIA), 30% INVALIDOS
(near-miss: PREDICCION o CONCLUSION contradice la direccion/numero del
resto). Cada doc lleva forzados los subtipos debiles de la eval dense L3:
numeros (%), negacion, conectivas causales/adversativas y temporales --
para que el modelo vea el CONTRASTE correcto/incorrecto fuera del texto
(leccion B0: el vocabulario existe, el contraste no).

Plantillas fijas y gramaticales (RSD: trazas cortas y limpias; CoT largo o
crudo degrada modelos pequenos). Verbos en PARTICIPIO (has fallen), nombres
de direccion ("a fall of 41%"), concordancia singular/plural del trigger.

Uso:
    python3 build_b3_synthetic.py --n 30000 --seed 42 --out train_corpus_synth_b3.jsonl
Test:
    python3 build_b3_synthetic.py --n 60 --seed 1 --out /tmp/b3test.jsonl --verbose
"""
import argparse, json, random

REGIONS = [
    "the Iberian Peninsula", "the Pacific Northwest", "the boreal forest of Scandinavia",
    "the drylands of southern Africa", "the alpine meadows of the Pyrenees",
    "the coastal wetlands of the Gulf of Mexico", "the Great Lakes basin",
    "the Tibetan Plateau", "the Amazon lowlands", "the Mediterranean basin",
    "the Patagonian steppe", "the montane cloud forests of Costa Rica",
]
SPECIES = [
    "Quercus robur", "Fagus sylvatica", "Picea abies", "Pinus sylvestris",
    "Salix alba", "Daphnia magna", "Salmo salar", "Oncorhynchus mykiss",
    "Cervus elaphus", "Capreolus capreolus", "Vulpes vulpes", "Ursus arctos",
    "Rangifer tarandus", "Apis mellifera", "Bombus terrestris", "Agrostis capillaris",
    "Festuca rubra", "Carex nigra", "Sphagnum magellanicum", "Mytilus edulis",
    "Zostera marina", "Thalassia testudinum", "Rana temporaria", "Bufo bufo",
    "Larus argentatus", "Alnus glutinosa", "Populus tremula", "Betula pubescens",
]
VARS = [
    "soil moisture", "canopy cover", "nutrient availability", "water depth",
    "light availability", "water pH", "salinity", "fire frequency",
    "snow cover duration", "growing season length", "drought frequency",
    "herbivore density", "predator abundance", "competitor cover",
    "wind exposure", "summer temperature",
]
# trigger -> variables que afecta de forma sensible (evita tautologias)
TRIGGER_VARS = {
    "Prolonged drought": ["soil moisture", "water depth", "fire frequency", "canopy cover"],
    "Warmer winter temperatures": ["snow cover duration", "growing season length", "soil moisture"],
    "The expansion of agriculture": ["canopy cover", "competitor cover"],
    "Increased fire frequency": ["canopy cover", "soil moisture"],
    "Rising atmospheric CO2": ["summer temperature", "growing season length", "nutrient availability"],
    "Eutrophication": ["water pH", "nutrient availability", "water depth"],
    "Overgrazing": ["herbivore density", "competitor cover", "soil moisture"],
    "Habitat fragmentation": ["predator abundance"],
    "Earlier snowmelt": ["snow cover duration", "growing season length", "soil moisture"],
    "The arrival of an invasive predator": ["predator abundance", "competitor cover"],
}
PLURAL_TRIGGERS = {"Warmer winter temperatures"}   # require verbo en forma base
PROCESS = [
    "germination", "seedling survival", "growth rate", "recruitment",
    "reproduction", "photosynthesis", "decomposition", "herbivory",
    "seed dispersal", "pollination", "dormancy release", "flowering phenology",
    "population growth", "nest success", "overwinter survival",
    "leaf emergence", "larval development", "foraging activity",
]
MECH = [
    "water stress", "thermal stress", "resource limitation", "competitive exclusion",
    "facilitation", "phenological mismatch", "trophic cascade", "disease transmission",
    "nutrient cycling", "habitat fragmentation", "oxygen depletion",
]
# formas en 3a persona singular; la forma base se usa con triggers plurales
DIR_V = {"increases": "increase", "decreases": "decrease", "raises": "raise",
         "lowers": "lower", "lengthens": "lengthen", "shortens": "shorten",
         "advances": "advance", "delays": "delay", "intensifies": "intensify",
         "reduces": "reduce", "boosts": "boost"}
# participios (con "has ...") y pasado simple (sin auxiliar)
DIR_PP = ["declined", "increased", "decreased", "advanced", "delayed", "risen",
          "fallen", "accelerated", "weakened", "strengthened", "lengthened",
          "shortened"]
DIR_SIMPLE = ["declined", "increased", "decreased", "advanced", "delayed", "rose",
              "fell", "accelerated", "weakened", "strengthened", "lengthened",
              "shortened"]
OPP_SIMPLE = {"declined": "increased", "increased": "declined", "decreased": "increased",
              "advanced": "delayed", "delayed": "advanced", "rose": "fell",
              "fell": "rose", "accelerated": "slowed", "weakened": "strengthened",
              "strengthened": "weakened", "lengthened": "shortened", "shortened": "lengthened"}
DIR_BARE = ["decline", "increase", "decrease", "rise", "fall", "advance", "delay"]
NOM = {"declined": "decline", "increased": "increase", "decreased": "decrease",
       "advanced": "advance", "delayed": "delay", "risen": "rise", "fallen": "fall",
       "fell": "fall", "rose": "rise", "slowed": "slow",
       "accelerated": "acceleration", "weakened": "weakening",
       "strengthened": "strengthening", "lengthened": "lengthening",
       "shortened": "shortening"}
ADV_LINK = ["therefore", "however", "thus", "consequently", "nevertheless",
            "nonetheless", "hence"]
NEG_EXP = ["plots not exposed", "plots rarely exposed", "plots never exposed",
           "plots not subjected"]
NUMS = [12, 17, 23, 28, 31, 35, 41, 44, 49, 52, 58, 63, 67, 74, 78, 83, 86, 91]
TEMP = [
    "over the past decade", "within three years", "after an unusually warm summer",
    "during the breeding season", "before the onset of winter",
    "across two consecutive growing seasons", "after a severe frost event",
    "over five consecutive years", "during the experimental period",
    "after the first autumn rains",
]
SITES = ["24 monitoring plots", "61 transects", "39 permanent quadrats",
         "17 catchment basins", "52 nest boxes", "30 sampling stations"]


def a_an(w):
    return "an" if w[0] in "aeiou" else "a"


def pick(rng, seq):
    return rng.choice(seq)


def dv_for(rng, trigger):
    s3 = pick(rng, list(DIR_V))
    return s3 if trigger not in PLURAL_TRIGGERS else DIR_V[s3]


def build_valid(rng):
    tr = pick(rng, list(TRIGGER_VARS))
    va = pick(rng, TRIGGER_VARS[tr])
    sp = pick(rng, SPECIES); pr = pick(rng, PROCESS); me = pick(rng, MECH)
    site = pick(rng, SITES); tmp = pick(rng, TEMP); tmp2 = pick(rng, TEMP)
    dv = dv_for(rng, tr); dv2 = dv_for(rng, tr); dv3 = dv_for(rng, tr)
    dp1 = pick(rng, DIR_PP); dp2 = pick(rng, DIR_SIMPLE)
    dpS1 = pick(rng, DIR_SIMPLE)
    db = pick(rng, DIR_BARE); conn = pick(rng, ADV_LINK); neg = pick(rng, NEG_EXP)
    num = pick(rng, NUMS); num2 = pick(rng, NUMS)

    obs = (f"In {pick(rng, REGIONS)}, {sp} {pr} has {dp1} following {tr.lower()}.")
    hip = (f"{tr} {dv} {va}, which {dv2} {pr} in {sp} through {me}; "
           f"{conn.lower()}, the effect amplifies under high {va}.")
    pre = (f"If this hypothesis holds, {sp} {pr} should {db} by {num}% {tmp}; "
           f"{neg} to {va} stress should show a weaker response.")
    evi = (f"Across {site}, mean {va} {dpS1} by {num}% {tmp}; "
           f"concurrently, {sp} {pr} {dp2} by {num2}% {tmp2}.")
    conc = (f"These results support the hypothesis that {me} {dv3} {pr} in {sp}: "
            f"{va} {dpS1} by {num}% while {pr} {dp2} by {num2}%.")
    return obs, hip, pre, evi, conc, sp, pr, va, dpS1, num, num2


def build_invalid(rng):
    obs, hip, pre, evi, conc, sp, pr, va, dp1, num, num2 = build_valid(rng)
    opp = OPP_SIMPLE[dp1]
    nom = NOM[dp1]
    art = a_an(nom)
    tmp = pick(rng, TEMP)
    r = rng.random()
    if r < 0.5:      # CONCLUSION contradice la direccion de la EVIDENCIA
        conc = (f"Although the recorded change was {art} {nom} of {num}%, these "
                f"results are sometimes interpreted as a {NOM[opp]} of {num}% "
                f"{tmp}.")
    elif r < 0.8:    # PREDICCION contradice la HIPOTESIS
        pre = (f"If this hypothesis holds, {sp} {pr} should {NOM[opp]} by {num}% "
               f"{tmp}, the opposite of the expected {dp1} response.")
    else:            # numero roto: CONCLUSION reporta otra magnitud
        other = pick(rng, [n for n in NUMS if abs(n - num) >= 10])
        conc = (f"The change was reported as {art} {nom} of {other}% {tmp}, "
                f"despite the measurements showing {art} {nom} of {num}%.")
    return obs, hip, pre, evi, conc, sp, pr, va, dp1, num, num2


def build(rng, valid=True):
    obs, hip, pre, evi, conc, *_ = build_valid(rng) if valid else build_invalid(rng)
    return f"[OBSERVACION] {obs}\n[HIPOTESIS] {hip}\n[PREDICCION] {pre}\n" \
           f"[EVIDENCIA] {evi}\n[CONCLUSION] {conc}"


def check(text, maxw=60, minw=4):
    import re
    order = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
    m = list(re.finditer(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]",
                         text))
    if len(m) != 5 or [mm.group(1) for mm in m] != order:
        return False
    for j, mm in enumerate(m):
        end = m[j + 1].start() if j + 1 < len(m) else len(text)
        w = len(text[mm.end():end].split())
        if not (minw <= w <= maxw):
            return False
    for bad in ("has fell", "has rised", "a fallen of", "a risen of", "a shortened of",
                "has shortened", "plots not not", "temperatures decreases"):
        if bad in text:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-ratio", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_inv = int(args.n * (1 - args.valid_ratio))
    labels = [True] * (args.n - n_inv) + [False] * n_inv
    rng.shuffle(labels)
    docs, rej = [], 0
    for lb in labels:
        t = build(rng, valid=lb)
        if check(t):
            docs.append((t, lb))
        else:
            rej += 1
            labels.append(lb)          # reintento el tipo
            rng.shuffle(labels)
    with open(args.out, "w") as f:
        for i, (t, valid) in enumerate(docs):
            f.write(json.dumps({"text": t, "pid": f"synth_b3_{i:06d}",
                                "domain": "synth_b3", "valid": valid}) + "\n")
    print(f"B3 OK: {len(docs)} docs (valid {args.n - n_inv}, invalid {n_inv}) "
          f"rechazados {rej} -> {args.out}", flush=True)
    if args.verbose:
        for t, v in docs[:2]:
            print(f"--- valid={v}\n{t[:700]}\n")


if __name__ == "__main__":
    main()