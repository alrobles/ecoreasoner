#!/usr/bin/env python3
"""build_deriv.py — corpus `deriv`: slots FORZADOS por contexto (issue #8, M2).

Spec: docs/designs/DERIV-CORPUS-SPEC.md. Diagnóstico: denoising sobre hechos
arbitrarios enseña "los slots son impredecibles"; aquí cada doc contiene al
menos un span DETERMINADO por el resto del documento — el generador COMPUTE
la respuesta, así que la forcedness es por construcción (verificable).

Familias locales (F2 `deriv_qa` llega por --qa-file desde el pipeline
teacher OLMo, no es generable localmente):
  F1 arith    : cloze aritmético verificable (%cambio, diff, razón, media,
                tasa, conversión, puntos %, total, fracción, resto)
  F3 contrast : valor explícito + negación de la alternativa ("30%, not 12%")
  F4 syll     : premisa->paso->conclusión con diversidad de superficie,
                incluye "no conclusion follows" honestos

Anti-template (lección B3/B4): cada mold de superficie ≤5% de su familia —
se registra el `mold_id` por item y se rechaza al superar el cap. Números
formateados en variantes ("35", "35%", "35 percent") y órdenes
derivación-antes/después + distracciones para romper el patrón fijo.

Formato: JSONL {text, lang:"en", src:"deriv_<fam>", domain:"deriv_<fam>",
mold, slot}. Docs con etiquetas [ETAPA] participan en CF/whole_stage; los
de prosa reciben masking random-denso (receta hinrcf10).

Uso:
    python3 build_deriv.py --n 50000 --seed 11 --out corpus_deriv_v1.jsonl
    python3 build_deriv.py --n 40 --seed 1 --out /tmp/dv.jsonl --verbose
Merge F2:  --qa-file deriv_qa.jsonl
"""
import argparse, json, math, random, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_b3_synthetic import (REGIONS, SPECIES, TRIGGER_VARS, PROCESS, MECH,
                                DIR_SIMPLE, OPP_SIMPLE, NOM, TEMP, SITES,
                                a_an, pick)

# ---------------------------------------------------------------- vocab ----
DOMAINS = {
    "eco": dict(
        subj=["seedling density", "nest success", "canopy cover", "soil moisture",
              "recruitment", "capture rate", "leaf area index", "flowering stems",
              "overwinter survival", "germination rate"],
        unit_pairs=[("nests", "hectare"), ("seedlings", "quadrat"),
                    ("individuals", "plot"), ("stems", "transect"),
                    ("specimens", "trap"), ("rosettes", "square meter")],
        groups=["treatment plots", "restored patches", "fenced plots",
                "northern transects", "exposed sites"],
        ctrls=["control plots", "degraded patches", "unfenced plots",
               "southern transects", "unexposed sites"]),
    "med": dict(
        subj=["systolic blood pressure", "serum glucose", "body mass index",
              "LDL cholesterol", "resting heart rate", "hemoglobin concentration",
              "C-reactive protein", "mean arterial pressure"],
        unit_pairs=[("milligrams", "deciliter"), ("events", "patient-year"),
                    ("doses", "week"), ("readmissions", "month")],
        groups=["the intervention arm", "the high-dose group",
                "the follow-up cohort", "treated patients"],
        ctrls=["the placebo arm", "the low-dose group", "the baseline cohort",
               "untreated controls"]),
    "phys": dict(
        subj=["oscillation period", "thermal conductivity", "photon flux",
              "resistance", "decay rate", "resonant frequency", "drift velocity"],
        unit_pairs=[("counts", "second"), ("photons", "second"),
                    ("events", "hour"), ("oscillations", "minute")],
        groups=["the cryostat run", "the doped sample", "the pumped cavity"],
        ctrls=["the control run", "the undoped sample", "the empty cavity"]),
    "soc": dict(
        subj=["literacy rate", "employment rate", "enrollment share",
              "internet access", "vaccination coverage", "electoral turnout"],
        unit_pairs=[("students", "school"), ("respondents", "district"),
                    ("households", "block"), ("clinics", "municipality")],
        groups=["the northern districts", "pilot provinces", "urban municipalities"],
        ctrls=["the southern districts", "non-pilot provinces",
               "rural municipalities"]),
}

CONVS = [("hectares", "square kilometers", 100.0),
         ("milligrams", "grams", 1000.0),
         ("grams", "kilograms", 1000.0),
         ("millimeters", "centimeters", 10.0),
         ("milliliters", "liters", 1000.0),
         ("minutes", "hours", 60.0),
         ("centimeters", "meters", 100.0)]

SETTINGS = {
    "eco": REGIONS,
    "med": ["a multicenter trial", "the outpatient cohort", "a phase-II study",
            "the longitudinal panel", "the screening program"],
    "phys": ["the beamline experiment", "the low-temperature setup",
             "the calibration campaign", "the vacuum test stand"],
    "soc": ["the national survey", "the census wave", "the panel study",
            "the municipal registry"],
}

DISTRACTORS = [
    "The fieldwork protocol remained unchanged throughout the study.",
    "All measurements were taken under standardized conditions.",
    "Sampling effort was identical across sites.",
    "Observers were trained before data collection began.",
    "The instruments were calibrated weekly.",
    "Weather conditions during sampling were unremarkable.",
    "Data were double-entered and cross-checked for errors.",
    "The study site has been monitored since the program began.",
    "Ethical approval was obtained before enrollment.",
    "Records with missing values were excluded beforehand.",
]

PCT_FMT = ["{n}%", "{n} percent", "a {n}% change", "{n}%"]
NUM_FMT = ["{n}", "{n}", "{n}"]  # may add commas variant later


def fmt_n(n):
    """45->'45', 45.0->'45', 12.5->'12.5'."""
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    return str(n)


TEMP_GEN = ["over the study period", "within three years", "during the experimental period",
            "over five consecutive years", "across two consecutive seasons",
            "during the follow-up window", "over the past decade"]
TEMP_BY_DOM = {"eco": TEMP, "med": TEMP_GEN, "phys": TEMP_GEN, "soc": TEMP_GEN}

SITES_BY_DOM = {
    "eco": SITES,
    "med": ["48 enrolled patients", "36 clinics", "12 hospital wards",
            "210 participants", "25 recruiting centers"],
    "phys": ["18 detector channels", "9 calibration runs", "30 repeated trials",
             "14 measurement sessions"],
    "soc": ["31 municipalities", "220 surveyed households", "45 schools",
            "18 districts"],
}


def intro(rng, dom, s=None):
    d = DOMAINS[dom]
    s = s or pick(rng, d["subj"])
    st = f"in {pick(rng, SETTINGS[dom])}"
    tp = pick(rng, TEMP_BY_DOM[dom])
    return pick(rng, [
        f"{st.capitalize()}, {s} was monitored {tp}.",
        f"Across {pick(rng, SITES_BY_DOM[dom])}, {s} was recorded {tp}.",
        f"The study tracked {s} {st} {tp}.",
        f"Researchers measured {s} {tp}.",
    ])


# ------------------------------------------------------------- F1 arith ----
def op_pct_change(rng, dom, s=None):
    d = DOMAINS[dom]
    a = pick(rng, [20, 25, 40, 50, 60, 80, 100, 120, 150, 200, 250, 400])
    p = pick(rng, [5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
    if a * p % 100 != 0:
        a = 100
    up = rng.random() < 0.55
    b = a + a * p // 100 if up else a - a * p // 100
    if b <= 0 or b == a:
        b = a + a * p // 100
        up = True
    s = s or pick(rng, d["subj"]); u = pick(rng, ["units", "individuals", "points"])
    v0, v1 = ("rose", "increase") if up else ("fell", "decrease")
    molds = [
        f"{s.capitalize()} {v0} from {a} to {b} {u}, {a_an(v1)} {v1} of {p}%.",
        f"{s.capitalize()} {v0} from {a} to {b} {u}; this is a change of {p}%.",
        f"A shift from {a} to {b} {u} in {s} corresponds to {a_an(v1)} {v1} of {p}%.",
        f"Because {s} went from {a} to {b} {u}, the relative {v1} was {p}%.",
        f"The observed {v1} in {s} ({a} to {b} {u}) equals {p}%.",
    ]
    meta = {"op": "pct_change", "expected": str(p), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_diff(rng, dom, s=None):
    d = DOMAINS[dom]
    a, b = pick(rng, [12, 18, 24, 30, 36, 45]), pick(rng, [52, 58, 64, 72, 80, 90])
    s = s or pick(rng, d["subj"]); g, c = pick(rng, d["groups"]), pick(rng, d["ctrls"])
    diff = b - a
    molds = [
        f"{s.capitalize()} reached {b}% in {g} versus {a}% in {c}, a difference of {diff} percentage points.",
        f"In {g}, {s} was {b}%; in {c}, {a}%. The gap is {diff} points.",
        f"{s.capitalize()} differed by {diff} percentage points between {g} ({b}%) and {c} ({a}%).",
        f"Comparing {g} ({b}%) with {c} ({a}%), {s} shows a {diff}-point gap.",
    ]
    meta = {"op": "diff", "expected": str(diff), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_ratio(rng, dom, s=None):
    d = DOMAINS[dom]
    k = pick(rng, [2, 3, 4, 5]); b = pick(rng, [10, 12, 15, 20, 25, 30])
    a = b * k
    s = s or pick(rng, d["subj"]); g, c = pick(rng, d["groups"]), pick(rng, d["ctrls"])
    fold = {2: "double", 3: "triple", 4: "four times", 5: "five times"}[k]
    molds = [
        f"{s.capitalize()} was {a} in {g} and {b} in {c} — {fold} the control value.",
        f"With {a} in {g} versus {b} in {c}, {s} reached {k}-fold the baseline.",
        f"{s.capitalize()} in {g} ({a}) was {k} times that in {c} ({b}).",
        f"The ratio of {s} between {g} and {c} is {k} to 1 ({a} versus {b}).",
    ]
    meta = {"op": "ratio", "expected": str(k), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_mean(rng, dom, s=None):
    d = DOMAINS[dom]
    a, b = pick(rng, [(20, 30), (40, 60), (12, 18), (25, 35), (50, 70), (33, 47)])
    m = (a + b) / 2
    s = s or pick(rng, d["subj"])
    molds = [
        f"{s.capitalize()} was {a} in the first season and {b} in the second, averaging {fmt_n(m)}.",
        f"Across the two seasons ({a} and {b}), mean {s} was {fmt_n(m)}.",
        f"The two measurements of {s}, {a} and {b}, give a mean of {fmt_n(m)}.",
    ]
    meta = {"op": "mean", "expected": fmt_n(m), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_rate(rng, dom, s=None):
    d = DOMAINS[dom]
    ev, un = pick(rng, d["unit_pairs"])
    per = pick(rng, [2, 3, 4, 5, 6, 8, 10])
    t = pick(rng, [4, 5, 6, 10, 12, 20])
    n = per * t
    molds = [
        f"{n} {ev} were recorded over {t} {un}s, a rate of {per} {ev} per {un}.",
        f"Over {t} {un}s there were {n} {ev}: {per} per {un} on average.",
        f"The observed rate was {per} {ev} per {un} ({n} {ev} in {t} {un}s).",
    ]
    meta = {"op": "rate", "expected": str(per), "inputs": {"n": n, "t": t}}
    return pick(rng, molds), "number", meta


def op_conv(rng, dom, s=None):
    u1, u2, div = pick(rng, CONVS)
    base = pick(rng, [2, 3, 4, 5, 8, 12, 15, 20, 25, 40])
    v2 = base
    v1 = base * div
    s = s or pick(rng, DOMAINS[dom]["subj"])
    molds = [
        f"The affected area of {s} covered {fmt_n(v1)} {u1}, i.e. {fmt_n(v2)} {u2}.",
        f"{s.capitalize()} spanned {fmt_n(v1)} {u1} ({fmt_n(v2)} {u2}).",
        f"Measured at {fmt_n(v1)} {u1}, the extent of {s} equals {fmt_n(v2)} {u2}.",
    ]
    meta = {"op": "conv", "expected": fmt_n(v2), "inputs": {"v1": v1, "u1": u1, "u2": u2, "div": div}}
    return pick(rng, molds), "number", meta


def op_pct_points(rng, dom, s=None):
    d = DOMAINS[dom]
    a, b = pick(rng, [(35, 60), (40, 65), (45, 70), (30, 55), (50, 75), (42, 67)])
    pp = b - a
    s = s or pick(rng, d["subj"]); g, c = pick(rng, d["groups"]), pick(rng, d["ctrls"])
    molds = [
        f"{s.capitalize()} rose from {a}% to {b}% under treatment in {g}, {a_an('increase')} increase of {pp} points relative to {c}.",
        f"From {a}% to {b}%: {s} gained {pp} percentage points in {g}.",
        f"In {g}, {s} moved from {a}% to {b}% — a {pp}-point rise over {c}.",
    ]
    meta = {"op": "pct_points", "expected": str(pp), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_total(rng, dom, s=None):
    d = DOMAINS[dom]
    ev, un = pick(rng, d["unit_pairs"])
    a, b = pick(rng, [(12, 18), (23, 31), (15, 25), (40, 35), (28, 22)])
    tot = a + b
    molds = [
        f"{a} {ev} were counted in spring and {b} in autumn, {tot} in total.",
        f"Combining both seasons ({a} + {b}), the survey recorded {tot} {ev}.",
        f"The census found {a} {ev} then {b} more — {tot} {ev} altogether.",
    ]
    meta = {"op": "total", "expected": str(tot), "inputs": {"a": a, "b": b}}
    return pick(rng, molds), "number", meta


def op_frac_pct(rng, dom, s=None):
    d = DOMAINS[dom]
    for _ in range(60):
        n = pick(rng, [20, 25, 40, 50, 80, 100, 200])
        k = pick(rng, [x for x in range(2, n // 2) if (100 * x) % n == 0] or [n // 4])
        pct_val = 100 * k // n
        if 0 < pct_val < 100:
            break
    s = s or pick(rng, d["subj"])
    molds = [
        f"{k} of {n} monitored plots showed elevated {s} — {pct_val}% of the sample.",
        f"{s.capitalize()} was elevated in {k} of {n} plots ({pct_val}%).",
        f"Of {n} plots, {k} showed elevated {s}, which is {pct_val}%.",
    ]
    meta = {"op": "frac_pct", "expected": str(pct_val), "inputs": {"k": k, "n": n}}
    return pick(rng, molds), "number", meta


def op_remainder(rng, dom, s=None):
    d = DOMAINS[dom]
    ev, un = pick(rng, d["unit_pairs"])
    n = pick(rng, [40, 50, 60, 80, 100, 120])
    k = pick(rng, [x for x in (7, 12, 15, 18, 23, 30) if x < n])
    r = n - k
    molds = [
        f"Of {n} {ev} tagged, {k} were lost to predation, leaving {r}.",
        f"{k} of the {n} {ev} failed to recover; {r} remained in the analysis.",
        f"The panel started with {n} {ev}; {k} dropped out, so {r} completed the study.",
    ]
    meta = {"op": "remainder", "expected": str(r), "inputs": {"n": n, "k": k}}
    return pick(rng, molds), "number", meta


F1_OPS = [op_pct_change, op_diff, op_ratio, op_mean, op_rate, op_conv,
          op_pct_points, op_total, op_frac_pct, op_remainder]

# moldes de marco (frame) para F1 — variación de orden y distracción
def f1_doc(rng):
    dom = pick(rng, list(DOMAINS))
    s_shared = pick(rng, DOMAINS[dom]["subj"])
    sent, slot, meta = pick(rng, F1_OPS)(rng, dom, s_shared)
    dom_note = intro(rng, dom, s_shared)
    distr = pick(rng, DISTRACTORS) if rng.random() < 0.4 else ""
    load_bearing = ""
    if rng.random() < 0.3:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|percentage points|points|fold|times)?", sent.split(",")[-1])
        lb_pool = [
            "This derived value exceeds the pre-registered threshold.",
            "The derived figure is what the hypothesis predicted.",
            "That result is the quantity the analysis required.",
        ]
        load_bearing = " " + pick(rng, lb_pool)
    if rng.random() < 0.45:  # esqueleto (participa en CF/whole_stage)
        obs = dom_note
        hip = f"{pick(rng, list(TRIGGER_VARS))} was expected to shift {s_shared} under the intervention."
        evi = sent + (f" {distr}" if distr and rng.random() < 0.5 else "")
        conc = (distr + " " if distr and rng.random() >= 0.5 else "") + \
            f"The reported figures are internally consistent.{load_bearing}".rstrip()
        text = (f"[OBSERVACION] {obs}\n[HIPOTESIS] {hip}\n"
                f"[EVIDENCIA] {evi}\n[CONCLUSION] {conc}")
        return text, "f1_skel", slot, meta
    parts = [dom_note, sent]
    if distr:
        parts.insert(pick(rng, [0, 1, 2]), distr)
    if load_bearing:
        parts.append(load_bearing.strip())
    return " ".join(p for p in parts if p), "f1_prose", slot, meta


# ---------------------------------------------------------- F3 contrast ----
CONTRAST_SUBJ = [
    "abundance", "richness", "survival", "cover", "density", "recruitment",
    "flowering rate", "soil moisture", "seed mass", "nest success",
    "body condition", "growth rate", "dispersal distance", "infection rate",
]


def f3_doc(rng):
    r = rng.random()
    if r < 0.45:  # dirección explícita + alternativa negada
        s = pick(rng, CONTRAST_SUBJ)
        v = pick(rng, ["increased", "decreased", "rose", "fell", "advanced", "delayed"])
        opp = OPP_SIMPLE.get(v, "changed")
        n = pick(rng, [8, 12, 15, 20, 23, 30, 35, 41, 48])
        direction_word = NOM.get(v, "change")  # increased->increase, fell->fall…
        molds = [
            f"{s.capitalize()} {v} by {n}%, not {opp}.",
            f"{s.capitalize()} {v} (not {opp}) by {n}% across the plots.",
            f"The recorded change in {s} was {a_an(direction_word)} {direction_word} of {n}%, not {a_an(NOM.get(opp, opp))} {NOM.get(opp, opp)}.",
            f"{s.capitalize()} {v} by {n}% — {a_an(direction_word)} {direction_word}, not {a_an(NOM.get(opp, opp))} {NOM.get(opp, opp)}.",
        ]
        sent = pick(rng, molds)
        slot = "direction"
    elif r < 0.7:  # número explícito + alternativa negada (caso number:copia)
        n = pick(rng, [12, 18, 24, 32, 36, 42, 48, 56, 64, 72])
        other = pick(rng, [x for x in (6, 9, 14, 21, 27, 33, 45, 60) if x != n])
        unit = pick(rng, ["individuals", "plots", "nests", "samples", "transects"])
        molds = [
            f"The final sample comprised N={n} {unit} (not {other}).",
            f"N={n} {unit} were retained — not {other}, as an early draft stated.",
            f"Counts settled at {n} {unit}, not {other}.",
            f"The verified total was {n} {unit} (not {other}).",
        ]
        sent = pick(rng, molds)
        slot = "number"
    elif r < 0.85:  # polaridad de efecto
        molds = [
            "The effect was positive, not null, in every stratum.",
            "The treatment effect was significant, not marginal.",
            "The response was present, not absent, in treated plots.",
            "Survival was higher, not lower, under the treatment.",
            "The correlation was negative, not positive, as predicted.",
        ]
        sent = pick(rng, molds)
        slot = "negation"
    else:          # posición respecto a umbral
        th = pick(rng, [30, 40, 50, 60, 70])
        v = pick(rng, [x for x in (72, 78, 83, 88, 91) if x > th])
        lo = pick(rng, [x for x in (12, 18, 22, 26) if x < th])
        above = rng.random() < 0.5
        val = v if above else lo
        word = ("above", "below") if above else ("below", "above")
        molds = [
            f"The observed {val}% lay {word[0]}, not {word[1]}, the {th}% threshold.",
            f"{val}% is {word[0]} the {th}% threshold, not {word[1]} it.",
        ]
        sent = pick(rng, molds)
        slot = "direction"
    dom = pick(rng, list(DOMAINS))
    frame = pick(rng, [
        f"Across {pick(rng, SITES_BY_DOM[dom])}, {sent[0].lower() + sent[1:] if sent[0].isupper() else sent}",
        f"{intro(rng, dom)} {sent}",
        sent + f" This held {pick(rng, TEMP_BY_DOM[dom])}.",
    ])
    meta = {"op": "contrast", "expected": sent}
    return frame, "f3", slot, meta


# -------------------------------------------------------------- F4 syll ----
def ger(prop):
    """'enhance nutrient uptake' -> 'enhancing nutrient uptake'."""
    w, rest = prop.split(" ", 1)
    return w[:-1] + "ing " + rest if w.endswith("e") else w + "ing " + rest


def v3(prop):
    """'enhance nutrient uptake' -> 'enhances nutrient uptake'."""
    w, rest = prop.split(" ", 1)
    suf = "es" if w.endswith(("s", "sh", "ch", "x", "z")) else "s"
    return w + suf + " " + rest


SYLL_ENTS = [
    ("mycorrhizal fungi", "soil symbionts"), ("nitrogen fixers", "legume symbionts"),
    ("apex predators", "top trophic consumers"), ("keystone species", "ecosystem engineers"),
    ("invasive annuals", "non-native plants"), ("pollinators", "mutualist insects"),
    ("seed dispersers", "frugivorous animals"), ("decomposers", "saprotrophic organisms"),
]
# instancias genéricas: encajan como miembros de cualquier clase sin absurdos
# ecológicos ("Quercus robur is a soil symbiont" sería falso)
INSTANCES = ["isolate B7", "strain K12", "population p3", "clade IV",
             "the R1 lineage", "sample group delta", "morphotype alpha",
             "cohort gamma", "the Q9 variant", "assemblage beta"]
SYLL_PROPS = [
    "enhance nutrient uptake", "stabilize population cycles", "increase soil aeration",
    "suppress competitor growth", "accelerate decomposition", "buffer drought stress",
    "reduce seedling mortality", "alter community composition",
]


def f4_doc(rng):
    r = rng.random()
    if r < 0.3:    # silogismo categórico (clase b -> miembro x)
        a, b = pick(rng, SYLL_ENTS); prop = pick(rng, SYLL_PROPS)
        x = pick(rng, INSTANCES); b_sing = b[:-1]
        molds = [
            (f"All {b} {prop}; {a} are a case in point. {x} is {a_an(b_sing)} {b_sing}.",
             f"Therefore, {x} {v3(prop)}."),
            (f"Every member of the {b} is known to {prop}. {x} belongs to the {b}.",
             f"It follows that {x} {v3(prop)}."),
            (f"{b.capitalize()} invariably {prop}. Since {x} is {a_an(b_sing)} {b_sing},",
             f"{x} {v3(prop)}."),
        ]
        prem, conc = pick(rng, molds)
        text = f"[EVIDENCIA] {prem}\n[CONCLUSION] {conc}"
        slot = "relation"; meta = {"op": "syll_cat", "expected": conc}
    elif r < 0.55:  # transitividad numérica
        e1, e2, e3 = rng.sample(["alpine", "coastal", "riparian", "old-growth", "secondary", "agricultural", "urban-edge", "restored"], 3)
        n1, n2, n3 = sorted(rng.sample([12, 18, 25, 33, 40, 47, 55, 62, 70, 78], 3), reverse=True)
        molds = [
            (f"Density in {e1} habitats reached {n1}, exceeding {e2} habitats ({n2}), which in turn exceeded {e3} habitats ({n3}).",
             f"Thus {e1} habitats exceeded {e3} habitats in density ({n1} versus {n3})."),
            (f"{e1} sites recorded {n1} individuals, {e2} sites {n2}, and {e3} sites {n3}.",
             f"Ranking them gives {e1} > {e2} > {e3}."),
        ]
        prem, conc = pick(rng, molds)
        text = f"[EVIDENCIA] {prem}\n[CONCLUSION] {conc}"
        slot = "number"; meta = {"op": "syll_num", "expected": conc}
    elif r < 0.8:   # negación load-bearing
        g = pick(rng, ["untreated plots", "control transects", "unexposed sites"])
        resp = pick(rng, ["seedling emergence", "flowering", "recovery", "recruitment"])
        site = pick(rng, ["plot 7", "transect C", "site delta", "the northern plot"])
        molds = [
            (f"No {g} showed {resp} this season. {site.capitalize()} is one of the {g}.",
             f"Therefore, {site} did not show {resp}."),
            (f"{resp.capitalize()} was absent from all {g}. {site.capitalize()} belongs to that group.",
             f"Hence {resp} was absent from {site}."),
            (f"Across all {g}, {resp} never occurred. {site.capitalize()} is untreated." if "untreated" in g else
             f"Across all {g}, {resp} never occurred. {site.capitalize()} is in that group.",
             f"It follows that {resp} did not occur in {site}."),
        ]
        prem, conc = pick(rng, molds)
        text = f"[EVIDENCIA] {prem}\n[CONCLUSION] {conc}"
        slot = "negation"; meta = {"op": "syll_neg", "expected": conc}
    else:            # honesto: no se sigue conclusión
        a, b = pick(rng, SYLL_ENTS); prop = pick(rng, SYLL_PROPS)
        x = pick(rng, INSTANCES); b_sing = b[:-1]
        molds = [
            (f"Some {b} {prop}. {x} is {a_an(b_sing)} {b_sing}.",
             "No definite conclusion follows: membership does not guarantee the trait here."),
            (f"{ger(prop).capitalize()} is typical of several {b}, but not universal. {x} is one of the {b}.",
             f"Whether {x} {v3(prop)} cannot be decided from these premises."),
        ]
        prem, conc = pick(rng, molds)
        text = f"[EVIDENCIA] {prem}\n[CONCLUSION] {conc}"
        slot = "negation"; meta = {"op": "syll_none", "expected": conc}
    return text, "f4", slot, meta


# ---------------------------------------------------------------- driver ---
BAD = ("has fell", "has rised", "None", "{}", "  ", "nan", "{n}", "{a}", "{b}",
       "not not", "a a ", "an an ")


def check(text):
    if any(b in text for b in BAD):
        return False
    labs = re.findall(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]", text)
    if labs:
        order = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
        if labs != sorted(labs, key=order.index):
            return False
        for j, mm in enumerate(re.finditer(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]", text)):
            end = re.search(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]", text[mm.end():])
            seg = text[mm.end(): mm.end() + end.start()] if end else text[mm.end():]
            if not (4 <= len(seg.split()) <= 80):
                return False
    return 10 <= len(text.split()) <= 140


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa-file", default=None,
                    help="JSONL de deriv_qa (teacher OLMo) para mergear")
    ap.add_argument("--mold-cap", type=float, default=0.05,
                    help="fracción máxima de un mold dentro de su familia")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gens = {"arith": (f1_doc, 0.40), "contrast": (f3_doc, 0.30), "syll": (f4_doc, 0.30)}
    molds = {f: {} for f in gens}
    cap = {f: max(20, int(args.n * w * args.mold_cap)) for f, (_, w) in gens.items()}
    docs, rej = [], 0
    while len(docs) < args.n:
        r = rng.random(); acc = 0.0
        for fam, (g, w) in gens.items():
            acc += w
            if r <= acc:
                text, mold, slot, meta = g(rng)
                # mold anti-template: la key incluye el op/molde interno via hash de estructura
                mkey = mold + "|" + re.sub(r"\d+", "#", re.sub(r"[A-Z][a-z]+ [a-z]+", "SP", text))[:60]
                if molds[fam].get(mkey, 0) >= cap[fam]:
                    rej += 1
                    break
                if not check(text):
                    rej += 1
                    break
                molds[fam][mkey] = molds[fam].get(mkey, 0) + 1
                docs.append({"text": text, "lang": "en", "src": f"deriv_{fam}",
                             "domain": f"deriv_{fam}", "mold": mkey,
                             "slot": slot, "forced": meta})
                break
        if rej > args.n * 15:
            raise SystemExit("demasiados rechazos; revisa moldes/caps")

    n_qa = 0
    if args.qa_file and os.path.exists(args.qa_file):
        with open(args.qa_file) as f:
            for line in f:
                d = json.loads(line)
                d.setdefault("src", "deriv_qa"); d.setdefault("domain", "deriv_qa")
                docs.append(d); n_qa += 1
        rng.shuffle(docs)

    with open(args.out, "w") as f:
        for i, d in enumerate(docs):
            d["pid"] = d.get("pid", f"deriv_{i:06d}")
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # self-report
    from collections import Counter
    famc = Counter(d["src"] for d in docs)
    slotc = Counter(d.get("slot", "qa") for d in docs)
    top = {f: (max(molds[f].values()) / max(1, sum(molds[f].values())))
           for f in molds if molds[f]}
    print(f"DERIV OK: {len(docs)} docs (qa-merge {n_qa}) rechazos {rej} -> {args.out}")
    print(" familias:", dict(famc))
    print(" slots:", dict(slotc))
    print(" max mold share por familia:", {k: round(v, 3) for k, v in top.items()})
    if args.verbose:
        for d in docs[:4]:
            print(f"--- {d['src']}/{d.get('slot')}\n{d['text'][:600]}\n")


if __name__ == "__main__":
    main()
