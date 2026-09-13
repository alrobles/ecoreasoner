#!/usr/bin/env python3
"""build_logic_synth.py — corpus sintetico de CONSISTENCIA FINA (gen g4-logic).

Disenado contra la frontera medida (number/negation ~0.41 en holdout v4).
Diferencia con build_b3_synthetic: aqui la cifra/polaridad es LOAD-BEARING —
la CONCLUSION queda determinada por la EVIDENCIA (misma magnitud, magnitud
derivada, o polaridad negativa preservada), de modo que bajo denoising el
token correcto esta forzado por el contexto y un near-miss numerico o de
negacion resulta inconsistente con la distribucion aprendida.

TODO doc es VALIDO (leccion B4: entrenar near-miss como positivo ensena que
la contradiccion es plausible). Familias:
  exact    20%: mismo numero en PRED/EVID/CONC, formas superficiales variadas
  derived  30%: EVID da absolutos (40->60), CONC deriva %/fold/puntos
  neg      25%: resultado nulo, la CONCLUSION preserva la polaridad negada
  cmp      15%: comparaciones A-vs-B y umbrales ("more than double", "below T%")
  temporal 10%: desplazamientos consistentes ("advanced by 12 days")

Uso:
    python3 build_logic_synth.py --n 30000 --seed 7 --out train_corpus_synth_logic.jsonl
Test:
    python3 build_logic_synth.py --n 40 --seed 1 --out /tmp/lg.jsonl --verbose
"""
import argparse, json, random, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_b3_synthetic import (REGIONS, SPECIES, TRIGGER_VARS, PLURAL_TRIGGERS,
                                PROCESS, MECH, DIR_V, DIR_PP, DIR_SIMPLE,
                                OPP_SIMPLE, NOM, ADV_LINK, NEG_EXP, TEMP, SITES,
                                a_an, pick, dv_for)

PCTS_CLEAN = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100]
BASES = [20, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800]
FOLDS = {2: ("doubled", "twofold"), 3: ("tripled", "threefold"),
         4: ("quadrupled", "fourfold")}
NULL_D = [1, 2, 3, 4]          # diferencias triviales para resultados nulos
DAYS = [3, 5, 7, 9, 11, 12, 14, 18, 21]
WEEKS = [2, 3, 4, 6, 8]
THRESH = [30, 40, 50, 60, 70]

UNITS = ["individuals per plot", "nests per hectare", "seedlings per quadrat",
         "flowering stems per transect", "specimens per trap", "rosettes per square meter"]
GROUPS = ["treatment plots", "exposed sites", "northern transects",
          "restored patches", "fertilized quadrats", "fenced plots"]
CTRLS = ["control plots", "unexposed sites", "southern transects",
         "degraded patches", "unfertilized quadrats", "unfenced plots"]

NULL_EVID = [
    "did not differ between {g} and {c} ({a}% versus {b}%)",
    "showed no detectable difference between {g} and {c} ({a}% versus {b}%)",
    "remained statistically unchanged between {g} and {c} ({a}% versus {b}%)",
    "failed to increase in {g} relative to {c} ({a}% versus {b}%)",
]
NULL_CONC = [
    "{tr} had no detectable effect on {pr} in {sp}",
    "the data do not support an effect of {tr} on {pr} in {sp}",
    "{tr} failed to alter {pr} in {sp}",
    "{pr} in {sp} was insensitive to {trl}",
]
PCT_FORMS = ["by {n}%", "by {n} percent", "a {n}% change", "a {n}-percent change"]


def pct(rng, n):
    return pick(rng, PCT_FORMS).format(n=n)


def header(rng):
    tr = pick(rng, list(TRIGGER_VARS))
    va = pick(rng, TRIGGER_VARS[tr])
    sp = pick(rng, SPECIES); pr = pick(rng, PROCESS); me = pick(rng, MECH)
    dv = dv_for(rng, tr); dv2 = pick(rng, list(DIR_V))
    dp1 = pick(rng, DIR_PP)
    obs = (f"In {pick(rng, REGIONS)}, {sp} {pr} has {dp1} following {tr.lower()}.")
    hip = (f"{tr} {dv} {va}, which {dv2} {pr} in {sp} through {me}.")
    return tr, va, sp, pr, me, obs, hip


def fam_exact(rng):
    tr, va, sp, pr, me, obs, hip = header(rng)
    n = pick(rng, [12, 17, 23, 28, 31, 35, 41, 44, 49, 52, 58, 63, 67, 74, 78, 83])
    db = pick(rng, ["increase", "decrease", "rise", "fall", "decline"])
    dp = {"increase": "increased", "decrease": "decreased", "rise": "rose",
          "fall": "fell", "decline": "declined"}[db]
    t1, t2 = pick(rng, TEMP), pick(rng, TEMP)
    nom = NOM.get(dp, "change")
    vb_pct = pick(rng, ["by {n}%", "by {n} percent"]).format(n=n)
    pre = (f"If this hypothesis holds, {sp} {pr} should {db} {vb_pct} {t1}.")
    evi = (f"Across {pick(rng, SITES)}, mean {pr} {dp} {pct(rng, n)} {t2}, "
           f"consistent with the predicted magnitude.")
    conc = (f"The hypothesis is supported: {pr} showed {a_an(nom)} {nom} of {n}%, "
            f"the same magnitude predicted and observed.")
    return obs, hip, pre, evi, conc


def fam_derived(rng):
    tr, va, sp, pr, me, obs, hip = header(rng)
    u = pick(rng, UNITS); t1, t2 = pick(rng, TEMP), pick(rng, TEMP)
    r = rng.random()
    if r < 0.45:   # absolutos -> porcentaje derivado
        for _ in range(50):
            x = pick(rng, BASES); p = pick(rng, PCTS_CLEAN)
            if x * p % 100 == 0:
                break
        up = rng.random() < 0.5
        y = x + x * p // 100 if up else x - x * p // 100
        if y <= 0 or y == x:
            return fam_exact(rng)
        v0, v1 = ("rose", "increased") if up else ("fell", "decreased")
        evi = (f"Across {pick(rng, SITES)}, mean {pr} {v0} from {x} to {y} {u} {t1}.")
        conc = (f"The hypothesis is supported: {pr} {v1} by {p}% "
                f"(from {x} to {y} {u}) {t2}.")
    elif r < 0.75:  # absolutos -> fold derivado
        k = pick(rng, list(FOLDS)); x = pick(rng, [10, 15, 20, 25, 30, 40, 50])
        y = x * k; vb, nm = FOLDS[k]
        evi = (f"Across {pick(rng, SITES)}, mean {pr} rose from {x} to {y} {u} {t1}.")
        conc = (f"The hypothesis is supported: {pr} {vb} {t2}, "
                f"{a_an(nm)} {nm} increase (from {x} to {y} {u}).")
    else:            # porcentajes -> puntos porcentuales derivados
        a, b = pick(rng, [35, 40, 45, 50, 55]), pick(rng, [60, 65, 70, 75, 80])
        d = b - a
        evi = (f"Across {pick(rng, SITES)}, {pr} reached {b}% in treatment plots "
               f"versus {a}% in controls {t1}.")
        conc = (f"The hypothesis is supported: {pr} was {d} percentage points "
                f"higher under treatment ({b}% versus {a}%).")
    pre = (f"If this hypothesis holds, {pr} in {sp} should shift measurably "
           f"when {va} changes {t2}.")
    return obs, hip, pre, evi, conc


def fam_neg(rng):
    tr, va, sp, pr, me, obs, hip = header(rng)
    g, c = pick(rng, GROUPS), pick(rng, CTRLS)
    a = pick(rng, [40, 45, 50, 55, 60]); d = pick(rng, NULL_D); b = a + d
    t1, t2 = pick(rng, TEMP), pick(rng, TEMP)
    evi = pick(rng, NULL_EVID).format(g=g, c=c, a=a, b=b)
    evi = f"Across {pick(rng, SITES)}, {pr} {evi} {t1}."
    conc = pick(rng, NULL_CONC).format(tr=tr, trl=tr.lower(), pr=pr, sp=sp) + \
        f" {t2}: the {d}% difference is within baseline variability."
    if rng.random() < 0.5:   # cadena nula consistente: prediccion ya anticipa el nulo
        pre = (f"If {me} does not mediate the response, {sp} {pr} should show "
               f"no measurable change between {g} and {c} {t1}.")
        hip2 = (f"{tr} may not affect {pr} in {sp} if {me} is the true driver.")
        return obs, hip2, pre, evi, conc
    pre = (f"If this hypothesis holds, {sp} {pr} should differ clearly between "
           f"{g} and {c} {t1}; the null outcome would refute it.")
    conc = pick(rng, NULL_CONC).format(tr=tr, trl=tr.lower(), pr=pr, sp=sp) + \
        f": the {d}% gap {t2} is within baseline variability, refuting the hypothesis."
    return obs, hip, pre, evi, conc


def fam_cmp(rng):
    tr, va, sp, pr, me, obs, hip = header(rng)
    g, c = pick(rng, GROUPS), pick(rng, CTRLS)
    t1, t2 = pick(rng, TEMP), pick(rng, TEMP)
    if rng.random() < 0.6:   # A vs B
        b = pick(rng, [15, 20, 25, 30, 35]); k = pick(rng, [2, 3])
        a = b * k
        evi = (f"Across {pick(rng, SITES)}, {pr} reached {a}% in {g} versus "
               f"only {b}% in {c} {t1}.")
        rel = "more than doubled" if k == 2 else "more than tripled"
        conc = (f"The hypothesis is supported: {pr} in {g} {rel} the value in "
                f"{c} ({a}% versus {b}%).")
    else:                    # umbral
        th = pick(rng, THRESH); below = rng.random() < 0.5
        v = pick(rng, [n for n in (12, 18, 22, 26, 33, 38) if n < th]) if below \
            else pick(rng, [n for n in (72, 78, 83, 86, 91) if n > th])
        evi = (f"In every site, {pr} remained at {v}% {t1}, "
               f"{'below' if below else 'above'} the {th}% threshold.")
        conc = (f"The hypothesis is supported: {pr} {'never reached' if below else 'exceeded'} "
                f"the {th}% threshold (observed: {v}%).")
    pre = (f"If this hypothesis holds, {pr} in {sp} should separate clearly "
           f"between {g} and {c} {t2}.")
    return obs, hip, pre, evi, conc


def fam_temporal(rng):
    tr, va, sp, pr, me, obs, hip = header(rng)
    t1 = pick(rng, TEMP)
    if rng.random() < 0.5:
        d, un, vb, nm = pick(rng, DAYS), "days", "advanced", "advance"
    else:
        d, un, vb, nm = pick(rng, WEEKS), "weeks", "delayed", "delay"
    pre = (f"If this hypothesis holds, {sp} {pr} should {nm} by roughly "
           f"{d} {un} {t1}.")
    evi = (f"Across {pick(rng, SITES)}, {pr} {vb} by {d} {un} relative to "
           f"the historical baseline {t1}.")
    conc = (f"The hypothesis is supported: {pr} showed {a_an(nm)} {nm} of "
            f"{d} {un}, matching the predicted displacement.")
    return obs, hip, pre, evi, conc


FAMS = [(fam_derived, 0.30), (fam_neg, 0.25), (fam_exact, 0.20),
        (fam_cmp, 0.15), (fam_temporal, 0.10)]


def build(rng):
    r = rng.random(); acc = 0.0
    for f, w in FAMS:
        acc += w
        if r <= acc:
            obs, hip, pre, evi, conc = f(rng)
            break
    return (f"[OBSERVACION] {obs}\n[HIPOTESIS] {hip}\n[PREDICCION] {pre}\n"
            f"[EVIDENCIA] {evi}\n[CONCLUSION] {conc}")


BAD = ("has fell", "has rised", "a fallen of", "a risen of", "has shortened",
       "plots not not", "temperatures decreases", "which lengthen ",
       "which reduce ", "which increase ", "which decrease ", "which raise ",
       "which lower ", "which advance ", "which delay ", "which intensify ",
       "which boost ", "which shorten ", "a slow of", "nan", "None", "{}", "  ")


def check(text):
    import re
    order = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
    m = list(re.finditer(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]", text))
    if len(m) != 5 or [mm.group(1) for mm in m] != order:
        return False
    for j, mm in enumerate(m):
        end = m[j + 1].start() if j + 1 < len(m) else len(text)
        w = len(text[mm.end():end].split())
        if not (4 <= w <= 60):
            return False
    return not any(b in text for b in BAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs, rej = [], 0
    while len(docs) < args.n:
        t = build(rng)
        if check(t):
            docs.append(t)
        else:
            rej += 1
            if rej > args.n * 10:
                raise SystemExit("demasiados rechazos; revisa plantillas")
    with open(args.out, "w") as f:
        for i, t in enumerate(docs):
            f.write(json.dumps({"text": t, "pid": f"synth_logic_{i:06d}",
                                "domain": "synth_logic", "valid": True}) + "\n")
    print(f"LOGIC OK: {len(docs)} docs (100% validos) rechazados {rej} -> {args.out}",
          flush=True)
    if args.verbose:
        for t in docs[:3]:
            print(f"---\n{t}\n")


if __name__ == "__main__":
    main()
