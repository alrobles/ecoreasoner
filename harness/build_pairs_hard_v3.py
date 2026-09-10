#!/usr/bin/env python3
"""build_pairs_hard_v3.py — batería L0-L3 con hard negatives semánticos para L3.

Mejora sobre build_pairs_hard.py (v2):
  - L3 ahora muta relaciones ecológicas reales:
      * dirección/efecto (increase <-> decrease)
      * número/magnitud (42% -> 21%)
      * entidad (reemplaza especie por otra real del corpus)
      * mecanismo ecológico (competition <-> facilitation <-> predation)
      * variable ambiental (temperature <-> precipitation <-> nitrogen)
      * conectiva causal (because <-> despite)
      * escala temporal (short-term <-> long-term)
  - Cada par L3 incluye 'l3_subtype' para diagnóstico.
  - Controla que el negativo conserve ~misma estructura y overlap para evitar
    atajos léxicos.

Uso:
  python3 harness/build_pairs_hard_v3.py \
      --in data/skeleton/train_skeleton_val.jsonl \
      --species docs/results/species_eco_fine_tagged.json \
      --tokenizer /beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
      --n 512 --out runs/pairs_hard_v3 --seed 7331
"""
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

_LABELS = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Mutaciones de dirección / efecto (la base de v2)
_DIR_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"), ("increases", "decreases"),
    ("higher", "lower"), ("greater", "smaller"), ("elevated", "reduced"),
    ("reduced", "increased"), ("improved", "worsened"), ("improve", "worsen"),
    ("upregulated", "downregulated"), ("activated", "inhibited"),
    ("promotes", "inhibits"), ("enhanced", "suppressed"),
    ("positive", "negative"), ("associated", "unrelated"),
    ("significant", "non-significant"), ("supports", "contradicts"),
    ("gain", "loss"), ("presence", "absence"), ("survival", "mortality"),
    ("more", "less"), ("rose", "fell"), ("higher", "lower"),
    ("faster", "slower"), ("stronger", "weaker"),
    ("positive effect", "negative effect"), ("inhibitory", "promoting"),
]
_DIR_MAP = {}
for _a, _b in _DIR_PAIRS:
    _DIR_MAP[_a] = _b
    _DIR_MAP[_b] = _a

# Mecanismos ecológicos (mutación de relación biológica)
_MECH_PAIRS = [
    ("competition", "facilitation"), ("competition", "predation"),
    ("predation", "herbivory"), ("herbivory", "parasitism"),
    ("mutualism", "competition"), ("facilitation", "mutualism"),
    ("dispersal", "pollination"), ("competition", "allelopathy"),
    ("predator", "prey"), ("host", "parasite"),
    ("drought stress", "heat stress"), ("salinity stress", "drought stress"),
    ("soil moisture", "precipitation"),
]
_MECH_MAP = {}
for _a, _b in _MECH_PAIRS:
    _MECH_MAP[_a] = _b
    _MECH_MAP[_b] = _a

# Variables ambientales
_ENV_PAIRS = [
    ("temperature", "precipitation"), ("temperature", "nitrogen"),
    ("temperature", "pH"), ("temperature", "salinity"),
    ("precipitation", "drought"), ("drought", "flooding"),
    ("nitrogen", "phosphorus"), ("pH", "salinity"),
    ("CO2", "temperature"), ("elevation", "latitude"),
    ("soil moisture", "humidity"), ("light", "temperature"),
]
_ENV_MAP = {}
for _a, _b in _ENV_PAIRS:
    _ENV_MAP[_a] = _b
    _ENV_MAP[_b] = _a

# Conectivas causales / lógicas
_CAUS_PAIRS = [
    ("because", "despite"), ("because of", "in spite of"),
    ("as a result", "although"), ("therefore", "however"),
    ("thus", "nevertheless"), ("consequently", "conversely"),
    ("due to", "regardless of"), ("leads to", "does not affect"),
    ("caused by", "independent of"), ("driven by", "unrelated to"),
]
_CAUS_MAP = {}
for _a, _b in _CAUS_PAIRS:
    _CAUS_MAP[_a] = _b
    _CAUS_MAP[_b] = _a

# Escala temporal / magnitud
_TEMP_PAIRS = [
    ("short-term", "long-term"), ("acute", "chronic"),
    ("temporary", "permanent"), ("rapidly", "gradually"),
    ("1 year", "5 years"), ("2 years", "10 years"),
    ("early", "late"), ("initial", "final"),
]
_TEMP_MAP = {}
for _a, _b in _TEMP_PAIRS:
    _TEMP_MAP[_a] = _b
    _TEMP_MAP[_b] = _a


def _load_species(species_path, rng):
    species = []
    if not species_path:
        return species
    try:
        d = json.loads(Path(species_path).read_text())
        species = [s["canonical"] for s in d.get("species", [])]
    except Exception as e:
        print(f"[warn] no se pudo cargar species: {e}", file=sys.stderr)
    return species


def _parse_stages(text):
    m = list(_RE.finditer(text))
    if len(m) < 3:
        return None, None
    stages = [x.group(1).lower() for x in m]
    texts = [text[m[i].end():m[i + 1].start() if i + 1 < len(m) else None].strip()
             for i in range(len(m))]
    return stages, texts


def _select_k(stages):
    pi = next((i for i, s in enumerate(stages) if s == "prediccion"), None)
    if pi is not None and pi + 1 < len(stages):
        return pi + 1
    if len(stages) == 3:
        return 1
    if len(stages) > 3:
        return 2
    return None


def _try_replace(text, mapping, case_sensitive=False, whole_phrases=None, subtype=""):
    """Reemplaza la primera ocurrencia de un key->value en mapping."""
    if whole_phrases:
        # probar frases completas primero (más específicas)
        for key, val in whole_phrases:
            if key in text:
                return text.replace(key, val, 1), key, f"{subtype}_phrase"
    words = text.split()
    for i, w in enumerate(words):
        key = w.strip(".,;:()").lower() if not case_sensitive else w.strip(".,;:()")
        if key in mapping:
            punct = w[len(w.rstrip(".,;:()")):]
            prefix = w[:len(w) - len(w.lstrip())]
            new = mapping[key]
            if not case_sensitive:
                # conservar capitalización aproximada
                if w[0].isupper():
                    new = new.capitalize()
            words[i] = prefix + new + punct
            return " ".join(words), key, f"{subtype}_word"
    return None, None, None


def _mutate_number(text, rng):
    m = _NUM_RE.search(text)
    if not m:
        return None, None, None
    old = m.group(0)
    try:
        v = float(old)
        # para magnitudes pequeñas (porcentajes) perturbaciones sutiles
        if 0 < v < 100:
            factor = rng.choice([0.5, 1.5, 2.0, 0.25])
            new = f"{v * factor:.3g}"
        else:
            new = f"{v * rng.choice([0.5, 2.0]):.3g}"
    except ValueError:
        new = str(int(old) + rng.choice([1, 2, 5, 10]))
    return text[:m.start()] + new + text[m.end():], old, "number"


def _mutate_entity(text, species, rng):
    if not species:
        return None, None, None
    # buscar si alguna especie aparece en el texto (case-insensitive)
    text_lower = text.lower()
    found = [s for s in species if s.lower() in text_lower]
    if not found:
        return None, None, None
    target = rng.choice(found)
    # elegir reemplazo de otra especie
    others = [s for s in species if s.lower() != target.lower()]
    if not others:
        return None, None, None
    repl = rng.choice(others)
    # conservar capitalización
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    def replacer(m):
        matched = m.group(0)
        return repl.title() if matched[0].isupper() and matched[1:].islower() else repl
    new_text = pattern.sub(replacer, text, count=1)
    return new_text, target, "entity"


def _mutate_payload(text, species, rng):
    """Aplica una mutación L3 elegida por prioridad; devuelve (text, subtype) o (None, None)."""
    # Orden: de más inferencial a menos
    # 1) conectiva causal (muy informativa, poco cambio léxico)
    new, key, st = _try_replace(text, _CAUS_MAP, whole_phrases=list(_CAUS_PAIRS), subtype="causal")
    if new:
        return new, st
    # 2) mecanismo ecológico
    new, key, st = _try_replace(text, _MECH_MAP, subtype="mechanism")
    if new:
        return new, st
    # 3) variable ambiental
    new, key, st = _try_replace(text, _ENV_MAP, subtype="environment")
    if new:
        return new, st
    # 4) escala temporal
    new, key, st = _try_replace(text, _TEMP_MAP, whole_phrases=list(_TEMP_PAIRS), subtype="temporal")
    if new:
        return new, st
    # 5) dirección de efecto
    new, key, st = _try_replace(text, _DIR_MAP, subtype="direction")
    if new:
        return new, st
    # 6) entidad (especie)
    new, key, st = _mutate_entity(text, species, rng)
    if new:
        return new, st
    # 7) número/magnitud
    new, key, st = _mutate_number(text, rng)
    if new:
        return new, st
    # 8) negación simple como fallback
    for aux, neg in ((" was ", " was not "), (" were ", " were not "),
                     (" is ", " is not "), (" are ", " are not "),
                     (" showed ", " did not show "), (" found ", " did not find "),
                     (" suggests ", " does not suggest "), (" indicates ", " does not indicate ")):
        if aux in text:
            return text.replace(aux, neg, 1), "negation"
    return None, None


def _jaccard(a_ids, b_ids):
    sa, sb = set(a_ids), set(b_ids)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def _iter_docs(path, limit):
    n = 0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("text"):
                yield d
                n += 1
                if limit and n >= limit:
                    return


def build_battery(docs, n_pairs, seed, species):
    rng = random.Random(seed)
    cands = []
    for i, d in enumerate(docs):
        st, tx = _parse_stages(d["text"])
        if not st:
            continue
        j = _select_k(st)
        if j is not None:
            cands.append((i, j, st, tx))
    by_dom = defaultdict(list)
    for ci, (i, j, st, tx) in enumerate(cands):
        by_dom[docs[i].get("domain", "?")].append(ci)
    order = list(range(len(cands)))
    rng.shuffle(order)
    out = {"L0": [], "L1": [], "L2": [], "L3": []}
    fallback = defaultdict(int)
    for ci in order:
        if len(out["L0"]) >= n_pairs:
            break
        i, j, st, tx = cands[ci]
        ctx = " ".join(f"[{st[k].upper()}] {tx[k]}" for k in range(j))
        ok = f"[{st[j].upper()}] {tx[j]}"
        # L0
        for _ in range(30):
            ci2 = rng.randrange(len(cands))
            if ci2 != ci:
                break
        i2, j2, st2, tx2 = cands[ci2]
        bad_l0 = f"[{st[j].upper()}] {tx2[min(j, len(tx2) - 1)]}"
        # L1
        pool = by_dom.get(docs[i].get("domain", "?"), [])
        pool = [c for c in pool if c != ci]
        if pool:
            ci1 = rng.choice(pool)
            i1, j1, st1, tx1 = cands[ci1]
            bad_l1 = f"[{st[j].upper()}] {tx1[min(j, len(tx1) - 1)]}"
        else:
            bad_l1 = bad_l0
            fallback["L1_no_domain_pool"] += 1
        # L2
        js = [x for x in range(len(st)) if x != j]
        fut = [x for x in js if x > j]
        if fut:
            jp = rng.choice(fut)
            bad_l2 = f"[{st[j].upper()}] {tx[jp]}"
        else:
            bad_l2 = None
            fallback["L2_no_future_stage"] += 1
        # L3
        mut, subtype = _mutate_payload(tx[j], species, rng)
        bad_l3 = f"[{st[j].upper()}] {mut}" if mut else None
        rec = {"ctx": ctx, "ok": ok}
        out["L0"].append({**rec, "bad": bad_l0})
        out["L1"].append({**rec, "bad": bad_l1})
        if bad_l2:
            out["L2"].append({**rec, "bad": bad_l2})
        if bad_l3:
            out["L3"].append({**rec, "bad": bad_l3, "l3_subtype": subtype})
        else:
            fallback["L3_no_mutable_payload"] += 1
    return out, fallback


def _tok_write(levels, encode, outdir, max_len):
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"levels": {}, "overlap": {}, "l3_subtype_counts": {}}
    for lvl, recs in levels.items():
        kept, ov_ok, ov_bad = [], [], []
        subtype_counts = defaultdict(int)
        for r in recs:
            ids = {k: encode(v) for k, v in r.items() if k in ("ctx", "ok", "bad")}
            if max(len(v) for v in ids.values()) > max_len:
                continue
            kept.append({**r, "ids": ids})
            ov_ok.append(_jaccard(ids["ctx"], ids["ok"]))
            ov_bad.append(_jaccard(ids["ctx"], ids["bad"]))
            if lvl == "L3" and "l3_subtype" in r:
                subtype_counts[r["l3_subtype"]] += 1
        # escribir jsonl con campos ctx/ok/bad (más extras)
        fp = outdir / f"pairs_{lvl}.jsonl"
        with open(fp, "w") as f:
            for r in kept:
                f.write(json.dumps({"ctx": r["ids"]["ctx"], "ok": r["ids"]["ok"],
                                    "bad": r["ids"]["bad"],
                                    **({"l3_subtype": r["l3_subtype"]} if "l3_subtype" in r else {})}) + "\n")
        meta["levels"][lvl] = {"n": len(kept), "file": str(fp)}
        meta["overlap"][lvl] = {
            "jaccard_ctx_ok": round(sum(ov_ok) / len(ov_ok), 4) if ov_ok else None,
            "jaccard_ctx_bad": round(sum(ov_bad) / len(ov_bad), 4) if ov_bad else None,
        }
        meta["l3_subtype_counts"][lvl] = dict(subtype_counts)
        print(f"[{lvl}] {len(kept)} pares -> {fp}  "
              f"jac(ctx,ok)={meta['overlap'][lvl]['jaccard_ctx_ok']} "
              f"jac(ctx,bad)={meta['overlap'][lvl]['jaccard_ctx_bad']}")
        if subtype_counts:
            print(f"  L3 subtypes: {dict(subtype_counts)}")
    return meta


def _selftest():
    docs = [
        {"domain": "eco", "text":
         "[OBSERVACION] bees declined in dry years. "
         "[HIPOTESIS] drought reduces floral resources. "
         "[PREDICCION] dry plots will show lower visitation. "
         "[EVIDENCIA] visitation increased 42% in irrigated plots. "
         "[CONCLUSION] irrigation buffers drought effects."},
        {"domain": "eco", "text":
         "[OBSERVACION] frogs vanished upstream. "
         "[HIPOTESIS] pesticide runoff drives declines. "
         "[PREDICCION] downstream sites show lower abundance. "
         "[EVIDENCIA] abundance decreased 30% downstream. "
         "[CONCLUSION] runoff likely contributes."},
    ]
    species = ["Arabidopsis thaliana", "Oryza sativa"]
    levels, fb = build_battery(docs, n_pairs=4, seed=0, species=species)
    for lvl, recs in levels.items():
        for r in recs:
            assert r["ctx"] and r["ok"] and r["bad"]
    print("selftest OK:", {k: len(v) for k, v in levels.items()}, "fallbacks:", dict(fb))
    for r in levels.get("L3", [])[:3]:
        print("L3 ejemplo:", r["bad"][:160], "subtype:", r.get("l3_subtype"))
    return 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="corpus de esqueletos .jsonl")
    ap.add_argument("--tokenizer", required=True, help="path al tokenizer LLaDA")
    ap.add_argument("--species", default="", help="JSON de especies eco (opcional, para mutación de entidad)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=576)
    ap.add_argument("--out", default="runs/pairs_hard_v3")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--limit-docs", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=True)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    species = _load_species(a.species, random.Random(a.seed))
    docs = list(_iter_docs(a.inp, a.limit_docs))
    print(f"docs cargados: {len(docs)}, species: {len(species)}")
    levels, fb = build_battery(docs, a.n, a.seed, species)
    if fb:
        print("fallbacks:", dict(fb))
    meta = _tok_write(levels, encode, Path(a.out), a.max_len)
    meta["fallbacks"] = dict(fb)
    meta["seed"] = a.seed
    with open(Path(a.out) / "pairs_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
