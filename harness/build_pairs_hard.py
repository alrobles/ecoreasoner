#!/usr/bin/env python3
"""build_pairs_hard.py — batería de pares con negativos graduados (L0–L3).

Motivación (auditoría Devin §2a + F2 FALSIFY 2026-09-08): los pares originales
(build_pairs.py) construyen `bad` = etapa de OTRO documento -> la tarea es
resoluble por coherencia temática sin inferencia (lección documentada de
NSP/NLI: Naik et al. 2018 stress-tests, ANLI, críticas a NSP). Esta batería
gradúa los negativos para localizar QUÉ aprendió el modelo:

  L0  bad = etapa k de otro doc (cualquier dominio)   baseline: coherencia
  L1  bad = etapa k de doc del MISMO dominio          controla tópico grueso
  L2  bad = contenido de etapa j!=k del MISMO doc     controla tópico 100%;
      con la etiqueta correcta                      mide orden/rol de etapa
  L3  bad = etapa k real con payload mutado           dirección del efecto /
      (increase<->decrease, número, negación)         inferencia de contenido

Interpretación esperada sobre un ckpt entrenado:
  L0 alto, L1-L3 ~azar  -> solo coherencia temática (el atajo)
  L2 > azar, L3 ~azar   -> gramática de etapas sin inferencia
  L3 > umbral           -> señal inferencial real

Salida: <out>/pairs_L{0..3}.jsonl con {"ctx","ok","bad"} ids (mismo formato que
build_pairs.py -> suite_smoke.py los consume sin cambios) + pairs_meta.json con
diagnóstico de overlap léxico (Jaccard de sets de tokens ctx<->ok vs ctx<->bad;
si el modelo decide por overlap, L0 es explotable y L2 no).

Uso:
  python3 harness/build_pairs_hard.py --in data/train_skeleton.jsonl \
      --tokenizer <path_llada> --n 512 --out runs/pairs_hard/ --seed 7331
  python3 harness/build_pairs_hard.py --diagnose runs/pairs.jsonl  # 0 GPU
  python3 harness/build_pairs_hard.py --selftest                    # 0 deps
"""
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

# mismas etiquetas y regex que build_pairs.py
_LABELS = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")

# --- payload mutations (L3) -------------------------------------------------
# pares bidireccionales de dirección/efecto: cubren el payload inferencial más
# frecuente en biomed (aumenta/disminuye, mejora/empeora, significativo/no).
_DIR_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"), ("increases", "decreases"),
    ("higher", "lower"), ("greater", "smaller"), ("elevated", "reduced"),
    ("reduced", "increased"), ("improved", "worsened"), ("improve", "worsen"),
    ("upregulated", "downregulated"), ("activated", "inhibited"),
    ("promotes", "inhibits"), ("enhanced", "suppressed"),
    ("positive", "negative"), ("associated", "unrelated"),
    ("significant", "non-significant"), ("supports", "contradicts"),
    ("gain", "loss"), ("presence", "absence"), ("survival", "mortality"),
    ("more", "less"), ("rose", "fell"),
]
_DIR_MAP = {}
for _a, _b in _DIR_PAIRS:
    _DIR_MAP[_a] = _b
    _DIR_MAP[_b] = _a
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _parse_stages(text):
    m = list(_RE.finditer(text))
    if len(m) < 3:
        return None, None
    stages = [x.group(1).lower() for x in m]
    texts = [text[m[i].end():m[i + 1].start() if i + 1 < len(m) else None].strip()
             for i in range(len(m))]
    return stages, texts


def _select_k(stages):
    """Igual que build_pairs: el candidato es la etapa tras PREDICCION (típicamente
    EVIDENCIA). Devuelve j o None."""
    pi = next((i for i, s in enumerate(stages) if s == "prediccion"), None)
    if pi is None or pi + 1 >= len(stages):
        return None
    return pi + 1


def _mutate_payload(text, rng):
    """Muta el contenido inferencial de una etapa: primero dirección/efecto,
    si no hay -> números; si tampoco -> negación simple; si nada aplica -> None."""
    words = text.split()
    # 1) swap de dirección (el flip más inferencial: "X increased Y" -> "decreased")
    for i, w in enumerate(words):
        key = w.strip(".,;:()").lower()
        if key in _DIR_MAP:
            punct = w[len(w.rstrip(".,;:()")):]
            words[i] = w[:len(w) - len(w.lstrip())] + _DIR_MAP[key] + punct
            return " ".join(words)
    # 2) perturbación numérica (magnitud o dígito distinto)
    m = _NUM_RE.search(text)
    if m:
        old = m.group(0)
        try:
            v = float(old)
            new = f"{v * rng.choice([0.5, 1.5, 2.0]):.3g}"
        except ValueError:
            new = str(int(old) + rng.choice([1, 2, 5, 10]))
        return text[:m.start()] + new + text[m.end():]
    # 3) negación del primer verbo auxiliar común
    for aux, neg in ((" was ", " was not "), (" were ", " were not "),
                     (" is ", " is not "), (" are ", " are not "),
                     (" showed ", " did not show "), (" found ", " did not find ")):
        if aux in text:
            return text.replace(aux, neg, 1)
    return None


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


def build_battery(docs, n_pairs, seed, max_len):
    """Construye, por doc elegible, ctx/ok compartidos y un `bad` por nivel.
    Devuelve {level: [ {ctx,ok,bad,texts...} ]} en TEXTO (sin tokenizar)."""
    rng = random.Random(seed)
    # candidatos: (idx_doc, j, stages, texts)
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
        # L0: otro doc, misma posición de etapa
        for _ in range(30):
            ci2 = rng.randrange(len(cands))
            if ci2 != ci:
                break
        i2, j2, st2, tx2 = cands[ci2]
        bad_l0 = f"[{st[j].upper()}] {tx2[min(j, len(tx2) - 1)]}"
        # L1: otro doc del MISMO dominio
        pool = by_dom.get(docs[i].get("domain", "?"), [])
        pool = [c for c in pool if c != ci]
        if pool:
            ci1 = rng.choice(pool)
            i1, j1, st1, tx1 = cands[ci1]
            bad_l1 = f"[{st[j].upper()}] {tx1[min(j, len(tx1) - 1)]}"
        else:
            bad_l1 = bad_l0
            fallback["L1_no_domain_pool"] += 1
        # L2: misma doc, etapa distinta (preferir etapa FUTURA j'>j = orden roto)
        js = [x for x in range(len(st)) if x != j]
        fut = [x for x in js if x > j]
        jp = rng.choice(fut if fut else js)
        bad_l2 = f"[{st[j].upper()}] {tx[jp]}"
        # L3: payload mutado de la etapa real
        mut = _mutate_payload(tx[j], rng)
        bad_l3 = f"[{st[j].upper()}] {mut}" if mut else None
        rec = {"ctx": ctx, "ok": ok}
        out["L0"].append({**rec, "bad": bad_l0})
        out["L1"].append({**rec, "bad": bad_l1})
        out["L2"].append({**rec, "bad": bad_l2})
        if bad_l3:
            out["L3"].append({**rec, "bad": bad_l3})
        else:
            fallback["L3_no_mutable_payload"] += 1
    return out, fallback


def _tok_write(levels, encode, outdir, max_len):
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"levels": {}, "overlap": {}}
    for lvl, recs in levels.items():
        kept, ov_ok, ov_bad = [], [], []
        for r in recs:
            ids = {k: encode(v) for k, v in r.items()}
            if max(len(v) for v in ids.values()) > max_len:
                continue
            kept.append(ids)
            ov_ok.append(_jaccard(ids["ctx"], ids["ok"]))
            ov_bad.append(_jaccard(ids["ctx"], ids["bad"]))
        fp = outdir / f"pairs_{lvl}.jsonl"
        with open(fp, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        meta["levels"][lvl] = {"n": len(kept), "file": str(fp)}
        meta["overlap"][lvl] = {
            "jaccard_ctx_ok": round(sum(ov_ok) / len(ov_ok), 4) if ov_ok else None,
            "jaccard_ctx_bad": round(sum(ov_bad) / len(ov_bad), 4) if ov_bad else None,
        }
        print(f"[{lvl}] {len(kept)} pares -> {fp}  "
              f"jac(ctx,ok)={meta['overlap'][lvl]['jaccard_ctx_ok']} "
              f"jac(ctx,bad)={meta['overlap'][lvl]['jaccard_ctx_bad']}")
    return meta


def diagnose(path):
    """Diagnóstico de overlap sobre un jsonl de pares YA tokenizado (0 GPU).
    Si jac(ctx,bad) << jac(ctx,ok), la tarea es explotable por coherencia."""
    n = 0
    so, sb, diff = 0.0, 0.0, 0.0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a, b, c = r["ctx"], r["ok"], r["bad"]
            so += _jaccard(a, b)
            sb += _jaccard(a, c)
            diff += _jaccard(a, b) - _jaccard(a, c)
            n += 1
    if not n:
        print("sin pares"); return 1
    print(f"{path}: n={n}")
    print(f"  jac(ctx,ok)  medio = {so / n:.4f}")
    print(f"  jac(ctx,bad) medio = {sb / n:.4f}")
    print(f"  margen ok-bad      = {diff / n:.4f}  "
          f"(>0 cuantifica el atajo temático disponible en L0)")
    return 0


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
        {"domain": "bio", "text":
         "[OBSERVACION] cells grew slower. "
         "[HIPOTESIS] the mutation impairs division. "
         "[PREDICCION] mutants show reduced colonies. "
         "[EVIDENCIA] colonies increased 12% in mutants. "
         "[CONCLUSION] mutation is tolerated."},
    ]
    enc = lambda s: [abs(hash(w)) % 30000 for w in s.split()]
    levels, fb = build_battery(docs, n_pairs=8, seed=0, max_len=2048)
    for lvl, recs in levels.items():
        for r in recs:
            ids = {k: enc(v) for k, v in r.items()}
            assert ids["ctx"] and ids["ok"] and ids["bad"]
    assert len(levels["L2"]) == len(levels["L0"])
    print("selftest OK:", {k: len(v) for k, v in levels.items()}, "fallbacks:", dict(fb))
    # muestra una mutación L3 para inspección
    print("L3 ejemplo:", levels["L3"][0]["bad"][:140] if levels["L3"] else "—")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="train_skeleton.jsonl")
    ap.add_argument("--tokenizer", help="path al tokenizer (local)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=576)
    ap.add_argument("--out", default="runs/pairs_hard")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--diagnose", metavar="PAIRS.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.diagnose:
        return diagnose(a.diagnose)
    if not a.inp or not a.tokenizer:
        ap.error("se requiere --in y --tokenizer (o --diagnose/--selftest)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=True)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    docs = list(_iter_docs(a.inp, a.limit_docs))
    print(f"docs cargados: {len(docs)}")
    levels, fb = build_battery(docs, a.n, a.seed, a.max_len)
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
