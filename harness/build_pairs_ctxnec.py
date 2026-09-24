#!/usr/bin/env python3
"""build_pairs_ctxnec.py — pares L3 'contexto-necesario' (post-C4).

El hallazgo C4: con 50% del contexto corrompido, la accuracy pairwise apenas
cae (~3pts) → los pares actuales se resuelven por plausibilidad del
candidato, no por verificación contra el contexto.

Invariante de este builder: la alternativa mutada usa material ATESTIGUADO
EN EL CONTEXTO del mismo documento. Ambos candidatos son igual de fluentes
y contienen valores/entidades/direcciones REALES del documento — solo el
binding (qué valor pertenece a qué sujeto) decide la verdad. Un modelo que
rankea plausibilidad sin ligar contexto debe caer a azar; uno que verifica
mantiene accuracy y su curva C4 debe degradar con la corrupción.

Estrategias (todas requieren el material alternativo presente en ctx):
  - value_rebind    : candidato tiene número N, ctx tiene otro número M
                      con formato/contexto compatible → bad usa M.
                      ("X reduced 30%" → "X reduced 55%" cuando ctx dice
                      que Y reduced 55%.)
  - entity_rebind   : candidato menciona entidad E, ctx menciona E2 de la
                      misma clase (especie/binomio/término capitalizado)
                      → bad usa E2.
  - role_swap       : candidato "A <relación> B" con A,B en ctx → "B <rel> A".
  - direction_rebind: swap de dirección SOLO si el antónimo está atestiguado
                      en ctx (la dirección mala existe en el documento,
                      ligada a otro sujeto).

Si ningún material alternativo está atestiguado → doc descartado (yield
menor pero limpio: context-necessity por construcción).

Diagnóstico esperado: en estos pares la curva consistency_profile debe
CAER al corromper ctx (si el modelo realmente liga), y el eval sin ctx
(--limit + stub) debe dar ~0.5.

Uso: idéntico a build_pairs_fluent.py.
"""
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

_LABELS = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_BINOM_RE = re.compile(r"\b([A-Z][a-z]{3,} [a-z]{3,})\b")
_ACRO_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

# swaps de dirección; el antónimo DEBE estar atestiguado en ctx
_DIR_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"), ("increases", "decreases"),
    ("higher", "lower"), ("greater", "smaller"), ("elevated", "reduced"),
    ("reduced", "increased"), ("improved", "worsened"),
    ("upregulated", "downregulated"), ("activated", "inhibited"),
    ("promotes", "inhibits"), ("enhanced", "suppressed"),
    ("positive", "negative"), ("supports", "contradicts"),
    ("gain", "loss"), ("presence", "absence"), ("survival", "mortality"),
    ("more", "less"), ("rose", "fell"), ("faster", "slower"),
    ("stronger", "weaker"), ("inhibited", "promoted"),
]
_DIR_MAP = {}
for _a, _b in _DIR_PAIRS:
    _DIR_MAP[_a.lower()] = _b
    _DIR_MAP[_b.lower()] = _a

_STOP_ACRO = {"DNA", "RNA", "PCR", "USA", "UK", "CI", "OR", "SD", "SE",
              "NLM", "ISSN", "EISSN", "ID", "DOI", "URL", "HTTP", "HTTPS",
              "JOURNAL", "PMID", "PMC", "ET", "AL", "VS", "P", "N", "NS"}


def _load_species(species_path):
    try:
        d = json.loads(Path(species_path).read_text())
        return [s["canonical"] for s in d.get("species", [])]
    except Exception:
        return []


def _parse_stages(text):
    parts = _RE.split(text)
    out, cur, buf = [], None, []
    for seg in parts:
        if seg in _LABELS:
            if cur is not None:
                out.append((cur, " ".join(buf).strip()))
            cur, buf = seg, []
        elif cur is not None:
            buf.append(seg)
    if cur is not None:
        out.append((cur, " ".join(buf).strip()))
    return [(l, t) for l, t in out if t]


def _select_k(stages):
    for i, (l, _) in enumerate(stages):
        if l == "PREDICCION" and i + 1 < len(stages):
            return i + 1
    if len(stages) == 3:
        return 1
    if len(stages) > 3:
        return 2
    return None


def _trailing(text, m, n=1):
    """Carácter/signo tras el match (para compatibilidad %, unidad, etc.)."""
    return text[m.end():m.end() + n].lstrip()[:n]


def _mutate_value_rebind(cand, ctx, rng):
    """N en candidato → M atestiguado en ctx, formato/contexto compatible."""
    ctx_nums = [(m.group(0), _trailing(ctx, m)) for m in _NUM_RE.finditer(ctx)]
    for m in _NUM_RE.finditer(cand):
        orig, trail = m.group(0), _trailing(cand, m)
        bounded = orig.startswith("0.") or orig.startswith(".")
        cands = [v for v, t in ctx_nums
                 if v != orig and ("." in v) == ("." in orig)
                 and len(v) <= len(orig) + 2 and len(v) <= 7
                 and (t == trail or (t == "%") == (trail == "%"))
                 and (not bounded or float(v) < 1)]
        if not cands:
            continue
        return cand[:m.start()] + rng.choice(cands) + cand[m.end():], "value_rebind"
    return None, None


def _ctx_entities(ctx, species):
    # solo especies de la lista curada + acrónimos; los binomios por regex
    # producen demasiados falsos positivos ("Neurologic phenotype", ...)
    ents = set(s for s in species if s in ctx)
    ents.update(t for t in _ACRO_RE.findall(ctx) if t not in _STOP_ACRO)
    return ents


def _mutate_entity_rebind(cand, ctx, species, rng):
    # swap dentro de la misma clase: especie↔especie, acrónimo↔acrónimo
    pools = [
        [s for s in species if s in ctx],
        [t for t in _ACRO_RE.findall(ctx) if t not in _STOP_ACRO],
    ]
    for pool in pools:
        found = [e for e in pool if e in cand]
        if not found:
            continue
        src = rng.choice(found)
        others = [e for e in pool if e != src and e not in cand]
        if not others:
            continue
        return cand.replace(src, rng.choice(others), 1), "entity_rebind"
    return None, None


def _mutate_role_swap(cand, ctx, rng):
    """'A <verbo-rel> B' → 'B <verbo-rel> A' si A y B están en ctx."""
    ents = [e for e in _ctx_entities(ctx, []) if e in cand]
    if len(ents) < 2:
        return None, None
    ents.sort(key=cand.find)
    a, b = ents[0], ents[1]
    if not (a in ctx and b in ctx):
        return None, None
    ia, ib = cand.find(a), cand.find(b)
    out = cand[:ia] + b + cand[ia + len(a):ib] + a + cand[ib + len(b):]
    if out != cand:
        return out, "role_swap"
    return None, None


def _mutate_direction_rebind(cand, ctx):
    """Swap de dirección solo si el antónimo está atestiguado en ctx."""
    keys = sorted(_DIR_MAP, key=len, reverse=True)
    for key in keys:
        pat = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        m = pat.search(cand)
        if not m:
            continue
        new = _DIR_MAP[m.group(0).lower()]
        if not re.search(r"\b" + re.escape(new) + r"\b", ctx, re.IGNORECASE):
            continue  # antónimo no atestiguado → no es contexto-necesario
        if m.group(0)[0].isupper():
            new = new[0].upper() + new[1:]
        kind = "phrase" if " " in key else "word"
        return cand[:m.start()] + new + cand[m.end():], f"direction_rebind_{kind}"
    return None, None


def _mutate_ctxnec(cand, ctx, species, rng):
    for fn in (
        lambda: _mutate_value_rebind(cand, ctx, rng),
        lambda: _mutate_entity_rebind(cand, ctx, species, rng),
        lambda: _mutate_role_swap(cand, ctx, rng),
        lambda: _mutate_direction_rebind(cand, ctx),
    ):
        r = fn()
        if r[0]:
            return r
    return None, None


def _fluency_check(text):
    if "  " in text:
        return False
    if re.search(r"\b(a|an|the)\s+(a|an|the)\b", text, re.I):
        return False
    if re.search(r"\ba\s+[aeiou]", text):
        return False
    if text.count("(") != text.count(")"):
        return False
    if re.search(r"\s[.,;:]", text):
        return False
    if re.search(r"\b(and|the|of|in|to|a|an|with|for|by)\s+\1\b", text, re.I):
        return False  # palabra funcional duplicada ("and and")
    return True


def _fix_article(text):
    # solo minúsculas: las mayúsculas pueden ser acrónimos con pronunciación
    # vocálica ("an F1", "an SLE") — tocarlas rompe más de lo que arregla
    text = re.sub(r"\b(a|A)\s+([aeiou])", r"an \2", text)
    text = re.sub(r"\b(an|An)\s+([bcdfghjklmnpqrstvwxyz])", r"a \2", text)
    return text


def _iter_docs(path, domains=None):
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if domains and d.get("domain") not in domains:
            continue
        yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--species", default="")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--domains", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        local_files_only=True,
                                        trust_remote_code=True)
    encode = lambda s: tok.encode(s, add_special_tokens=False)

    rng = random.Random(args.seed)
    species = _load_species(args.species)
    domains = set(args.domains.split(",")) if args.domains else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "pairs_L3.jsonl"

    n_ok, n_nomut, n_flu, counts = 0, 0, 0, defaultdict(int)
    with fp.open("w") as f:
        for d in _iter_docs(args.src, domains):
            if n_ok >= args.n:
                break
            stages = _parse_stages(d["text"])
            if len(stages) < 2:
                continue
            j = _select_k(stages)
            if j is None:
                continue
            ctx = " ".join(f"[{l}] {t}" for l, t in stages[:j])
            cand = f"[{stages[j][0]}] {stages[j][1]}"
            if len(_NUM_RE.findall(stages[j][1])) > 8:
                continue  # zona de referencias/tablas: mutación sobre IDs
            mut, subtype = _mutate_ctxnec(stages[j][1], ctx, species, rng)
            if not mut:
                n_nomut += 1
                continue
            bad = _fix_article(f"[{stages[j][0]}] {mut}")
            if not _fluency_check(bad):
                n_flu += 1
                continue
            ids = {"ctx": encode(ctx), "ok": encode(cand), "bad": encode(bad)}
            if max(len(v) for v in ids.values()) > args.max_len:
                continue
            f.write(json.dumps({**ids, "l3_subtype": subtype,
                                "domain": d.get("domain")}) + "\n")
            counts[subtype] += 1
            n_ok += 1
    print(f"pairs={n_ok} nomut={n_nomut} fluency_rej={n_flu} -> {fp}")
    print(f"subtypes: {dict(counts)}")


if __name__ == "__main__":
    main()
