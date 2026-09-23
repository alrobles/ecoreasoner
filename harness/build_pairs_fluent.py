#!/usr/bin/env python3
"""build_pairs_fluent.py — pares L3 'fluidez-negativa' (contraste C1).

Variante estricta de build_pairs_hard_v3: SOLO mutaciones gramaticalmente
naturales por construcción, para que el par NO se decida por detección de
superficie sino por verificación contra el contexto.

Reglas:
  - Clases permitidas: direction, environment, mechanism, temporal,
    number, entity, negation (inserciones auxiliares fluidas).
  - EXCLUIDO: causal (because→despite etc. produce roturas gramaticales).
  - number: preserva formato (int→int, dec→dec misma precisión) y en
    contextos de conteo (patients, samples, ...) exige entero >= 1 —
    evita el artefacto '3.25 patients' que delataba la mutación.
  - Post-checks del candidato mutado: concordancia a/an, sin dobles
    espacios, capitalización tras punto, puntuación pegada, paréntesis
    balanceados.

Uso: igual que build_pairs_hard_v3; añade --fluent-min para n mínimo.
"""
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

_LABELS = ["OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"]
_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_COUNT_NOUN_RE = re.compile(
    r"\b(patients|samples|specimens|sites|plots|individuals|replicates|"
    r"trials|nests|trees|cells|mice|birds|fish|participants|subjects|"
    r"stations| transects| quadrats)\b", re.I)

# --- mapas de mutación (solo swaps cerrados de la misma categoría) ---
_DIR_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"), ("increases", "decreases"),
    ("higher", "lower"), ("greater", "smaller"), ("elevated", "reduced"),
    ("reduced", "increased"), ("improved", "worsened"), ("improve", "worsen"),
    ("upregulated", "downregulated"), ("activated", "inhibited"),
    ("promotes", "inhibits"), ("enhanced", "suppressed"),
    ("positive", "negative"), ("associated", "unrelated"),
    ("significant", "non-significant"), ("supports", "contradicts"),
    ("gain", "loss"), ("presence", "absence"), ("survival", "mortality"),
    ("more", "less"), ("rose", "fell"), ("faster", "slower"),
    ("stronger", "weaker"), ("positive effect", "negative effect"),
    ("inhibitory", "promoting"),
]
_DIR_MAP = {}
for _a, _b in _DIR_PAIRS:
    _DIR_MAP[_a.lower()] = _b; _DIR_MAP[_b.lower()] = _a

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
    _MECH_MAP[_a.lower()] = _b; _MECH_MAP[_b.lower()] = _a

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
    _ENV_MAP[_a.lower()] = _b; _ENV_MAP[_b.lower()] = _a

_TEMP_PAIRS = [
    ("short-term", "long-term"), ("acute", "chronic"),
    ("temporary", "permanent"), ("rapidly", "gradually"),
    ("1 year", "5 years"), ("2 years", "10 years"),
    ("early", "late"), ("initial", "final"),
]
_TEMP_MAP = {}
for _a, _b in _TEMP_PAIRS:
    _TEMP_MAP[_a.lower()] = _b; _TEMP_MAP[_b.lower()] = _a


def _load_species(species_path):
    try:
        d = json.loads(Path(species_path).read_text())
        return [s["canonical"] for s in d.get("species", [])]
    except Exception:
        return []


def _parse_stages(text):
    """[(label, span_text)] del formato [ETAPA] ..."""
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


def _try_replace(text, mapping, subtype):
    """Primera clave presente → swap bidireccional. Conserva whitespace."""
    keys = sorted(mapping, key=len, reverse=True)
    for key in keys:
        pat = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        m = pat.search(text)
        if m:
            new = mapping[m.group(0).lower()]
            if m.group(0)[0].isupper():
                new = new[0].upper() + new[1:]
            kind = "phrase" if " " in key else "word"
            return text[:m.start()] + new + text[m.end():], f"{subtype}_{kind}"
    return None, None


def _fmt_like(orig, val):
    """Formatea val como orig: int→int, dec→dec misma precisión."""
    if "." in orig:
        dec = len(orig.split(".")[1])
        return f"{val:.{dec}f}"
    return str(int(round(val)))


def _mutate_number(text, rng):
    """Mutación de magnitud preservando formato y validez de conteo."""
    for m in _NUM_RE.finditer(text):
        orig = m.group(0)
        try:
            v = float(orig)
        except ValueError:
            continue
        if v == 0:
            continue
        f = rng.choice([1.5, 2.0, 2.5, 3.0, 0.5, 0.4, 0.33, 0.25])
        new_s = _fmt_like(orig, v * f)
        if new_s == orig:
            continue
        # contexto de conteo: exigir entero >= 1
        tail = text[m.end():m.end() + 60]
        if _COUNT_NOUN_RE.search(tail):
            iv = int(round(v * f))
            if iv < 1 or str(iv) == orig:
                continue
            new_s = str(iv)
        return text[:m.start()] + new_s + text[m.end():], "number"
    return None, None


def _mutate_entity(text, species, rng):
    if not species:
        return None, None
    found = [s for s in species if s in text]
    if not found:
        return None, None
    src = rng.choice(found)
    others = [s for s in species if s != src]
    if not others:
        return None, None
    return text.replace(src, rng.choice(others), 1), "entity"


def _mutate_negation(text):
    """Inserciones auxiliares fluidas (todas gramaticales)."""
    for aux, neg in ((" was ", " was not "), (" were ", " were not "),
                     (" is ", " is not "), (" are ", " are not "),
                     (" showed ", " did not show "), (" found ", " did not find "),
                     (" suggests ", " does not suggest "),
                     (" indicates ", " does not indicate ")):
        if aux in text:
            return text.replace(aux, neg, 1), "negation"
    return None, None


def _mutate_payload(text, species, rng):
    """Orden de prioridad; SOLO clases fluidas (sin causal)."""
    for fn in (
        lambda t: _try_replace(t, _MECH_MAP, "mechanism"),
        lambda t: _try_replace(t, _ENV_MAP, "environment"),
        lambda t: _try_replace(t, _TEMP_MAP, "temporal"),
        lambda t: _try_replace(t, _DIR_MAP, "direction"),
        lambda t: _mutate_entity(t, species, rng),
        lambda t: _mutate_number(t, rng),
        _mutate_negation,
    ):
        r = fn(text)
        if r[0]:
            return r
    return None, None


def _fluency_check(text):
    """Heurísticas de superficie; True si el texto parece natural."""
    if "  " in text:
        return False
    if re.search(r"\b(a|an|the)\s+(a|an|the)\b", text, re.I):
        return False
    if re.search(r"\ba\s+[aeiou]", text):  # 'a apple'
        return False
    if text.count("(") != text.count(")"):
        return False
    if re.search(r"\s[.,;:]", text):      # espacio antes de puntuación
        return False
    return True


def _fix_article(text):
    """Repara concordancia a/an tras la mutación (barato y seguro)."""
    text = re.sub(r"\b(a|A)\s+([aeiouAEIOU])", r"an \2", text)
    text = re.sub(r"\b(an|An)\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])",
                  r"a \2", text)
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
    ap.add_argument("--tokenizer", required=True, help="path al tokenizer LLaDA")
    ap.add_argument("--species", default="")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--domains", default="",
                    help="csv de dominios (vacío = todos); p.ej. phys-*")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        local_files_only=True,
                                        trust_remote_code=True)
    encode = lambda s: tok.encode(s, add_special_tokens=False)

    rng = random.Random(args.seed)
    species = _load_species(args.species)
    domains = set(args.domains.split(",")) if args.domains else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
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
            mut, subtype = _mutate_payload(stages[j][1], species, rng)
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
