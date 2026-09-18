#!/usr/bin/env python3
"""G10 paso 0: audita cobertura de la clase mutable (is_num/is_flip) contra las
mutaciones reales del eval pairwise.

Mide, por subtipo L3, que fraccion de los tokens que cambian entre `ok` y `bad`
cae dentro de la clase mutable actual del trainer vs. definiciones expandidas.

Sin cobertura, el hinge contrastivo (y el corrective) no pueden ensenar los
casos duros: es la leccion Nemotron — el objetivo no ensena lo que la tabla no ve.

Uso:
    python3 audit_mutable_coverage.py --pairs runs/pairs_hard_v3_eval \
        --tokenizer $TOK
"""
import argparse, json, os, re, sys
from collections import Counter
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_NUMBER_RE = re.compile(r"^[+-]?(\d+([.,]\d+)?|\d*\.\d+)(\s*%)?$")


def norm_tok(t):
    if not isinstance(t, str):
        return ""
    t = t.strip()
    for p in ("▁", "Ġ", "##", "Ċ"):
        if t.startswith(p):
            t = t[len(p):]
    return t


def rule_current(t):
    return bool(_NUMBER_RE.match(t))


def rule_has_digit(t):
    return bool(re.search(r"\d", t))


def rule_num_piece(t):
    """Digito o pieza atomica de numeros: punto/coma/%/signo/solo-digitos."""
    if re.search(r"\d", t):
        return True
    return bool(re.fullmatch(r"[.,%+\-−×x*/^=<>≤≥~ ]+", t))


# Tablas actuales del trainer (copia de train_mdlm_moe_v2.py)
_CORRUPT_FLIP_TEXT = {
    "not": ["also", "indeed", "often", "still"],
    "no": ["a", "the", "some"],
    "never": ["often", "sometimes", "usually", "always"],
    "cannot": ["can", "may", "could"],
    "without": ["with"],
    "nor": ["or", "and"],
    "neither": ["either", "both"],
    "fails": ["succeeds", "manages"],
    "fail": ["succeed", "manage"],
    "failed": ["succeeded", "managed"],
    "increases": ["decreases", "reduces"],
    "decreases": ["increases", "enhances"],
    "inhibits": ["promotes", "enhances"],
    "promotes": ["inhibits", "suppresses"],
    "enhances": ["suppresses", "reduces"],
    "suppresses": ["enhances", "promotes"],
    "improves": ["worsens", "impairs"],
    "reduces": ["increases", "raises"],
    "higher": ["lower"],
    "lower": ["higher"],
    "more": ["less", "fewer"],
    "less": ["more"],
    "above": ["below"],
    "below": ["above"],
    "before": ["after"],
    "after": ["before"],
    "positive": ["negative"],
    "negative": ["positive"],
    "significant": ["negligible", "marginal"],
}
_FLIP_SET = set(_CORRUPT_FLIP_TEXT)

# Expansion PROPUESTA (G10 paso 0): familias medidas del eval.
# Claves = tokens mutados observados en pairs_L3 (dev+holdout).
_PROPOSED_FLIP = set(_FLIP_SET) | {
    # direccion: inflecciones + familias nuevas
    "increased", "reduced", "decreased", "increase", "decrease",
    "greater", "smaller", "enhanced", "suppressed", "improve", "worsen",
    "improved", "worsened", "associated", "unrelated", "presence",
    "absence", "raises", "raises", "gain", "loss", "stronger", "weaker",
    "elevated", "reduced", "larger", "fewer", "exceeds", "exceeded",
    "surpasses", "outweighs", "dominates",
    # causal: conectivas contrastivas medidas
    "however", "therefore", "thus", "nevertheless", "conversely",
    "consequently", "because", "despite", "although", "hence",
    "accordingly", "moreover", "instead", "whereas", "while",
    "nonetheless", "furthermore", "otherwise",
    # temporal: familias medidas
    "early", "late", "initial", "final", "initially", "finally",
    "acute", "chronic", "rapidly", "gradually", "long", "short",
    "temporary", "permanent", "sudden", "gradual", "rapid", "slow",
    "delayed", "immediate", "earlier", "later", "sooner", "longer",
    "shorter", "transient", "persistent", "brief", "prolonged",
    # negacion auxiliares + verbos de reporte (polaridad)
    "did", "does", "found", "find", "showed", "show", "shows",
    "suggests", "suggest", "indicates", "indicate", "supports",
    "contradicts", "confirms", "refutes", "consistent", "inconsistent",
    "demonstrated", "demonstrates", "reveals", "revealed",
    # entorno/mecanismo (open-class: familias observadas mas frecuentes)
    "temperature", "light", "elevation", "latitude", "nitrogen",
    "phosphorus", "drought", "flooding", "salinity", "host",
    "parasite", "prey", "predator",
}
# Simbolos relacion (byte-BPE, claves ya en minusculas como el lookup):
# 'âł'~≤ 'âģī'~→ 'â±'=± 'âī¥'=≥ + > < = flechas.
_PROPOSED_FLIP |= {"âł", "âī¥", "âī¤", "âģī", "âĩĵ", "âĩĳ", "â±",
                   "<", ">", "=", "âĪā"}

RULES = {"current": rule_current, "has_digit": rule_has_digit,
         "num_piece": rule_num_piece,
         "is_flip": lambda t: t.lower() in _FLIP_SET,
         "mutable": lambda t: bool(_NUMBER_RE.match(t)) or t.lower() in _FLIP_SET,
         "proposed": lambda t: rule_num_piece(t) or t.lower() in _PROPOSED_FLIP}


def diff_ids(ok, bad):
    """Ids cambiados en cada lado + pares alineados 1:1 de 'replace'."""
    sm = SequenceMatcher(None, ok, bad, autojunk=False)
    ok_ch, bad_ch, pairs = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            ok_ch.extend(ok[i1:i2])
        if tag in ("replace", "insert"):
            bad_ch.extend(bad[j1:j2])
        if tag == "replace" and (i2 - i1) == (j2 - j1) == 1:
            pairs.append((ok[i1], bad[j1]))
    return ok_ch, bad_ch, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            trust_remote_code=True)
        vocab = tok.vocab_size
        strs = tok.convert_ids_to_tokens(list(range(vocab)))
    except Exception:
        # Fallback: tokenizer.json puro (sin transformers).
        with open(f"{args.tokenizer}/tokenizer.json") as f:
            tj = json.load(f)
        id2s = {}
        for s, i in tj["model"]["vocab"].items():
            id2s[i] = s
        for at in tj.get("added_tokens", []):
            id2s[at["id"]] = at["content"]
        vocab = max(id2s) + 1
        strs = [id2s.get(i, "") for i in range(vocab)]
    norm = [norm_tok(s) for s in strs]
    rule_hits = {name: [r(t) for t in norm] for name, r in RULES.items()}
    print(f"vocab={vocab} " + " ".join(
        f"{n}={sum(h)}" for n, h in rule_hits.items()))

    from collections import defaultdict
    sub_stats = {}
    mutated_strs = Counter()
    uncovered = {}   # subtype -> Counter de tokens mutados NO cubiertos
    pair_counts = defaultdict(Counter)  # subtype -> Counter (ok_str,bad_str)
    with open(f"{args.pairs}/pairs_L3.jsonl") as f:
        for line in f:
            r = json.loads(line)
            st = r.get("l3_subtype", "?")
            ok_ch, bad_ch, pairs = diff_ids(r["ok"], r["bad"])
            for o, b in pairs:
                pair_counts[st][(norm[o].lower(), norm[b].lower())] += 1
            d = sub_stats.setdefault(st, {"n": 0, "ok_ch": 0, "bad_ch": 0,
                                          **{f"ok_{k}": 0 for k in RULES},
                                          **{f"bad_{k}": 0 for k in RULES}})
            uc = uncovered.setdefault(st, Counter())
            d["n"] += 1
            for tid in ok_ch + bad_ch:
                mutated_strs[norm[tid]] += 1
            for tid in ok_ch:
                d["ok_ch"] += 1
                if not rule_hits["mutable"][tid]:
                    uc[norm[tid].lower()] += 1
                for k in RULES:
                    d[f"ok_{k}"] += rule_hits[k][tid]
            for tid in bad_ch:
                d["bad_ch"] += 1
                if not rule_hits["mutable"][tid]:
                    uc[norm[tid].lower()] += 1
                for k in RULES:
                    d[f"bad_{k}"] += rule_hits[k][tid]

    hdr = " ".join(f"{k:>9}" for k in RULES)
    print(f"\n{'subtype':<18}{'n':>5}{'chg':>7} | {hdr}")
    for st, d in sorted(sub_stats.items()):
        tot = d["ok_ch"] + d["bad_ch"]
        cov = {k: (d[f"ok_{k}"] + d[f"bad_{k}"]) / max(tot, 1)
               for k in RULES}
        print(f"{st:<18}{d['n']:>5}{tot:>7} | " +
              " ".join(f"{cov[k]*100:>8.1f}%" for k in RULES))

    print("\ntop-40 tokens mutados (normalizados):")
    for s, c in mutated_strs.most_common(40):
        print(f"  {c:>5}  {s!r}")

    print("\ntop tokens NO cubiertos por 'mutable' actual, por subtipo:")
    for st in sorted(uncovered):
        top = uncovered[st].most_common(12)
        if top:
            print(f"  {st}: " + ", ".join(f"{w!r}({c})" for w, c in top))

    print("\npares de mutacion 1:1 (ok -> bad), top-15 por subtipo:")
    for st in sorted(pair_counts):
        top = pair_counts[st].most_common(15)
        if top:
            print(f"  {st}:")
            for (o, b), c in top:
                print(f"    {c:>4}  {o!r} -> {b!r}")


if __name__ == "__main__":
    main()
