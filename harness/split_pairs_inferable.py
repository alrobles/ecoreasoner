#!/usr/bin/env python3
"""split_pairs_inferable.py — separa pares L3 en `inferable` vs `provenance-only`.

Motivo (docs/designs/REPLANTEO-2026-09-22 §1.1): en modo `dense` el candidato
va 100% enmascarado → un subtipo solo es ganable si el slot mutado está
DETERMINADO por el ctx o por el resto del candidato. Pares donde el valor
correcto lo es solo "porque así lo escribió el paper" (provenance) son
ruido para la frontera.

Heurística (transparente, auditable):
  - direction/causal/temporal/negation/mechanism/environment → INFERABLE
    (mutación semántica; ctx o coherencia interna la determinan)
  - number/entity → INFERABLE solo si
      (a) el span verdadero aparece literalmente en ctx o en el resto del
          candidato (consistencia por copia), o
      (b) number: la mutación produce no-entero donde el frame exige
          contador ("3.25 patients").
    Si no → PROVENANCE-ONLY.
  - diff que cubre >60% del candidato → PROVENANCE-ONLY (no es un slot).

Juez opcional (--endpoint OpenAI-compatible): muestra estratificada por
subtipo; pregunta si alguna de las dos continuaciones es detectablemente
errónea. Reporta % acuerdo con la heurística a stdout + judge_report.json.

Uso:
  python3 harness/split_pairs_inferable.py \
      --pairs runs/pairs_hard_v4_holdout_clean/pairs_L3.jsonl \
      --tokenizer-json /beegfs/.../LLaDA-8B-Instruct/.../tokenizer.json \
      --out runs/pairs_hard_v4_holdout_clean \
      [--endpoint http://nodo:port/v1/chat/completions --judge-model m \
       --judge-sample 200 --api-key-file ~/.openrouter-key]
"""
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

INFERABLE_SUBS = ("direction", "causal", "temporal", "negation",
                  "mechanism", "environment")
_COUNT_NOUNS = re.compile(
    r"\b(patients?|samples?|specimens?|cells?|individuals?|plots?|sites?|"
    r"mice|rats?|birds?|trees?|populations?|replicates?|clones?|nodes?|"
    r"species|genera|families|participants?|subjects?|isolates?|strains?|"
    r"experiments?|trials?|measurements?|observations?)\b", re.I)
_NUM = re.compile(r"^\d+\.\d+$")


def byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip([chr(c) for c in cs], bs))


def load_id2tok(tok_json):
    d = json.loads(Path(tok_json).read_text())
    return {v: k for k, v in d["model"]["vocab"].items()}


def make_decoder(id2tok):
    bd = byte_decoder()

    def dec(ids):
        s = "".join(id2tok.get(i, f"<{i}>") for i in ids)
        try:
            return bytearray([bd[c] for c in s]).decode("utf-8",
                                                        errors="replace")
        except KeyError:
            return s
    return dec


def _common_affixes(a, b):
    n, m = len(a), len(b)
    p = 0
    while p < n and p < m and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and s < m - p and a[n - 1 - s] == b[m - 1 - s]:
        s += 1
    return p, s


_WS = re.compile(r"\s+")


def _norm(t):
    return _WS.sub(" ", t)


def classify(rec, dec):
    """Devuelve (inferable: bool, razon: str).

    Trabaja sobre TEXTO decodificado con whitespace normalizado: el builder
    (_try_replace via split/join) normaliza \u2009/\xa0 en `bad`, lo que
    infla el diff a nivel de ids aunque la mutación sea de 1 token.
    """
    ctx, ok, bad = rec["ctx"], rec["ok"], rec["bad"]
    sub = rec.get("l3_subtype", "?")
    ok_t, bad_t, ctx_t = _norm(dec(ok)), _norm(dec(bad)), _norm(dec(ctx))
    p, s = _common_affixes(ok_t, bad_t)
    ok_span = ok_t[p:len(ok_t) - s]
    if len(ok_span) > 0.6 * len(ok_t):
        return False, "diff_grande"
    # consistencia por copia: el valor verdadero aparece literalmente en
    # ctx o en el resto visible del candidato (con fronteras de palabra)
    resto = ok_t[:p] + ok_t[len(ok_t) - s:]
    frag = ok_span.strip()
    if frag:
        pat = re.compile(r"(?<!\w)" + re.escape(frag) + r"(?!\w)")
        if pat.search(ctx_t) or pat.search(resto):
            return True, f"{sub}:copia"
    if sub.startswith(INFERABLE_SUBS):
        return True, f"{sub}:semantico"
    if sub == "number":
        # no-entero donde el frame exige contador -> inferable por tipo
        bad_span = bad_t[p:len(bad_t) - s]
        if _NUM.match(bad_span.strip()) and _COUNT_NOUNS.search(
                bad_t[len(bad_t) - s:][:60]):
            return True, "number:count_noun"
    return False, f"{sub}:provenance"


JUDGE_TMPL = """You are auditing an evaluation item.

CONTEXT (scientific argument prefix):
"{ctx}"

Two candidate continuations differ in exactly one place:
A: "{a}"
B: "{b}"

Question: is one of them detectably wrong — logically inconsistent with the
context, internally incoherent, or grammatically impossible — WITHOUT knowing
which was the original source text?
Answer with exactly one word: WRONG or PLAUSIBLE."""


def judge(endpoint, model, pairs, dec, sample_n, seed, key=None):
    import urllib.request
    rng = random.Random(seed)
    by_sub = defaultdict(list)
    for r in pairs:
        by_sub[r.get("l3_subtype", "?")].append(r)
    per = max(1, sample_n // max(1, len(by_sub)))
    sample = [r for sub in sorted(by_sub)
              for r in rng.sample(by_sub[sub], min(per, len(by_sub[sub])))]
    rows, agree = [], 0
    hdrs = {"Content-Type": "application/json"}
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    for i, r in enumerate(sample):
        a, b = (dec(r["ok"]), dec(r["bad"]))
        if rng.random() < 0.5:          # desordenar posición
            a, b = b, a
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": JUDGE_TMPL.format(
                    ctx=dec(r["ctx"])[-1500:], a=a[:1500], b=b[:1500])}],
                "max_tokens": 8}
        try:
            req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                         headers=hdrs)
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            txt = out["choices"][0]["message"]["content"].strip().upper()
            judge_inf = txt.startswith("WRONG")
        except Exception as e:
            txt = f"ERR:{e}"; judge_inf = None
        heur_inf, why = classify(r, dec)
        if judge_inf is not None:
            agree += int(judge_inf == heur_inf)
        rows.append({"sub": r.get("l3_subtype"), "heur": heur_inf,
                     "why": why, "judge": judge_inf, "raw": txt})
        print(f"[{i+1}/{len(sample)}] sub={r.get('l3_subtype')} "
              f"heur={heur_inf}({why}) judge={judge_inf} raw={txt!r}",
              flush=True)
    n_valid = sum(1 for x in rows if x["judge"] is not None)
    return {"n": len(rows), "n_valid": n_valid,
            "agreement": round(agree / n_valid, 4) if n_valid else None,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tokenizer-json", required=True,
                    help="tokenizer.json LLaDA (decode local, sin transformers)")
    ap.add_argument("--out", required=True,
                    help="dir de salida (pairs_L3_inferable/_provenance.jsonl)")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--judge-sample", type=int, default=200)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--seed", type=int, default=7331)
    a = ap.parse_args()

    id2tok = load_id2tok(a.tokenizer_json)
    dec = make_decoder(id2tok)
    pairs = [json.loads(l) for l in Path(a.pairs).read_text().splitlines()
             if l.strip()]
    print(f"pares: {len(pairs)}")

    inf, prov = [], []
    reasons = defaultdict(int)
    for r in pairs:
        ok_inf, why = classify(r, dec)
        reasons[why] += 1
        r2 = dict(r); r2["inferable"] = ok_inf; r2["infer_why"] = why
        (inf if ok_inf else prov).append(r2)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp_inf = out / "pairs_L3_inferable.jsonl"
    fp_prov = out / "pairs_L3_provenance.jsonl"
    fp_inf.write_text("".join(json.dumps(r) + "\n" for r in inf))
    fp_prov.write_text("".join(json.dumps(r) + "\n" for r in prov))
    print(f"inferable: {len(inf)}  provenance: {len(prov)}")
    by_sub = defaultdict(lambda: [0, 0])
    for r in inf:
        by_sub[r.get("l3_subtype", "?")][0] += 1
    for r in prov:
        by_sub[r.get("l3_subtype", "?")][1] += 1
    print("por subtipo (inf/prov):")
    for s in sorted(by_sub):
        i, pv = by_sub[s]
        print(f"  {s:20s} {i:5d} / {pv:5d}")
    print("razones:", dict(reasons))

    if a.endpoint and a.judge_model:
        key = Path(a.api_key_file).read_text().strip() if a.api_key_file else None
        rep = judge(a.endpoint, a.judge_model, pairs, dec,
                    a.judge_sample, a.seed, key)
        (out / "judge_report.json").write_text(json.dumps(rep, indent=2))
        print(f"juez: agreement={rep['agreement']} en n={rep['n_valid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
