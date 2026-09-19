#!/usr/bin/env python3
"""build_hneg_pairs.py — G11 hneg-data: pair-docs desde el grafo kNN.

Gen de datos hneg-data (EVOG9-V5-DESIGN §4 paso 3, Nemotron leccion 2):
corpus += pares knn_edges verificados. Mineria REAL (no sintetica):
para cada edge near-miss (A,B) del audit emb_v1, emite UN doc fusionado
que contiene los DOS esqueletos completos — dos argumentos paralelos del
mismo tema cuyos valores load-bearing difieren. Al enmascarar una posicion
mutable, la otra claim queda visible en la misma ventana -> el modelo debe
ligar el valor a SU argumento, no copiar el del vecino (binding).

Verificacion (leccion: hneg fuera de cobertura = label-noise):
  - el par solo cuenta si el diff doc-level incluye >=1 token de clase
    mutable expandida: conjunto de valores numericos distinto ENTRE docs
    (frontera `number`) o conjunto de flip-words distinto (`negation`,
    `direction`, causal, temporal).
  - ambos docs llevan etiquetas de etapa (esqueletos validos).
  - len(A)+len(B) <= MAXW palabras -> el par cabe en una ventana seq_len
    (pre_tokenize trunca a 767; si no cabe, el contraste se rompe entre
    ventanas y el par no sirve).

Dedup: edge no-dir (min,max); cada doc participa en <=MAX_PER pair-docs
(el "contestedness" manda, pero sin dejar que un doc domine).

Salida: --out hneg_pairs.jsonl con docs {"text": skelA\n\nskelB (orden
aleatorio), "src":"hneg_knn", "lang":"en", "domain", "pair":[pidA,pidB],
"sim": sim} + reporte stats.

Uso:
  python3 build_hneg_pairs.py --out data/hneg_pairs.jsonl \
      [--min-sim 0.93] [--max-docs 40000] [--maxw 560] [--max-per 2]
"""
import argparse, json, os, re, random
from collections import Counter

BASE = "/beegfs/a474r867/ecoreasoner"
IDX_PATH = f"{BASE}/emb_v1/skeleton_v2/idx.jsonl"
EDGES_PATH = f"{BASE}/emb_v1/audit_skeleton_v2/knn_edges.tsv"
DOCS_PATH = f"{BASE}/data/skeleton/train_skeleton_train.jsonl"

_STAGE_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
_NUM_TOK_RE = re.compile(r"^[+-]?(\d+([.,]\d+)?|\d*\.\d+)(%)?$|^\d+(\.\d+)?%$")
_FLIP_WORDS = {
    # negacion / direccion / causal / temporal / reporte (clase mutable expandida)
    "not", "no", "never", "cannot", "without", "nor", "neither", "fails",
    "fail", "failed", "increase", "increases", "increased", "decrease",
    "decreases", "decreased", "reduces", "reduced", "reduce", "inhibits",
    "promotes", "enhances", "enhanced", "suppresses", "suppressed",
    "improves", "improved", "worsens", "worsened", "higher", "lower",
    "more", "less", "fewer", "above", "below", "before", "after",
    "positive", "negative", "significant", "greater", "smaller",
    "associated", "unrelated", "presence", "absence", "raises", "lowers",
    "stronger", "weaker", "elevated", "larger", "exceeds", "gain", "loss",
    "however", "therefore", "thus", "nevertheless", "conversely",
    "consequently", "because", "despite", "although", "hence", "whereas",
    "nonetheless", "moreover", "furthermore", "instead", "early", "late",
    "initial", "final", "initially", "finally", "acute", "chronic",
    "rapidly", "gradually", "rapid", "gradual", "long", "short", "longer",
    "shorter", "earlier", "later", "temporary", "permanent", "transient",
    "persistent", "delayed", "immediate", "brief", "prolonged", "supports",
    "contradicts", "confirms", "refutes", "consistent", "inconsistent",
}


def mutable_sets(text):
    """(nums, flips): conjuntos de valores numericos y flip-words del doc."""
    nums, flips = set(), set()
    for w in text.split():
        wl = w.lower().strip(".,;:()[]")
        if _NUM_TOK_RE.match(w):
            nums.add(w)
        if wl in _FLIP_WORDS:
            flips.add(wl)
    return nums, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BASE}/data/hneg_pairs.jsonl")
    ap.add_argument("--min-sim", type=float, default=0.93)
    ap.add_argument("--max-docs", type=int, default=40000)
    ap.add_argument("--maxw", type=int, default=560,
                    help="max palabras A+B para caber en la ventana 768")
    ap.add_argument("--max-per", type=int, default=2,
                    help="max pair-docs por doc fuente")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    stats = Counter()

    # 1) edges >= min_sim, dedup no-dir, orden por sim desc
    edges = {}
    with open(EDGES_PATH) as f:
        next(f)
        for line in f:
            s, d, sim = line.split("\t")
            sim = float(sim)
            if sim < a.min_sim:
                continue
            s, d = int(s), int(d)
            if s == d:
                continue
            k = (min(s, d), max(s, d))
            if k not in edges or sim > edges[k]:
                edges[k] = sim
    cand = sorted(edges.items(), key=lambda kv: -kv[1])
    print(f"[hneg] edges>= {a.min_sim} dedup: {len(cand)}", flush=True)

    # 2) rows de idx -> cargar solo los docs implicados
    idx = [json.loads(l) for l in open(IDX_PATH)]
    need = set()
    pool = cand[: a.max_docs * 12]         # pool de candidatos (sobra al filtrar)
    for (s, d), _ in pool:
        need.add(s); need.add(d)
    rows_needed = {idx[i]["row"]: i for i in need if i < len(idx)}
    texts = {}
    with open(DOCS_PATH) as f:
        for n, line in enumerate(f):
            if n in rows_needed:
                try:
                    texts[rows_needed[n]] = json.loads(line)["text"]
                except json.JSONDecodeError:
                    pass
            if len(texts) == len(rows_needed):
                break
    print(f"[hneg] docs cargados: {len(texts)}", flush=True)

    # 3) pre-computar sets mutables + longitudes
    info = {}
    for i, t in texts.items():
        if not _STAGE_RE.search(t):
            continue
        nums, flips = mutable_sets(t)
        info[i] = (len(t.split()), nums, flips)
    print(f"[hneg] docs con etapas: {len(info)}", flush=True)

    # 4) emitir pair-docs hasta cap
    per_doc = Counter()
    out = open(a.out, "w")
    n_em = 0
    for (s, d), sim in pool:
        if n_em >= a.max_docs:
            break
        ia, ib = info.get(s), info.get(d)
        if not ia or not ib:
            stats["skip_nodoc"] += 1
            continue
        if ia[0] + ib[0] > a.maxw:
            stats["skip_len"] += 1
            continue
        if per_doc[s] >= a.max_per or per_doc[d] >= a.max_per:
            stats["skip_perdoc"] += 1
            continue
        num_diff = (ia[1] - ib[1]) or (ib[1] - ia[1])
        flip_diff = (ia[2] - ib[2]) or (ib[2] - ia[2])
        if not num_diff and not flip_diff:
            stats["skip_nomut"] += 1
            continue
        stats["num_diff" if num_diff else "flip_only"] += 1
        ta, tb = texts[s], texts[d]
        (t1, i1), (t2, i2) = ((ta, s), (tb, d)) if rng.random() < 0.5 \
            else ((tb, d), (ta, s))
        rec = {
            "text": t1 + "\n\n" + t2,
            "src": "hneg_knn", "lang": "en",
            "domain": idx[i1].get("domain"),
            "pair": [idx[s].get("pid"), idx[d].get("pid")],
            "sim": round(sim, 4),
            "n_num_diff": len(num_diff), "n_flip_diff": len(flip_diff),
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        per_doc[s] += 1; per_doc[d] += 1
        n_em += 1
        stats["emitted"] += 1
    out.close()
    stats["total_out"] = n_em
    stats["edges_pool"] = len(pool)
    rep = dict(stats)
    print("[hneg] DONE", json.dumps(rep, indent=1)[:3000], flush=True)
    with open(a.out.replace(".jsonl", ".stats.json"), "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
