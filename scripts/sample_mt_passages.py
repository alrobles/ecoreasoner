"""sample_mt_passages.py — muestrea N pasajes del corpus para qa_gen_mt.

Corre EN EL CLUSTER (login/compute). Fuente: corpus_v5_pdb.jsonl (44GB,
los docs crudos del pretrain v5). Reservoir-sample con preferencia a docs
con >=2 stage-labels ([OBSERVACION]/[HIPOTESIS]/...): estructura
argumentativa -> follow-ups mas ricos. Fallback: docs largos cualquiera.

Salida: jsonl {"pid","text"} — el formato que load_passages() espera.

Uso:
  python3 sample_mt_passages.py --src data/corpus_v5_pdb.jsonl \
      --out data/mt_passages_v4.jsonl --n 30000
"""
import argparse
import json
import random
import re
import sys

STAGE_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="pasajes mas largos se truncan (el gen ya subdivide por tokens)")
    ap.add_argument("--seed", type=int, default=7331)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # dos reservorios: stage-labeled y resto
    res_st, res_ot = [], []
    n_st = n_ot = 0
    kept_total = 0
    for i, line in enumerate(open(a.src, encoding="utf-8", errors="replace")):
        if i % 200000 == 0:
            print(f"  ...{i} lineas, kept={kept_total}", file=sys.stderr, flush=True)
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("text") or ""
        if len(t) < a.min_chars:
            continue
        t = t[: a.max_chars]
        pid = r.get("pid") or r.get("id") or f"doc{i}"
        n_labels = len(STAGE_RE.findall(t))
        kept_total += 1
        if n_labels >= 2:
            n_st += 1
            if len(res_st) < a.n:
                res_st.append({"pid": pid, "text": t})
            else:
                j = rng.randrange(n_st)
                if j < a.n:
                    res_st[j] = {"pid": pid, "text": t}
        else:
            n_ot += 1
            if len(res_ot) < a.n:
                res_ot.append({"pid": pid, "text": t})
            else:
                j = rng.randrange(n_ot)
                if j < a.n:
                    res_ot[j] = {"pid": pid, "text": t}

    pool = res_st + res_ot[: max(0, a.n - len(res_st))]
    rng.shuffle(pool)
    pool = pool[: a.n]
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in pool:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE: {len(pool)} pasajes -> {a.out} "
          f"(stage-labeled {len(res_st)} de {n_st} vistos, otros {n_ot})", file=sys.stderr)


if __name__ == "__main__":
    main()
