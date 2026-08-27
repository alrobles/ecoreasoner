#!/usr/bin/env python3
"""
concat_corpus_v4.py — Construye train_corpus_v4.jsonl.

Dataset v4 = train_corpus_v3.jsonl (1.01M docs previos) + los papers PMC fulltext
nuevos (pmc_corpus_v4/shard_*.jsonl, ~1.74M docs) + opcionalmente destilación.

merge: simple concatenación con shuffle. Cada doc:
  v3: {"text", ...}
  pmc: {"text", pmcid, year, license}
Salida: {"text": <texto>, "source": "v3"|"pmc-v4", ...metadata}

Regla: se conserva TODO (no hay dedupe por clave porque v3 ya está fusionado y
los PMC son full-text nuevos). Opcional --max-docs-pmc para balancear.

Uso (en cluster):
  python3 concat_corpus_v4.py --v3 data/train_corpus_v3.jsonl \
        --pmc data/pmc_corpus_v4 --out data/train_corpus_v4.jsonl --shuffle
"""
import argparse, json, glob, random, time, os


def read_jsonl(path, src_tag, add_meta=True):
    n = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        d["source"] = src_tag
        n += 1
        yield d
    # return n via generator final? use total outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v3.jsonl")
    ap.add_argument("--pmc", default="/beegfs/a474r867/ecoreasoner/data/pmc_corpus_v4")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v4.jsonl")
    ap.add_argument("--max-docs-pmc", type=int, default=0, help="0= todos, o limitar")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    out = open(a.out, "w")
    n_v3 = 0; n_pmc = 0

    # v3 primero
    for d in read_jsonl(a.v3, "v3"):
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_v3 += 1

    # pmc shards
    import glob
    files = sorted(glob.glob(os.path.join(a.pmc, "shard_*.jsonl")))
    for f in files:
        for d in read_jsonl(f, "pmc-v4"):
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_pmc += 1
            if a.max_docs_pmc and n_pmc >= a.max_docs_pmc:
                break
        if a.max_docs_pmc and n_pmc >= a.max_docs_pmc:
            break

    out.close()

    print(f"[concat] v3={n_v3} pmc={n_pmc} total={n_v3+n_pmc}, {time.time()-t0:.0f}s", flush=True)

    if a.shuffle:
        # shuffle en dos pasadas: leer todas las lineas, reordenar, escribir
        lines = open(a.out).readlines()
        random.seed(a.seed)
        random.shuffle(lines)
        with open(a.out, "w") as f:
            f.writelines(lines)
        print(f"[concat] shuffled {len(lines)} lines", flush=True)


if __name__ == "__main__":
    main()