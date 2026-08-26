#!/usr/bin/env python3
"""merge_corpus.py — fusiona abstracts + full-text PMC en un corpus unico.

Regla de prioridad: para un mismo PMID, se conserva el FULL-TEXT (mas rico) y
se descarta el abstract duplicado. Se conservan los abstracts sin full-text.
Salida: train_corpus.jsonl (shuffled), campos unificados:
  {"text","pmid","domain","year","source"}  source: pmc-full | pubmed-abstract

Uso: python3 merge_corpus.py --abstracts eco_corpus.jsonl --fulltext fulltext_corpus.jsonl \
        --out train_corpus.jsonl [--shuffle] [--seed 42]
"""
import argparse, json, random, time
from pathlib import Path

def read_jsonl(path):
    for line in open(path):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstracts", default="/beegfs/a474r867/ecoreasoner/data/eco_corpus.jsonl")
    ap.add_argument("--fulltext", default="/beegfs/a474r867/ecoreasoner/data/fulltext_corpus.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus.jsonl")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    # 1) cargar fulltext primero (prioridad): pmid -> doc
    ft = {}
    for d in read_jsonl(args.fulltext):
        pmid = str(d.get("pmid"))
        if pmid and d.get("text"):
            ft[pmid] = {"text": d["text"], "pmid": pmid,
                        "domain": None, "year": None, "source": "pmc-full"}
    print(f"[{time.strftime('%H:%M:%S')}] fulltext: {len(ft)} pmids", flush=True)

    # 2) abstracts: conservar solo los que NO tienen fulltext
    ab = []
    n_skip = 0
    for d in read_jsonl(args.abstracts):
        pmid = str(d.get("pmid"))
        if pmid in ft:
            n_skip += 1  # duplicado, fulltext gana
            continue
        text = (d.get("text") or "").strip()
        if len(text) < 50:
            continue
        ab.append({"text": text, "pmid": pmid, "domain": d.get("domain"),
                   "year": d.get("year"), "source": "pubmed-abstract"})
    print(f"[{time.strftime('%H:%M:%S')}] abstracts: {len(ab)} (skip {n_skip} fulltext duplicados)", flush=True)

    # 3) merge
    merged = list(ft.values()) + ab
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)

    # tokenizar nota: sin tokenizer aqui, usar chars/4 como estimado
    total_chars = sum(len(x["text"]) for x in merged)
    out = Path(args.out)
    with open(out, "w") as f:
        for x in merged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {len(merged)} docs, "
          f"{total_chars/1e6:.0f}M chars ~{total_chars/4/1e6:.0f}M tokens -> {out}", flush=True)

if __name__ == "__main__":
    main()