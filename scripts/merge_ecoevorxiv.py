#!/usr/bin/env python3
"""merge_ecoevorxiv.py — integra full-text EcoEvoRxiv en el corpus ampliado.

Concatena los ecoevorxiv_fulltext_c*.jsonl con el train_corpus_v2.jsonl
(abstracts+PMC) para producir train_corpus_v3.jsonl. Los preprints no usan
PMID, se identifican por article_id (prefijo 'EEV-'), fuente ecoevorxiv.

Prioridad: si un PMID ya está, el full-text PMC v2 gana. EcoEvoRxiv son
preprints independientes (article_id) -> se añaden como docs nuevos.

Uso: python3 merge_ecoevorxiv.py [--v2 ...] [--eev-glob ...] [--out ...] [--shuffle]
"""
import argparse, glob, json, random, time
from pathlib import Path

def read_jsonl(paths):
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v2.jsonl")
    ap.add_argument("--eev_glob", default="/beegfs/a474r867/ecoreasoner/data/ecoevorxiv_fulltext_c*.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v3.jsonl")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    # 1) ecoevorxiv full-text
    eev_files = sorted(glob.glob(args.eev_glob))
    print(f"[{time.strftime('%H:%M:%S')}] ecoevorxiv files: {eev_files}", flush=True)
    eev = []
    seen_eev = set()
    for d in read_jsonl(eev_files):
        aid = d.get("article_id")
        if not aid or aid in seen_eev:
            continue
        seen_eev.add(aid)
        txt = (d.get("text") or "").strip()
        if len(txt) < 100:
            continue
        eev.append({"text": txt, "pmid": f"EEV-{aid}", "domain": None,
                    "year": None, "source": d.get("source", "ecoevorxiv-pdf"),
                    "title": d.get("title")})
    print(f"[{time.strftime('%H:%M:%S')}] ecoevorxiv: {len(eev)} docs", flush=True)

    # 2) corpus v2 base
    v2 = list(read_jsonl([args.v2]))
    print(f"[{time.strftime('%H:%M:%S')}] v2 base: {len(v2)} docs", flush=True)

    # 3) merge: v2 + eev
    merged = v2 + eev
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)
    total_chars = sum(len(x["text"]) for x in merged)
    with open(args.out, "w") as f:
        for x in merged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {len(merged)} docs, "
          f"{total_chars/1e6:.0f}M chars ~{total_chars/4/1e6:.0f}M tokens -> {args.out}", flush=True)

if __name__ == "__main__":
    main()