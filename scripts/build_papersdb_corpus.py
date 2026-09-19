#!/usr/bin/env python3
"""build_papersdb_corpus.py — corpus de pretrain v5 desde papers_db (1B run).

Filtra papers_db (DB canónica, NO es corpus de entrenamiento directo):
  - lang == "en" (lección g8-c1: ES crudo degrada)
  - texto mínimo --min-chars (descarta stubs/abstracts vacíos)
Salida: JSONL {"text","key","kind","domain","source"} listo para pretok.

El pretok corre con --split-long: los fulltexts se cortan en ventanas de
768 tok PRESERVANDO todos los tokens (no truncar — un fulltext ~10K tok
daría ~13 ventanas). Así el corpus rinde su ~10.5B tok estimado.

Mezcla opcional: --skel corpus_v4hneg.jsonl añade los esqueletos
(structural signal, ~132M tok ≈ 1.3% — no diluye).

Uso:
  python3 build_papersdb_corpus.py \
      --db data/papersdb/papers_db.jsonl --out data/corpus_v5_pdb.jsonl \
      [--skel data/corpus_v4hneg.jsonl] [--min-chars 200] [--max-docs N]
"""
import argparse, json, time


def emit(fo, text, key, kind, domain, source):
    fo.write(json.dumps({"text": text, "key": key, "kind": kind,
                         "domain": domain, "source": source},
                        ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skel", default=None)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    n_in = n_out = n_skip_lang = n_skip_len = 0
    kinds = {}
    with open(args.out, "w") as fo:
        for line in open(args.db):
            n_in += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("lang") != "en":
                n_skip_lang += 1
                continue
            text = (r.get("text") or "").strip()
            if len(text) < args.min_chars:
                n_skip_len += 1
                continue
            emit(fo, text, r.get("key"), r.get("kind"), r.get("domain"), r.get("source"))
            k = r.get("kind") or "?"
            kinds[k] = kinds.get(k, 0) + 1
            n_out += 1
            if args.max_docs and n_out >= args.max_docs:
                break
            if n_in % 400000 == 0:
                print(f"[pdb] {n_in} leidos, {n_out} emitidos ({time.time()-t0:.0f}s)",
                      flush=True)
        n_skel = 0
        if args.skel:
            for line in open(args.skel):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("lang", "en") != "en":
                    continue
                emit(fo, r.get("text", ""), r.get("key"), "skeleton",
                     r.get("domain"), r.get("src", "skel"))
                n_skel += 1
    print(f"[pdb] DONE in={n_in} out={n_out} skel={n_skel} "
          f"skip_lang={n_skip_lang} skip_len={n_skip_len} "
          f"kinds={kinds} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
