#!/usr/bin/env python3
"""unam_en_to_jsonl.py — convierte unam_md_en/*.en.md a filas corpus jsonl.

Salida: {"pid","domain","lang","source","text"} por doc. Filtra docs vacíos
o degenerados (<500 chars tras strip).
Uso: python3 unam_en_to_jsonl.py --indir /beegfs/.../unam_md_en --out .../unam_en.jsonl
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = skip = 0
    with open(a.out, "w") as fo:
        for p in sorted(glob.glob(os.path.join(a.indir, "*.en.md"))):
            txt = open(p, encoding="utf-8").read().strip()
            if len(txt) < 500:
                skip += 1
                continue
            pid = os.path.basename(p)[:-6]  # quita .en.md
            fo.write(json.dumps({
                "pid": pid, "domain": "unam", "lang": "en",
                "source": "unam-en", "text": txt,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[unam_en] {n} docs -> {a.out} (skip={skip})", flush=True)


if __name__ == "__main__":
    main()
