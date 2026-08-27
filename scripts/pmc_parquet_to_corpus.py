#!/usr/bin/env python3
"""
pmc_parquet_to_corpus.py — Convierte los 55 parquet PMC (1.74M papers full-text)
en un corpus de entrenamiento jsonl para el dataset v4.

Salida por shard: data/pmc_corpus_v4/shard_<i>.jsonl con
  {"text": <texto completo>, "pmcid": "PMC...", "year": N, "license": "..."}
El texto se limpia de espacios excesivos y saltos de línea repetidos (conserva
párrafos), recortado a un máximo de chars por paper (opcional --max-chars).
La salida se escribe por shard para permitir merge incremental y paralelismo.

Uso (en el cluster donde están los parquet):
  python3 pmc_parquet_to_corpus.py --parquet data/pmc_parquet --out data/pmc_corpus_v4
"""
import argparse, json, os, glob, re, time


def clean_text(t):
    if not t:
        return ""
    # normaliza espacios excesivos conservando párrafos/nuevas líneas
    t = re.sub(r"[ \t]+", " ", t)          # colapsa espacios múltiples en línea
    t = re.sub(r"\n{3,}", "\n\n", t)       # máx 1 línea en blanco entre párrafos
    t = re.sub(r" +\n", "\n", t)           # sin espacios finales de línea
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/beegfs/a474r867/ecoreasoner/data/pmc_parquet")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/pmc_corpus_v4")
    ap.add_argument("--max-chars", type=int, default=20000, help="recorte por paper (0=sin límite)")
    ap.add_argument("--min-chars", type=int, default=500, help="descartar papers < N chars")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.parquet, "shard_*.parquet")))
    print(f"[conv] {len(files)} parquet -> {a.out}", flush=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("need pyarrow"); raise

    t0 = time.time(); total = 0; dropped = 0
    for f in files:
        out_name = os.path.join(a.out, os.path.basename(f).replace(".parquet", ".jsonl"))
        t = pq.read_table(f)  # columnas: pmcid, year, license, text
        rows = t.to_pylist()
        with open(out_name, "w") as fh:
            for r in rows:
                text = clean_text(r.get("text") or "")
                if len(text) < a.min_chars:
                    dropped += 1
                    continue
                if a.max_chars and len(text) > a.max_chars:
                    # recorta en límite de bloque (~última línea completa)
                    text = text[:a.max_chars]
                    cut = text.rfind("\n")
                    if cut > a.max_chars * 0.8:
                        text = text[:cut]
                doc = {
                    "text": text,
                    "pmcid": r.get("pmcid"),
                    "year": r.get("year"),
                    "license": r.get("license"),
                }
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {os.path.basename(f)}: +{total} (drop {dropped}, {time.time()-t0:.0f}s)", flush=True)

    print(f"[conv] DONE {total} docs, {dropped} dropped, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()