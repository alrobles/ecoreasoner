#!/usr/bin/env python3
"""port_pubmed_parquet.py — porta PubMed SQLite (articles, 30.8M rows) a Parquet
particionado por año, usando DuckDB. Habilita búsquedas complejas columnar
(filtros multi-campo, agregaciones, muestreo estratificado) que SQLite FTS5 no da.

Fuente: /beegfs/a474r867/litdump/pubmed/pubmed_full.db (tabla articles)
Destino: /beegfs/a474r867/litdump/pubmed/parquet/ (year=YYYY/*.parquet) + catalog

Uso: python3 port_pubmed_parquet.py [--db ...] [--out ...] [--dry]
"""
import argparse, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/beegfs/a474r867/litdump/pubmed/pubmed_full.db")
    ap.add_argument("--out", default="/beegfs/a474r867/litdump/pubmed/parquet")
    ap.add_argument("--dry", action="store_true", help="solo contar y estimar, no escribir")
    args = ap.parse_args()

    import duckdb
    t0 = time.time()
    con = duckdb.connect()  # in-memory
    # sqlite_scan extension lee el SQLite directamente (necesita el archivo local)
    con.execute("INSTALL sqlite_scanner; LOAD sqlite_scanner;")
    q = f"""
        SELECT pmid, year, journal, title, abstract, mesh
        FROM sqlite_scan('{args.db}', 'articles')
    """
    # saneamiento de año: TRY_CAST a INT (text vacíos/inválidos => NULL=>1900 para particionar)
    q2 = f"""
        SELECT pmid,
               COALESCE(CAST(TRY_CAST(year AS INTEGER) AS VARCHAR), '1900') AS year,
               journal, title, abstract, mesh
        FROM ({q})
    """
    if args.dry:
        n = con.execute(f"SELECT COUNT(*) FROM ({q2})").fetchone()[0]
        yrs = con.execute(f"SELECT MIN(CAST(year AS INT)), MAX(CAST(year AS INT)) FROM ({q2})").fetchone()
        print(f"DRY: {n} rows, años {yrs[0]}-{yrs[1]}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] portando a Parquet por año -> {out} ...", flush=True)
    con.execute(f"""
        COPY ({q2}) TO '{out}'
        (FORMAT PARQUET, PARTITION_BY year, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)
    """)
    # catálogo: total rows + tamaño
    total = con.execute(f"SELECT COUNT(*) FROM ({q2})").fetchone()[0]
    size = sum(f.stat().st_size for f in out.rglob("*.parquet"))
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {total} rows, {size/1e9:.1f} GB -> {out}", flush=True)

if __name__ == "__main__":
    main()