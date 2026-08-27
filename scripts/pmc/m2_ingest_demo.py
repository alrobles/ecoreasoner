#!/usr/bin/env python3
"""
m2_ingest_demo.py — M2: Ingestor S3→Parquet (shard demo) + validación DuckDB.

Descarga el TEXTO de un subconjunto de PMCIDs del OA subset (desde `text_url`),
escribe Parquet por shard con columnas (pmcid, year, license, text), y valida
con DuckDB `SELECT count(*)`.

Uso (en kuhpc):
  python3 m2_ingest_demo.py --n 1000 --out pmc_demo.parquet --duckdb-check
"""
import argparse, json, re, time, os, random
import urllib.request

S3="https://pmc-oa-opendata.s3.amazonaws.com/"
UA={"User-Agent":"research/1.0"}

def get_text_url(url):
    """Descarga un objeto S3 (texto .txt) y devuelve el texto limpio."""
    try:
        h=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60).read().decode("utf-8","replace")
        return h
    except Exception:
        return ""

def get_metadata_item(pmcid, ver="1"):
    """Baja metadata JSON de un PMCID.ver para year/license/text_url."""
    url=f"{S3}metadata/{pmcid}.{ver}.json"
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=30))
        m=re.search(r"(\d{4})",d.get("citation",""))
        text_url=d.get("text_url","")  # s3://...; convertir a https
        # convertir s3:// a https://pmc-oa-opendata.s3.amazonaws.com/
        https=text_url.replace("s3://pmc-oa-opendata","https://pmc-oa-opendata.s3.amazonaws.com").split("?")[0] if text_url else ""
        return {"pmcid":pmcid,"ver":ver,"year":int(m.group(1)) if m else None,
                "license":d.get("license_code"),"text_url":https}
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=500,help="cuántos artículos descargar en el demo")
    ap.add_argument("--out",default="pmc_demo.parquet")
    ap.add_argument("--duckdb",action="store_true")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()

    # Tomar PMCIDs del inventario (el m1_resultado.json tiene samples reales)
    random.seed(a.seed)
    try:
        d=json.load(open("m1_resultado.json"))
        samples=[c["pmcid"] for c in d["cal"]][:a.n]
    except Exception:
        samples=[f"PMC{random.randint(10000000,12800000)}" for _ in range(a.n)]
        print("(usando PMCIDs aleatorios del rango reciente)")
    print(f"Demo: {len(samples)} PMCIDs")

    rows=[]
    t0=time.time()
    for pmc in samples:
        # metadata para year/license/text_url
        meta=get_metadata_item(pmc)
        if not meta: continue
        text=get_text_url(meta["text_url"])
        if not text: continue
        rows.append({"pmcid":pmc,"year":meta["year"],"license":meta["license"],"text":text[:5000]})
        if len(rows)%50==0: print(f"  +{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    # escribir parquet
    import pyarrow as pa, pyarrow.parquet as pq
    tbl=pa.Table.from_pylist(rows)
    pq.write_table(tbl, a.out)
    print(f"Parquet escrito: {a.out} | {len(rows)} filas | {time.time()-t0:.0f}s")

    if a.duckdb:
        import duckdb
        con=duckdb.connect()
        n=con.execute(f"SELECT count(*) FROM read_parquet('{a.out}')").fetchone()[0]
        lic=con.execute("SELECT license,count(*) FROM read_parquet('{0}') GROUP BY 1 ORDER BY 2 DESC".format(a.out)).fetchall()
        print(f"DuckDB COUNT(*): {n}")
        print("por licencia:", lic)

if __name__=="__main__":
    main()