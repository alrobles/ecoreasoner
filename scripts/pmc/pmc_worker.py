#!/usr/bin/env python3
"""
pmc_worker.py — Worker de un shard del Slurm array. Baja PMCIDs (rango num), filtra
por año>=anio_min y licencia reusable, descarga texto, escribe Parquet por shard.

Uso (desde el slurm):
  python3 pmc_worker.py --num-min 10000000 --num-max 10100000 --anio-min 2024 \
      --out shard_0.parquet

Estrategia de filtro exacto por año: por cada PMCID candidato, bajar metadata JSON
(año real desde citation) y solo si >=anio_min (y licencia usable) bajar el texto.
"""
import argparse, json, re, time, os
import urllib.request

S3="https://pmc-oa-opendata.s3.amazonaws.com/"
UA={"User-Agent":"research/1.0"}
LIC_OK={"CC BY","CC BY-NC","CC BY-NC-ND","CC BY-NC-SA","CC0","TDM"}

def get_meta(num, ver=1):
    url=f"{S3}metadata/PMC{num}.{ver}.json"
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=25))
        m=re.search(r"(\d{4})",d.get("citation",""))
        year=int(m.group(1)) if m else None
        lic=d.get("license_code")
        text_url=d.get("text_url","")
        https=text_url.replace("s3://pmc-oa-opendata","https://pmc-oa-opendata.s3.amazonaws.com").split("?")[0] if text_url else ""
        return {"pmcid":f"PMC{num}","year":year,"license":lic,"text_url":https}
    except Exception:
        return None

def get_text(url):
    try:
        return urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60).read().decode("utf-8","replace")
    except Exception:
        return ""

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--num-min",type=int)
    ap.add_argument("--num-max",type=int)
    ap.add_argument("--list-file", help="archivo con PMCIDs (uno por línea) a procesar (preferido)")
    ap.add_argument("--anio-min",type=int,default=2024)
    ap.add_argument("--out",required=True)
    ap.add_argument("--step",type=int,default=1,help="cada cuántos nums probar (default 1 = todos)")
    a=ap.parse_args()

    # modos: lista de PMCIDs exacta (inventory) o rango numérico
    if a.list_file and os.path.exists(a.list_file):
        nums=[int(line.strip().replace("PMC","")) for line in open(a.list_file) if line.strip().startswith("PMC")]
        cands=nums
    elif a.num_min is not None and a.num_max is not None:
        cands=range(a.num_min, a.num_max, a.step)
    else:
        print("error: --list-file o --num-min/--num-max requerido"); return

    rows=[]; t0=time.time()
    for num in cands:
        meta=get_meta(num)
        if not meta or not meta["year"]: continue
        if meta["year"]<a.anio_min: continue
        if meta["license"] not in LIC_OK: continue
        text=get_text(meta["text_url"])
        if len(text)<500: continue
        rows.append({"pmcid":meta["pmcid"],"year":meta["year"],"license":meta["license"],"text":text[:8000]})
        if len(rows)%20==0: print(f"  shard {a.out}: +{len(rows)} ({(time.time()-t0):.0f}s)", flush=True)
    # escribir parquet
    import pyarrow as pa, pyarrow.parquet as pq
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), a.out)
        print(f"  shard {a.out}: {len(rows)} rows OK")
    else:
        print(f"  shard {a.out}: 0 rows (sin artículos válidos)")
if __name__=="__main__":
    main()