#!/usr/bin/env python3
"""
gen_shards.py — Genera las listas de PMCIDs por shard para el Slurm array.
Toma el CSV del S3 inventory (metadata) y particiona los PMCIDs en N shards.

Uso (en kuhpc, donde está el CSV o se descarga):
  python3 gen_shards.py --csv <inventory.csv.gz> --shards 20 --out pmc_shards_20/
"""
import argparse, gzip, re, os, sys
from collections import defaultdict

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV del inventory (puede .gz)")
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-min", type=int, default=None, help="filtrar PMCIDs con num >= (proxy año; 2024-2026 ~ 11M+)")
    ap.add_argument("--num-max", type=int, default=None, help="filtrar PMCIDs con num <= (proxy; cota alta)")
    a=ap.parse_args()

    # leer PMCIDs del CSV (formato inventory S3: bucket,key,lastmod,etag); filtrar por num si se pide
    pmcs=set()
    if a.csv.endswith(".gz"):
        f=gzip.open(a.csv, "rt")
    else:
        f=open(a.csv, "r")
    with f:
        for line in f:
            parts=line.split(",")
            if len(parts)<2: continue
            key=parts[1].strip().strip('"')
            m=re.match(r"metadata/(PMC\d+)\.\d+\.json$", key)
            if not m: continue
            num=int(m.group(1)[3:])
            if a.num_min is not None and num<a.num_min: continue
            if a.num_max is not None and num>a.num_max: continue
            pmcs.add(m.group(1))
    pmcs=sorted(pmcs)
    print(f"PMCIDs únicos del inventory: {len(pmcs)}")
    os.makedirs(a.out, exist_ok=True)
    n=len(pmcs)//a.shards
    for i in range(a.shards):
        chunk=pmcs[i*n:(i+1)*n] if i<a.shards-1 else pmcs[i*n:]
        with open(f"{a.out}/shard_{i}.txt","w") as f:
            f.write("\n".join(chunk)+"\n")
    print(f"Escritos {a.shards} shards en {a.out}")

if __name__=="__main__":
    main()