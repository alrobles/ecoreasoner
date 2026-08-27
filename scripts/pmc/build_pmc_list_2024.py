#!/usr/bin/env python3
"""
build_pmc_list_2024.py — Construye la lista de PMCIDs ≥2024 del OA subset.

Usa el S3 Inventory (metadata CSV) y filtra por año (del citation del metadata JSON)
para dar la LISTA de PMCIDs del subrango 2024-2026 que alimentará el Slurm array.

El inventory CSV es enorme (3M+ líneas). Aquí NO se descarga todo el texto: solo el
CSV de nombres (80MB) + metadata JSON de una muestra para decidir año. Como el filtro
por año exacto requeriría bajar el .json de cada PMCID (costoso), usamos la CALIBRACIÓN
PMCID→año aprendida en M1 para estimar el corte numérico de PMCIDs ≥2024. Luego, para
el array, OPTIMIZAMOS: no hace falta metadata de todos — cada shard baja directo el texto
del PMCID de su slice y conserva si el año del metadata >=2024 (decisión en el worker).

Simplificación práctica: este script produce el rango PMCID estimado (num) para 2024-2026,
materializado como archivo de rangos PMCID (MIN, MAX) por shard.
"""
import argparse, json, re, csv, sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--anio-min",type=int,default=2024)
    ap.add_argument("--n-shards",type=int,default=20)
    ap.add_argument("--out",default="pmcid_2024_ranges.csv")
    a=ap.parse_args()
    # estimar num PMC para 2024->? usando calibración M1 (pmcnum vs año)
    # De m1_resultado tenemos cal; tomar regresión simple num~año
    cal=json.load(open("m1_resultado.json"))["cal"]
    pts=[(c["num"],c["año"]) for c in cal if c["año"] and c["num"]]
    if not pts:
        print("sin datos calib", file=sys.stderr); return
    # regresión lineal num = m*año + b
    n=len(pts); Sx=sum(a0 for a0,_ in pts); Sy=sum(b0 for _,b0 in pts)
    Sxy=sum(a0*b0 for a0,b0 in pts); Sxx=sum(a0*a0 for a0,_ in pts)
    m=(n*Sxy-Sx*Sy)/(n*Sxx-Sx*Sx); b=(Sy-m*Sx)/n
    num_corte=m*a.anio_min+b
    print(f"regresión num ~ {m:.2f}*year + {b:.0f} | PMC num para año>={a.anio_min}: {num_corte:.0f}")
    # PM acts known: recent PMCIDs ~ 10000000+. corte en num
    import csv
    with open(a.out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["shard","num_min","num_max"])
        num_max=13000000  # cota superior aproximada del rango (admitir 13000000)
        total=num_max-num_corte
        step=total//a.n_shards
        for s in range(a.n_shards):
            lo=int(num_corte+s*step); hi=int(num_corte+(s+1)*step)
            w.writerow([s,lo,hi])
    print(f"escrito {a.out} con {a.n_shards} shards en rango de num PMC {num_corte:.0f}-{num_max}")

if __name__=="__main__":
    main()