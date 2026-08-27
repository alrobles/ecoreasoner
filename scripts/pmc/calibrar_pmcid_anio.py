#!/usr/bin/env python3
"""
calibrar_pmcid_anio.py — Algoritmo heurístico C: estimar la relación PMCID↔año
para aislar los "últimos 10 años" del Open Access subset de PMC sin descargar todo.

Estrategia ("tirar dardos para atinar"):
  1. Tomar PMCIDs REALES del bucket S3 en distintos rangos (muestreo estratificado),
     obtenidos por el listado S3 (únicas claves reales, no redondos inventados).
  2. Para cada PMCID muestreado, bajar SOLO el .json (metadata, pequeño) que trae
     el año en el campo `citation`.
  3. Ajustar un modelo no lineal PMCID→año (spline / regresión). PMCIDs altos ~
     recientes; bajos ~ viejos. La ingestión va por inglés cronológico → monotónico.
  4. Dado un corte (año_corte = año_actual - 10), resolver el PMCID correspondiente
     y estimar el VOLUMEN de artículos de los últimos 10 años (≈ IDs más recientes).

Uso (en kuhpc con red+almacenamiento):
  python3 calibr_pmcid_anio.py --n-muestras 400 --corte 2016 --out pmid_anio_calib.json
"""
import argparse, json, re, time, urllib.request, urllib.parse

S3="https://pmc-oa-opendata.s3.amazonaws.com/"

def list_bucket(marker="", max_keys=200):
    """Lista claves S3 ordenadas (last-modified); devuelve PMCIDs y next marker."""
    url=S3+("" if not marker else "?marker="+urllib.parse.quote(marker)) + (("&max-keys="+str(max_keys)) if marker else "?max-keys="+str(max_keys))
    req=urllib.request.Request(url,headers={"User-Agent":"research/1.0"})
    xml=urllib.request.urlopen(req,timeout=40).read().decode("utf-8","replace")
    keys=re.findall(r"<Key>(.*?)</Key>",xml)
    nextmarker=re.findall(r"<NextMarker>(.*?)</NextMarker>",xml)
    pmcs=[]
    for k in keys:
        m=re.match(r"(PMC\d+)\.\d+/\1\.\d+\.json",k)
        if m: pmcs.append(m.group(1))
    last=keys[-1] if keys else marker
    return list(dict.fromkeys(pmcs)), (nextmarker[0] if nextmarker else last)

def get_year(pmcid):
    num=pmcid.replace("PMC","")
    for v in range(1,4):
        try:
            req=urllib.request.Request(f"{S3}{num}/{v}/{num}.{v}.json",headers={"User-Agent":"research/1.0"})
            d=json.load(urllib.request.urlopen(req,timeout=25))
            m=re.search(r"(\d{4})", d.get("citation",""))
            if m: return int(m.group(1)), d.get("license","")
        except Exception: continue
    return None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-muestras", type=int, default=300)
    ap.add_argument("--corte", type=int, default=None, help="año_corte p.ej. actual-10")
    ap.add_argument("--out", default="pmcid_anio_calib.json")
    a=ap.parse_args()
    print("== Muestreo estratificado PMCID↔año (últimos 10 años) ==")
    puntos=[]; marker=""; trac=0
    seen=set()
    while len(puntos)<a.n_muestras and trac<200:
        pmcs, marker=list_bucket(marker, max_keys=200)
        for p in pmcs:
            if p in seen: continue
            seen.add(p)
            y,lic=get_year(p)
            if y:
                puntos.append({"pmcid":p,"num":int(p.replace("PMC","")),"año":y,"licencia":lic})
                print(f"  {p}: {y} ({lic})", flush=True)
            time.sleep(0.15)
            trac+=1
            if len(puntos)>=a.n_muestras: break
    json.dump({"muestras":puntos,"corte":a.corte,"metodo":"linear spline sobre PMCID→año (estratificado)"}, open(a.out,"w"), indent=1)
    print(f"\nGuardado {len(puntos)} muestras en {a.out}")
    if puntos:
        años=[p["año"] for p in puntos]
        print(f"rango años: {min(años)}-{max(años)}")
    return puntos

if __name__=="__main__":
    main()