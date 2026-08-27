#!/usr/bin/env python3
"""
m1_inventory_calibrar.py — M1: S3 Inventory + calibración PMCID↔año + estimación volumen.

Objetivo: medir CUÁNTOS artículos (y GB) hay en los últimos ~10 años del PMC OA subset,
sin descargar los .txt (solo metadata e inventory). Esto da el dato para decidir la ingesta.

Pasos:
  1. Localizar el manifest de inventory más reciente (inventory-reports/.../metadata/).
  2. Leer el manifest.json → ruta del CSV inventory (o .gz).
  3. Descargar el CSV del inventory (lista de objetos metadata/*.json) y parsear los PMCIDs.
  4. Muestrear algunos metadata/*.json para calibrar PMCID→año (citation → year, license_code).
  5. Estimar: cuántos metadata (≈ artículos OA reutilizables) hay ≥ año_corte y su tamaño aprox.

Uso (en kuhpc):
  python3 m1_inventory_calibrar.py --corte 2016 --n-muestras 300 --out m1_resultado.json
"""
import argparse, json, re, time, os, csv, io
import urllib.request, urllib.parse

S3="https://pmc-oa-opendata.s3.amazonaws.com/"
INV_PREFIX="inventory-reports/pmc-oa-opendata/metadata/"

def s3_list(prefix, max_keys=400):
    url=f"{S3}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys={max_keys}"
    xml=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"research/1.0"}),timeout=40).read().decode("utf-8","replace")
    keys=re.findall(r"<Key>(.*?)</Key>",xml)
    return keys

def get_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"research/1.0"})
    return json.load(urllib.request.urlopen(req,timeout=40))

def year_from_citation(cit):
    m=re.search(r"(\d{4})",cit or "")
    return int(m.group(1)) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corte", type=int, default=2016, help="año corte: últimos >=corte")
    ap.add_argument("--n-muestras", type=int, default=300)
    ap.add_argument("--out", default="m1_resultado.json")
    a=ap.parse_args()

    print("== M1: S3 Inventory + calibración PMCIDs↔año ==")
    # 1. manifests
    keys=s3_list(INV_PREFIX, max_keys=500)
    manifests=[k for k in keys if k.endswith("manifest.json")]
    if not manifests:
        print("ERR: no manifests"); return
    # el más reciente por fecha en el nombre
    def fecha(k):
        m=re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z)",k)
        return m.group(1) if m else ""
    latest=sorted(manifests, key=fecha)[-1]
    print("manifest más reciente:", latest)

    # 2. manifest.json -> rutas de los CSV inventory
    man=get_json(S3+latest)
    files=man.get("files",[])
    csvs=[f for f in files if f.get("key","").endswith(".csv.gz") or f.get("key","").endswith(".csv")]
    print(f" inventory files en manifest: {len(csvs)}")
    # 3. del primer CSV, descargar (puede ser gz) y parsear NOMBRES de metadata
    if not csvs:
        print("ERR: sin csv en manifest"); return
    csv_key=csvs[0]["key"]
    print("csv:", csv_key, "size=", csvs[0].get("size","?"))
    csv_url=S3+csv_key
    raw=urllib.request.urlopen(urllib.request.Request(csv_url,headers={"User-Agent":"research/1.0"}),timeout=120).read()
    if csv_key.endswith(".gz"):
        import gzip
        raw=gzip.decompress(raw)
    text=raw.decode("utf-8","replace")
    # el CSV tiene líneas; primera línea cabecera
    lines=text.splitlines()
    print(f" inventory CSV: {len(lines)} líneas (~artículos)")

    # 4. muestrear metadata de distintos PMCIDs del inventory para calibrar año
    #    Muestreo ALEATORIO UNIFORME sobre el CSV completo (no solo primeras líneas)
    #    para que la proporción ≥corte sea representativa: cada línea tiene 1/N de ser elegida.
    import random
    random.seed(42)
    all_lines=text.splitlines()
    total_meta=len(all_lines)-1  # -cabecera
    # elegir n-muestras índices aleatorios de las líneas de datos
    idxs=random.sample(range(1,len(all_lines)), min(a.n_muestras, len(all_lines)-1))
    samples=[]
    for i in idxs:
        line=all_lines[i]
        parts=line.split(",")
        if len(parts)>=2:
            m=re.match(r'metadata/(PMC\d+)\.(\d+)\.json$', parts[1].strip().strip('"'))
            if m: samples.append((m.group(1), m.group(2)))
    print(f" Muestreo aleatorio de {len(samples)} PMCIDs del inventory (total_meta={total_meta})")

    # 5. calibrar con metadata real (hasta n-muestras)
    cal=[]
    for pmc,ver in samples[:a.n_muestras]:
        try:
            d=get_json(f"{S3}metadata/{pmc}.{ver}.json")
            y=year_from_citation(d.get("citation"))
            lic=d.get("license_code")
            if y:
                cal.append({"pmcid":pmc,"num":int(pmc.replace("PMC","")),"año":y,"license":lic,
                            "is_oa":d.get("is_pmc_openaccess"),"is_manu":d.get("is_manuscript")})
            time.sleep(0.05)
        except Exception: continue
    print(f" calibrados: {len(cal)}")

    # 6. estimar: distribución años y tamaño
    n_recientes=sum(1 for c in cal if c["año"]>=a.corte)
    pct_10y=100*n_recientes/max(1,len(cal))
    # tamaño promedio de text: estimar 2 downloads de texto para calibrar tamaño? 
    # Por ahora estimación con tamaño medio documentado (~54k chars/doc de v3)
    est_docs_10y = None
    print(f"\n== ESTIMACIÓN (muestra de {len(cal)}) ==")
    print(f" docs ≥{a.corte}: {n_recientes} ({pct_10y:.1f}% de la muestra)")
    if cal:
        años=sorted(set(c["año"] for c in cal))
        print(f" rango años en muestra: {min(años)}-{max(años)}")
    json.dump({"manifest":latest,"n_parsed_inventory":len(samples),"cal":cal,
               "corte":a.corte,"n_recientes":n_recientes,"pct_10y":pct_10y},
              open(a.out,"w"), indent=1)
    print(f" guardado -> {a.out}")

if __name__=="__main__":
    main()