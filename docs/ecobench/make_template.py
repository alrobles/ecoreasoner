#!/usr/bin/env python3
"""Genera la plantilla eco_bench_new_items.csv / .json para que el usuario rellene más preguntas.

Cada fila = una pregunta del estilo ScienceAgentBench. Rellenar y luego:
  python3 build_ecobench_eval.py --add-csv eco_new_items.csv
(o ingerir manualmente al ecobench_eval.json).
"""
import csv, json, os

ROOT=os.path.dirname(os.path.abspath(__file__))
COLUMNS=["id","family","question","lang","code_hint","data_path","expected_type","expected_value","expected_tolerance","output_artifact","split","license","notes"]

TEMPLATE=[
  # 3 ejemplos en blanco con el patrón (para coprar)
  {"id":"eco-sdm-002","family":"SDM",
   "question":"Dado que 120 registros de <especie> y 3 bioclim, ajusta MaxEnt con <paquete>, aplica umbral, reporta AUC",
   "lang":"R","code_hint":"<paquete/método, EJ: maxentcpp>","data_path":"<ruta datos>",
   "expected_type":"numeric","expected_value":"<AUC u otra num>","expected_tolerance":"0.01",
   "output_artifact":"pred_results/<archivo>.csv","license":"own","notes":"plantilla SDM"},
  {"id":"eco-phylo-002","family":"phylo",
   "question":"Dada una matriz de distancias genéticas y filogenia de <N> especies, estima lambda y K de Blomberg y di si OU supera a BM",
   "lang":"R","code_hint":"phytools","data_path":"<matriz/filogenia>",
   "expected_type":"numeric","expected_value":"lambda","expected_tolerance":"0.05",
   "output_artifact":"pred_results/phylo.csv","license":"own","notes":"plantilla filo"},
]

def main():
    out_csv=os.path.join(ROOT,"eco_new_items.csv")
    out_json=os.path.join(ROOT,"eco_new_items_template.json")
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader(); w.writerows(TEMPLATE)
    json.dump(TEMPLATE, open(out_json,"w"), indent=1, ensure_ascii=False)
    print("Plantilla creada:", out_csv, "y", out_json)

if __name__=="__main__":
    main()