#!/usr/bin/env python3
"""curate_gold_toolcalls.py — filtra el gold a tool-calls PURAS de las 3 tools.

CURA MANUAL auditable (2026-09-09): de los 60 prompts del corpus, solo estos
14 son llamadas directas de gbif_occurrence (presencia/distribución/taxonomía/
ocurrencias GBIF) o bioclim_download (capas climáticas). Los otros 46 piden
análisis (RF/filogenia/diversidad funcional/correlación), infraestructura
(slurm/watchdog/github/ollama), literatura (PubMed/artículos) u otras APIs
(IUCN/fitogeografía) — su gold 'gbif_occurrence' es ruido del teacher original.

Inclusión por fragmento estable del prompt (no índice, robusto a orden):
"""
import argparse, json, sys
from pathlib import Path

# fragmentos (en minúsculas) que identifican los 14 prompts puros
PURE_FRAGS = [
    "busca registros de presencia de la especie panthera onca",          # 1-3
    "descarga datos climaticos de la base era5",                          # 4-5 (bioclim)
    "descarga informacion sobre la distribucion de la mariposa monarca",  # 11-13 (GBIF)
    "busca en gbif la taxonomia resuelta de la especie",                  # 18-19, 51
    "consulta el exactor de ocurrencias gbif",                            # 29-31
]

def is_pure(prompt):
    p = (prompt or "").lower()
    return any(frag in p for frag in PURE_FRAGS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    items = json.load(open(a.inp))
    pure, rejected = [], []
    for it in items:
        p = it.get("prompt") or ""
        if is_pure(p):
            pure.append(it)
        else:
            rejected.append({"prompt": p[:200], "gold": it["gold"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pure, open(a.out, "w"), ensure_ascii=False, indent=0)
    json.dump({"total": len(items), "pure": len(pure), "rejected": len(rejected),
               "rejected_samples": rejected}, open(a.report, "w"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter(it["gold"][0]["tool"] for it in pure)
    print(f"total={len(items)} | tool-calls PURAS={len(pure)} | descartados={len(rejected)}")
    print("tools en subset puro:", dict(c))
    return 0

if __name__ == "__main__":
    sys.exit(main())