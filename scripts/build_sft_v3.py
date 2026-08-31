#!/usr/bin/env python3
"""
build_sft_v3.py — Dataset SFT V3: generacion de CODIGO Python ejecutable.

Fuentes (TODAS train-safe, sin contaminar los 14 GIS del evalset):
  1. gold_programs SAFE de ScienceAgentBench (87 de 103; excluidos los 16 GIS
     que corresponden a los items del eval_holdout: BurnScar, elk, mountainLion,
     sst, tec, transit, coral, water, deforest, mineral, polynomial...).
     Par: [task_inst + dataset_folder_tree] -> [gold code].
  2. ecocode del corpus L1 (14,178 docs): [INSTRUCCION]+[CONTEXTO] -> codigo.
Salida: data/l1/sft_v3_pairs.jsonl (kind=gold|ecocode)
"""
import csv, json, os, re

BASE = "/beegfs/a474r867/ecoreasoner"
CSV = BASE + "/ecobench/ecoseek-benchmark/ecobench_raw/sciagentbench/ScienceAgentBench.csv"
GOLD = BASE + "/ecobench/ecobench_raw/sciagentbench/benchmark/gold_programs"
L1 = BASE + "/data/l1/train_corpus_l1.jsonl"
DST = BASE + "/data/l1/sft_v3_pairs.jsonl"

# golds GIS contaminados (excluir) — los 16 que corresponden a items del evalset
GIS_TERMS = ["burn", "scar", "elk", "oggm", "tec", "sst", "mountain", "lion",
             "transit", "coral", "sponge", "water", "deforest", "polynomial",
             "mineral", "prospect"]


def is_gis(gold_name):
    g = gold_name.lower()
    return any(t in g for t in GIS_TERMS)


def main():
    # 1) CSV -> mapeo gold -> (task_inst, tree)
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            gp = (r.get("gold_program_name") or "").strip()
            if gp:
                rows[gp] = r
    print(f"CSV rows con gold: {len(rows)}")

    # 2) golds safe
    import os as _os
    golds = sorted(_os.listdir(GOLD))
    golds = [g for g in golds if g.endswith(".py") and not g.startswith(".")]
    safe = [g for g in golds if not is_gis(g)]
    print(f"golds totales {len(golds)} | safe {len(safe)} | GIS excluidos {len(golds)-len(safe)}")

    n_gold = 0
    with open(DST, "w", encoding="utf-8") as out:
        for g in safe:
            r = rows.get(g)
            src = open(os.path.join(GOLD, g), encoding="utf-8", errors="ignore").read()
            # quitar canary/header
            src = re.sub(r"# BENCHMARK DATA SHOULD NEVER APPEAR.*?\n", "", src, flags=re.S)
            src = re.sub(r"# canary GUID.*\n", "", src)
            src = src.strip()
            if not src:
                continue
            tree = (r.get("dataset_folder_tree") or "") if r else ""
            task = (r.get("task_inst") or "") if r else ""
            # prompt: pregunta + tree
            prompt = f"{task}"
            if tree:
                prompt += f"\n\nDataset folder tree (use these EXACT paths, relative to CWD):\n{tree}"
            out.write(json.dumps({"kind": "gold", "prompt": prompt,
                                  "response": src}, ensure_ascii=False) + "\n")
            n_gold += 1

        # 3) ecocode L1 — docs con kind != final (evitar tool-calls? no: los
        # ecocode SON codigo python plano; los gen/repair son JSON tool-calls)
        n_eco = 0
        for ln in open(L1, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("kind") not in ("gen", "repair-M1", "repair-M2", "repair-M3",
                                     "repair-M4", "repair-M5"):
                continue
            text = d.get("text", "")
            idx = text.rfind("[ACCION] ")
            if idx < 0:
                continue
            resp = text[idx + len("[ACCION] "):].strip()
            # solo si es codigo (no JSON tool-call) — ecocode = empieza con #/import/from/library
            if not resp or resp.lstrip().startswith("{"):
                continue
            prompt = text[:idx].rstrip()
            out.write(json.dumps({"kind": "ecocode", "prompt": prompt,
                                  "response": resp}, ensure_ascii=False) + "\n")
            n_eco += 1
        print(f"OK: {n_gold} gold + {n_eco} ecocode = {n_gold+n_eco} pares -> {DST}")


if __name__ == "__main__":
    main()