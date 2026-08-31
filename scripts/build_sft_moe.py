#!/usr/bin/env python3
"""
build_sft_moe.py — Convierte train_corpus_l1.jsonl (15,775 docs) en pares
(prompt, response) para SFT del LLaDA-MoE-7B-A1B-Instruct (receta paper LLaDA:
p0 no se enmascara, r0 se enmascara, loss solo en mascaras).

Formato de salida (jsonl): {"prompt": "...", "response": "..."}
  - gen:      prompt = [INSTRUCCION]+[CONTEXTO]  ; response = [ACCION] {json}
  - repair-M*:prompt = [INSTRUCCION]+[CONTEXTO]+[ERROR] ; response = [ACCION] {json reparado}
  - final:    prompt = hasta [RESPUESTA] ; response = tras [RESPUESTA]
Salida: data/l1/sft_moe_pairs.jsonl (+ conteo por kind)
"""
import json, sys

SRC = "/beegfs/a474r867/ecoreasoner/data/l1/train_corpus_l1.jsonl"
DST = "/beegfs/a474r867/ecoreasoner/data/l1/sft_moe_pairs_v2.jsonl"

def split_gen(text):
    """prompt hasta [ACCION] (sin incluir), response = [ACCION] + json."""
    idx = text.rfind("[ACCION]")
    if idx < 0:
        return None, None
    return text[:idx].rstrip(), text[idx:].strip()

def split_final(text):
    idx = text.rfind("[RESPUESTA]")
    if idx < 0:
        return None, None
    return text[:idx + len("[RESPUESTA]")].rstrip(), text[idx + len("[RESPUESTA]"):].strip()

def main():
    counts = {}
    n_ok = n_skip = 0
    with open(SRC, encoding="utf-8") as f, open(DST, "w", encoding="utf-8") as out:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            kind = d.get("kind", "")
            text = d.get("text", "")
            # SFT V2 (2026-08-30): SOLO kinds JSON (gen + repairs), excluir "final"
            # (codigo R/python) — el "final" contamina hacia modo codigo.
            if kind in ("gen", "repair-M1", "repair-M2", "repair-M3", "repair-M4", "repair-M5"):
                p, r = split_gen(text)
            else:
                continue
            if not p or not r:
                n_skip += 1
                continue
            counts[kind] = counts.get(kind, 0) + 1
            out.write(json.dumps({"kind": kind, "prompt": p, "response": r},
                                 ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"OK {n_ok} pares | skip {n_skip} | por kind: {counts}")
    print(f"-> {DST}")

if __name__ == "__main__":
    main()