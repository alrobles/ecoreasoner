#!/usr/bin/env python3
"""prep_distill_train.py — convierte distill_data.jsonl (trayectorias con tool-calls)
en un corpus de texto plano con formato estructurado, compatible con
train_mdlm_moe.py (--data).

DATASET (ver docs/dataset-registry.md): lee 3_distill_* (distill_v4_round*.jsonl /
distill_data.jsonl) y escribe 3_distill_train (distill_train.jsonl), input directo
de Fase B/C del entrenamiento.

Cada trayectoria se serializa como texto con delimitadores que preservan la
estructura agéntica (user -> assistant reasoning/tool_call -> tool_result -> final),
para que el dLLM masked-diffusion aprenda el formato de tool-use y razonamiento.

Uso: python3 prep_distill_train.py --input distill_data.jsonl --out distill_train.jsonl
"""
import argparse, json, time

DELIM_USER = "[USER]"
DELIM_ASST = "[ASST]"
DELIM_TOOL = "[TOOL_RESULT]"
DELIM_TOCALL = "[TOOL_CALL]"

def format_tool_call(tc):
    fn = tc.get("function", {})
    return f"{DELIM_TOCALL} {fn.get('name','tool')}({fn.get('arguments','{}')})"

def serialize(rec):
    parts = []
    for s in rec.get("trajectory", []):
        role = s.get("role")
        content = (s.get("content") or "").strip()
        tcs = s.get("tool_calls") or []
        if role == "user":
            if content:
                parts.append(f"{DELIM_USER} {content}")
        elif role == "assistant":
            tok = [format_tool_call(t) for t in tcs]
            if tok:
                txt = (content + " " if content else "") + " ".join(tok)
                parts.append(f"{DELIM_ASST} {txt}")
            elif content:
                parts.append(f"{DELIM_ASST} {content}")
        elif role == "tool":
            parts.append(f"{DELIM_TOOL} {content}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/beegfs/a474r867/ecoreasoner/data/distill_data.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/distill_train.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    n = 0; written = 0
    with open(args.input) as fin, open(args.out, "w") as f:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            if not rec.get("final"):
                continue
            txt = serialize(rec)
            if len(txt) < 20:
                continue
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            written += 1
    el = time.time() - t0
    print(f"DONE {n} trayectorias -> {written} textos entrenables in {el:.1f}s -> {args.out}", flush=True)

if __name__ == "__main__":
    main()