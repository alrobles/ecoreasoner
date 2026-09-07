#!/usr/bin/env python3
"""build_pairs.py — genera pares (ctx/ok/bad) tokenizados para la eval de
discriminación inferencial (Fase 3, F0). Lee el corpus de esqueletos en el
formato REAL que escribe build_skeleton.py --mode full:

    {"pid": ..., "domain": ..., "text": "[OBSERVACION] <t1>\\n[HIPOTESIS] <t2>\\n...",
     "etapas": 3}   # etapas = nº de etapas (int), el texto ya viene serializado

Para cada doc con >=3 etapas construye:

    ctx = etapas 1..k-1 (texto serializado de esas etapas)
    ok  = la etapa k REAL del mismo doc (la continuación correcta)
    bad = una etapa k de OTRO doc (plausible pero inferencialmente incorrecta)

Salida: jsonl con {ctx, ok, bad} en IDs (tokenizer LLaDA), uno por línea.
suite_smoke.py la consume con --pairs.

Uso:
  python3 harness/build_pairs.py \
      --input data/skeleton/train_skeleton.jsonl \
      --tokenizer <dir-hf> \
      --out runs/pairs.jsonl \
      --n-pairs 256 --seed 7331
"""
import argparse, json, re, random, sys
from pathlib import Path

_LABELS = ["[OBSERVACION]", "[HIPOTESIS]", "[PREDICCION]", "[EVIDENCIA]", "[CONCLUSION]"]
# regex: captura (etiqueta, contenido) en orden; las etiquetas son textuales
_SPLIT = re.compile(
    r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]\s*"
)


def _parse_stages(text: str):
    """Devuelve ['<t1>', '<t2>', ...] en orden canónico extrayendo cada etapa."""
    parts = _SPLIT.split(text or "")
    # parts alterna: ['', 'OBSERVACION', 't1', 'HIPOTESIS', 't2', ...] (o con prefijo)
    stages, cur = [], None
    for piece in parts:
        if piece in ("OBSERVACION", "HIPOTESIS", "PREDICCION", "EVIDENCIA", "CONCLUSION"):
            cur = piece
            stages.append("")          # placeholder para el contenido que sigue
        elif cur is not None and stages:
            stages[-1] = piece.strip()  # contenido de la última etiqueta
            cur = None
    return [s for s in stages if s]     # descartar vacíos


def _ser_ctx(seq, upto):
    """Serializa las primeras `upto` etapas como el corpus (etiquetas + \\n)."""
    return "\n".join(f"{_LABELS[i]} {seq[i]}" for i in range(upto))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="train_skeleton.jsonl")
    ap.add_argument("--tokenizer", required=True, help="dir HF tokenizer LLaDA")
    ap.add_argument("--out", required=True, help="pairs.jsonl (ids)")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--max-ctx", type=int, default=512, help="tokens máx de ctx")
    ap.add_argument("--max-cand", type=int, default=128, help="tokens máx de ok/bad")
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    # el snapshot del tokenizer LLaDA tiene código custom -> trust_remote_code
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True,
                                        trust_remote_code=True)
    rng = random.Random(args.seed)

    # 1) cargar docs con >=3 etapas (parseando el texto serializado)
    docs = []  # (pid, [t1, t2, ...])
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            seq = _parse_stages(rec.get("text", ""))
            if len(seq) >= 3:
                docs.append((rec.get("pid") or rec.get("id", "?"), seq))
    if not docs:
        print(f"[fatal] sin docs con >=3 etapas en {args.input}", file=sys.stderr)
        sys.exit(2)
    print(f"[info] {len(docs)} docs con >=3 etapas", flush=True)

    # 2) construir pares: ctx = etapas 1..k-1, ok = etapa k del mismo doc,
    #    bad = etapa k de un doc distinto (misma posición k -> plausible)
    pairs, tried = [], 0
    max_tries = args.n_pairs * 40
    while len(pairs) < args.n_pairs and tried < max_tries:
        tried += 1
        i = rng.randrange(len(docs))
        pid, seq = docs[i]
        k = rng.randrange(2, len(seq))           # etapa a predecir (2..n-1)
        ctx_text = _ser_ctx(seq, k)              # etapas 1..k (0-based k-1)
        ok_text = f"{_LABELS[k]} {seq[k]}"
        j = rng.randrange(len(docs))
        while j == i and len(docs) > 1:
            j = rng.randrange(len(docs))
        _, seq2 = docs[j]
        k2 = min(k, len(seq2) - 1)
        bad_text = f"{_LABELS[k]} {seq2[k2]}"
        if bad_text == ok_text:
            continue
        ids_ctx = tok(ctx_text, add_special_tokens=False)["input_ids"][: args.max_ctx]
        ids_ok = tok(ok_text, add_special_tokens=False)["input_ids"][: args.max_cand]
        ids_bad = tok(bad_text, add_special_tokens=False)["input_ids"][: args.max_cand]
        if len(ids_ctx) < 2 or not ids_ok or not ids_bad:
            continue
        pairs.append({"ctx": ids_ctx, "ok": ids_ok, "bad": ids_bad})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(json.dumps({"out": args.out, "n_pairs": len(pairs),
                      "docs": len(docs), "seed": args.seed}))
    if len(pairs) < args.n_pairs:
        print(f"[warn] solo {len(pairs)}/{args.n_pairs} pares (corpus corto)",
              file=sys.stderr)


if __name__ == "__main__":
    main()