#!/usr/bin/env python3
"""build_pairs.py — genera pares (ctx/ok/bad) tokenizados para la eval de
discriminación inferencial (Fase 3, F0). Lee el corpus de esqueletos y, para
cada doc con >=3 etapas, construye:

    ctx = etapas 1..k-1 serializadas (e.g. [OBSERVACION]...[HIPOTESIS])
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
import argparse, json, random, sys
from pathlib import Path

_STAGES = ["obs", "hyp", "pred", "evid", "conc"]
_LABEL = {"obs": "[OBSERVACION]", "hyp": "[HIPOTESIS]", "pred": "[PREDICCION]",
          "evid": "[EVIDENCIA]", "conc": "[CONCLUSION]"}


def _serialize(stages: dict, upto: int) -> str:
    """Serializa las primeras `upto` etapas en orden canónico (formato F0b)."""
    parts = []
    for st in _STAGES[:upto]:
        if st in stages and stages[st]:
            parts.append(f"{_LABEL[st]} {stages[st]}")
    return "\n".join(parts)


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
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rng = random.Random(args.seed)

    # 1) cargar docs con >=3 etapas y su secuencia canónica de texto
    docs = []  # (pid, [texto_etapa_1, texto_etapa_2, ...]) en orden canónico
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            st = rec.get("etapas") or {}
            seq = [st[k] for k in _STAGES if st.get(k)]
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
        k = rng.randrange(2, len(seq))          # etapa a predecir (2..n-1)
        ctx_text = _serialize({"obs": seq[0], "hyp": seq[1]}, 2) if k >= 2 else ""
        # contexto más rico: primeras k-1 etapas
        ctx_text = _serialize({_STAGES[j]: seq[j] for j in range(k - 1)}, k - 1)
        ok_text = f"{_LABEL[_STAGES[k]]} {seq[k]}"
        # bad: etapa k de otro doc (misma posición -> misma etiqueta, otro contenido)
        j = rng.randrange(len(docs))
        while j == i and len(docs) > 1:
            j = rng.randrange(len(docs))
        _, seq2 = docs[j]
        k2 = min(k, len(seq2) - 1)
        bad_text = f"{_LABEL[_STAGES[k]]} {seq2[k2]}"
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