#!/usr/bin/env python3
"""build_logicdiff_dataset.py — Crea dataset de roles logic-diff para entrenar la cabeza.

Roles:
  0 FILLER
  1 PREMISE  (etiquetas OBSERVACION / HIPOTESIS)
  2 CONNECTIVE (palabras de conexion logica/causal)
  3 DERIVED  (PREDICCION / EVIDENCIA)
  4 CONCLUSION

Uso:
  /bb/bwvenv/bin/python scripts/build_logicdiff_dataset.py \
      --in data/skeleton/train_skeleton_train.jsonl \
      --out data/logicdiff/role_train.pt \
      --tokenizer /beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
      --max-len 768

Requiere offset_mapping del tokenizador (transformers >= 4.36).
"""
import argparse, faulthandler, json, re, sys, time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROLE2ID = {
    "FILLER": 0,
    "PREMISE": 1,
    "CONNECTIVE": 2,
    "DERIVED": 3,
    "CONCLUSION": 4,
}

STAGE_LABELS = {
    "OBSERVACION": "PREMISE",
    "HIPOTESIS": "PREMISE",
    "PREDICCION": "DERIVED",
    "EVIDENCIA": "DERIVED",
    "CONCLUSION": "CONCLUSION",
}

CONNECTIVES = {
    "because", "despite", "therefore", "thus", "however", "since", "while",
    "whereas", "yet", "but", "so", "if", "although", "consequently",
    "furthermore", "moreover", "nevertheless", "otherwise", "hence",
    "accordingly", "due", "unless", "besides", "instead", "meanwhile",
    "afterwards", "otherwise", "likewise", "similarly", "conversely",
}

STAGE_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")


def parse_stage_spans(text):
    """Devuelve lista de (start, end, stage_label) para el texto crudo."""
    ms = list(STAGE_RE.finditer(text))
    spans = []
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        spans.append((start, end, STAGE_LABELS[m.group(1)]))
    return spans


def label_from_offsets(text, ids, offsets):
    """Asigna un rol a cada token dados ids+offsets ya tokenizados."""
    spans = parse_stage_spans(text)
    roles = ["FILLER"] * len(ids)

    # 1) etiquetar por etapa
    for tstart, tend, stage_role in spans:
        for i, (a, b) in enumerate(offsets):
            if a >= tstart and b <= tend and (b - a) > 0:
                roles[i] = stage_role

    # 2) sobrescribir conectivas (pueden estar dentro de etapas, prioridad mayor)
    text_lower = text.lower()
    for i, (a, b) in enumerate(offsets):
        span_text = text_lower[a:b]
        # normalizar ligeramente
        span_text = re.sub(r"[^a-z0-9]", "", span_text)
        if span_text in CONNECTIVES:
            roles[i] = "CONNECTIVE"

    return [ROLE2ID[r] for r in roles]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="jsonl de esqueletos")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    faulthandler.dump_traceback_later(120, repeat=True)
    print("[logicdiff] parse args done", flush=True)
    print("[logicdiff] loading tokenizer", a.tokenizer, flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    print(f"[logicdiff] tokenizer loaded (fast={getattr(tok, 'is_fast', '?')})",
          flush=True)

    # 1) recolectar textos
    texts = []
    with open(a.inp) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text", "")
            if text:
                texts.append(text)
    print(f"[logicdiff] {len(texts)} textos recolectados", flush=True)

    # 2) batch-encoding (fast tokenizer en Rust) por chunks
    all_ids = []
    all_roles = []
    stats = Counter()
    n = 0
    skipped = 0
    B = 512
    t0 = time.time()
    for c0 in range(0, len(texts), B):
        chunk = texts[c0:c0 + B]
        enc = tok(chunk, return_offsets_mapping=True, max_length=a.max_len,
                  truncation=True, add_special_tokens=False)
        for text, ids, offsets in zip(chunk, enc["input_ids"],
                                      enc["offset_mapping"]):
            if len(ids) < 8:
                skipped += 1
                continue
            try:
                roles = label_from_offsets(text, ids, offsets)
            except Exception as e:
                print(f"[warn] skip doc {c0}: {e}", file=sys.stderr)
                skipped += 1
                continue
            all_ids.append(ids)
            all_roles.append(roles)
            stats.update(roles)
            n += 1
        dt = time.time() - t0
        try:
            with open("/proc/self/status") as _f:
                rss = next(l.split()[1] for l in _f if l.startswith("VmRSS"))
        except Exception:
            rss = "?"
        print(f"[logicdiff] {c0 + len(chunk)}/{len(texts)} docs "
              f"({(c0 + len(chunk)) / max(dt, 1e-9):.0f} docs/s) "
              f"RSS={rss}kB", flush=True)

    if not all_ids:
        sys.exit("[fatal] sin datos")

    # padding a max_len del batch
    T = max(len(x) for x in all_ids)
    ids_t = torch.full((len(all_ids), T), 0, dtype=torch.long)
    roles_t = torch.full((len(all_roles), T), -100, dtype=torch.long)
    lens = torch.zeros(len(all_ids), dtype=torch.long)
    for i, (ids, roles) in enumerate(zip(all_ids, all_roles)):
        ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        roles_t[i, :len(roles)] = torch.tensor(roles, dtype=torch.long)
        lens[i] = len(ids)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids_t, "roles": roles_t, "lengths": lens, "stats": dict(stats)}, a.out)
    print(f"docs: {n}  secuencias: {len(all_ids)}  max_len: {T}")
    print("distribucion roles:", {k: v for k, v in sorted(stats.items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
