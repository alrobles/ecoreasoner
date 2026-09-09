#!/usr/bin/env python3
"""build_prosa_v8.py — construye el corpus de prosa cientifica v8.

Pipeline:
  1. Cargar fuentes (v7, arxiv, fulltexts PMC, ecoseek-litdump).
  2. Normalizar a schema canonico.
  3. Dedup exacto + fuzzy.
  4. Limpiar boilerplate / disclaimers.
  5. Filtrar por calidad (heuristico; TODO: perplexity filtering con ref model).
  6. EOS packing a ventanas de max_seq tokens.
  7. Tokenizar con LLaDA y guardar .npy.
  8. Emitir v8_report.json.

Uso:
  python3 scripts/build_prosa_v8.py --config scripts/build_prosa_v8.yaml
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# TODO (Hermes): importar tokenizador cuando el env este disponible
# from transformers import AutoTokenizer

DEFAULT_LABELS = ["[OBSERVACION]", "[HIPOTESIS]", "[PREDICCION]", "[EVIDENCIA]", "[CONCLUSION]"]


def _load_jsonl(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def _normalize(doc, source, idx):
    """Normaliza un doc de cualquier fuente al schema canonico."""
    text = doc.get("text") or doc.get("abstract") or doc.get("content") or ""
    return {
        "doc_id": doc.get("pmcid") or doc.get("pmid") or doc.get("doi") or f"{source}_{idx}",
        "source": source,
        "title": doc.get("title", ""),
        "text": text,
        "year": doc.get("year"),
        "domain": doc.get("domain") or doc.get("field") or source,
        "pmid": doc.get("pmid"),
        "pmcid": doc.get("pmcid"),
        "doi": doc.get("doi"),
    }


def _dedup_exact(docs, keys=("pmid", "pmcid", "doi")):
    seen = set()
    out = []
    for d in docs:
        sig = tuple(d.get(k) for k in keys)
        if any(sig) and sig in seen:
            continue
        if any(sig):
            seen.add(sig)
        out.append(d)
    return out


def _dedup_fuzzy(docs, threshold=0.95):
    """TODO: dedup por similitud de titulo con MinHash/LSH o embedding ligero."""
    # stub: conserva todos
    return docs


def _clean_boilerplate(text):
    """Quita boilerplate y plantillas metodologicas comunes."""
    # TODO (Hermes): aprender plantillas frecuentes del corpus y removerlas.
    # Patrones preliminares:
    patterns = [
        r"©\s*\d{4}.*?(?=[A-Z]|$)",
        r"Published by.*?(?=[A-Z]|$)",
        r"This is an open access article.*?license",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def _quality_filter(docs, mode="heuristic"):
    """Filtra por calidad. TODO: anadir perplexity filtering con referencia."""
    out = []
    for d in docs:
        t = d["text"]
        if len(t) < 256:
            continue
        words = len(t.split())
        if words < 50:
            continue
        # simple heuristic: evitar documentos con >30% de lineas cortas
        lines = t.split("\n")
        short = sum(1 for l in lines if len(l) < 20)
        if short / max(len(lines), 1) > 0.5:
            continue
        out.append(d)
    return out


def _pack_with_eos(docs, tokenizer, max_seq=768, eos_token="<|endoftext|>"):
    """Concatena documentos con EOS y corta en ventanas de max_seq ids."""
    # TODO (Hermes): asegurar que cada ventana nueva empiece con EOS previo.
    all_ids = []
    for d in docs:
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        eid = tokenizer.encode(eos_token, add_special_tokens=False)
        all_ids.extend(ids + eid)
    windows = []
    for i in range(0, len(all_ids), max_seq):
        w = all_ids[i:i + max_seq]
        if len(w) == max_seq:
            windows.append(w)
    return windows


def _write_npy(windows, outdir):
    import numpy as np
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arr = np.array(windows, dtype=np.uint32)
    path = outdir / "train_ids_v8.npy"
    np.save(path, arr)
    return path


def _write_jsonl(docs, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "train_corpus_v8.jsonl"
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


def _write_report(docs, windows, outdir, config):
    outdir = Path(outdir)
    domains = {}
    years = {}
    for d in docs:
        domains[d.get("domain", "?")] = domains.get(d.get("domain", "?"), 0) + 1
        years[str(d.get("year", "?"))] = years.get(str(d.get("year", "?")), 0) + 1
    report = {
        "n_docs": len(docs),
        "n_windows": len(windows),
        "n_tokens": sum(len(w) for w in windows),
        "domains": domains,
        "years": years,
        "sources": config.get("sources", []),
    }
    path = outdir / "v8_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/build_prosa_v8.yaml",
                    help="YAML con sources, dedup, filter, packing")
    ap.add_argument("--limit-docs", type=int, default=0,
                    help="limitar a N docs por fuente para test rapido")
    ap.add_argument("--dry", action="store_true",
                    help="no tokenizar, solo armar report y corpus plano")
    args = ap.parse_args()

    # TODO (Hermes): cargar YAML real con safe_load
    # import yaml
    # config = yaml.safe_load(open(args.config))
    # Stub: config por defecto mientras no exista el YAML
    config = {
        "sources": [
            {"path": "data/train_corpus_v7_clean.jsonl", "weight": 0.5},
            {"path": "data/arxiv/fulltext/fulltext_corpus.jsonl", "weight": 0.25},
        ],
        "dedup": {"keys": ["pmid", "pmcid", "doi"]},
        "filter": {"min_chars": 256, "mode": "heuristic"},
        "packing": {"eos_token": "<|endoftext|>", "max_seq": 768},
        "tokenizer": "/beegfs/a474r867/ecoreasoner/tokenizer/LLaDA",
        "outdir": "data/prosa_v8",
    }

    # Cargar y normalizar
    all_docs = []
    for src in config["sources"]:
        p = src["path"]
        name = Path(p).stem
        for idx, doc in _load_jsonl(p, limit=args.limit_docs or None):
            all_docs.append(_normalize(doc, name, idx))

    print(f"[load] {len(all_docs)} docs crudos")

    # Dedup
    all_docs = _dedup_exact(all_docs, tuple(config["dedup"]["keys"]))
    all_docs = _dedup_fuzzy(all_docs)
    print(f"[dedup] {len(all_docs)} docs")

    # Limpiar y filtrar
    for d in all_docs:
        d["text"] = _clean_boilerplate(d["text"])
    all_docs = _quality_filter(all_docs, mode=config["filter"].get("mode", "heuristic"))
    print(f"[filter] {len(all_docs)} docs")

    # Guardar corpus plano
    _write_jsonl(all_docs, config["outdir"])

    if args.dry:
        print("[dry] sin tokenizar")
        return 0

    # Tokenizar y packing
    # TODO (Hermes): usar AutoTokenizer con trust_remote_code=True
    # tok = AutoTokenizer.from_pretrained(config["tokenizer"], trust_remote_code=True)
    # stub: tokenizador dummy para que el script corra en CPU sin HF
    class DummyTok:
        vocab_size = 126080
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % self.vocab_size for w in text.split()]
    tok = DummyTok()

    windows = _pack_with_eos(all_docs, tok,
                            max_seq=config["packing"]["max_seq"],
                            eos_token=config["packing"]["eos_token"])
    print(f"[pack] {len(windows)} ventanas, {sum(len(w) for w in windows)} tokens")

    _write_npy(windows, config["outdir"])
    report = _write_report(all_docs, windows, config["outdir"], config)
    print(json.dumps(report, indent=2))
    print(f"[done] en {config['outdir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
