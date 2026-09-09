#!/usr/bin/env python3
"""build_prosa_v8.py — construye el corpus de prosa cientifica v8.

Pipeline:
  1. Cargar fuentes (v7, arxiv, fulltexts PMC, ecoseek-litdump).
  2. Normalizar a schema canonico.
  3. Dedup exacto (keys + titulo normalizado).
  4. Limpiar boilerplate / disclaimers.
  5. Filtrar por calidad (heuristico; TODO: perplexity filtering con ref model).
  6. EOS packing a ventanas de max_seq tokens.
  7. Tokenizar con LLaDA y guardar .npy.
  8. Emitir v8_report.json.

Uso:
  python3 scripts/build_prosa_v8.py --config scripts/build_prosa_v8.yaml
  python3 scripts/build_prosa_v8.py --selftest
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import yaml
from collections import Counter
from pathlib import Path


def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _load_jsonl(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def _normalize(doc, source, idx):
    """Normaliza un doc de cualquier fuente al schema canonico."""
    text = doc.get("text") or doc.get("abstract") or doc.get("content") or doc.get("body", "")
    title = doc.get("title", "") or ""
    return {
        "doc_id": doc.get("pmcid") or doc.get("pmid") or doc.get("doi") or f"{source}_{idx}",
        "source": source,
        "title": title,
        "text": text,
        "year": doc.get("year"),
        "domain": doc.get("domain") or doc.get("field") or source,
        "pmid": doc.get("pmid"),
        "pmcid": doc.get("pmcid"),
        "doi": doc.get("doi"),
    }


def _clean_text(text: str) -> str:
    """Limpieza ligera: espacios, copyright, disclaimers."""
    text = re.sub(r"\s+", " ", text)
    patterns = [
        r"©\s*\d{4}[^\n]{0,200}",
        r"Published by[^\n]{0,200}",
        r"This is an open access article[^\n]{0,400}",
        r"All rights reserved[^\n]{0,100}",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def _boilerplate_filter(text: str) -> str:
    """Elimina lineas/plantillas metodologicas muy repetidas.

    TODO (futuro): aprender n-gramas frecuentes del corpus y filtrar.
    """
    lines = text.split(". ")
    # Descartar frases de agradecimiento comunes
    filtered = []
    for line in lines:
        low = line.lower()
        if any(x in low for x in ["we thank ", "we acknowledge ", "funding:", "acknowledgements"]):
            continue
        if len(line) > 10:
            filtered.append(line)
    return ". ".join(filtered)


def _quality_heuristic(text: str, cfg: dict) -> bool:
    """Filtro de calidad heurístico."""
    if len(text) < cfg.get("min_chars", 256):
        return False
    words = text.split()
    if len(words) < cfg.get("min_words", 50):
        return False
    # Evitar documentos con demasiadas lineas muy cortas
    lines = text.split("\n")
    short = sum(1 for l in lines if len(l) < 20)
    if short / max(len(lines), 1) > 0.5:
        return False
    # Evitar demasiado repeticion de tokens
    uniq = len(set(words))
    if not words:
        return False
    if uniq / len(words) < 0.2:
        return False
    return True


def _dedup(docs, keys=("pmid", "pmcid", "doi")):
    """Dedup exacto por keys + titulo normalizado."""
    seen = set()
    out = []
    for d in docs:
        sig = tuple(sorted([(k, str(d.get(k, "")).lower().strip()) for k in keys if d.get(k)]))
        title_norm = d.get("title", "").lower().strip()
        key = (sig, _md5_hex(title_norm))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _build_tokenizer(path: str, allow_dummy=False):
    try:
        from transformers import AutoTokenizer
        print(f"[tokenizer] cargando {path}")
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        if allow_dummy:
            print(f"[warn] no pude cargar tokenizer ({e}); usando dummy para selftest")
            class DummyTok:
                vocab_size = 126080
                eos_token = "<|endoftext|>"
                def encode(self, text, add_special_tokens=False):
                    return [abs(hash(w)) % self.vocab_size for w in text.split()]
            return DummyTok()
        raise RuntimeError(f"No se pudo cargar tokenizer {path}: {e}")


def _pack_with_eos(docs, tokenizer, max_seq=768, eos_token=None):
    """Concatena documentos con EOS y corta en ventanas de max_seq ids."""
    eos_token = eos_token or getattr(tokenizer, "eos_token", "<|endoftext|>")
    all_ids = []
    eid = tokenizer.encode(eos_token, add_special_tokens=False)
    if not eid:
        eid = [tokenizer.vocab_size - 1]
    for d in docs:
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.extend(eid)
        # Si el buffer crece mucho, ir cortando ventanas para no saturar memoria
        while len(all_ids) >= max_seq:
            yield all_ids[:max_seq]
            all_ids = all_ids[max_seq:]
    # No yield final corto (para evitar ventanas con mucho EOS)


def _write_jsonl(docs, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "train_corpus_v8.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def _write_npy(windows, outdir):
    import numpy as np
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arr = np.array(windows, dtype=np.uint32)
    path = outdir / "train_ids_v8.npy"
    np.save(path, arr)
    return path


def _write_report(docs, windows, outdir, config):
    outdir = Path(outdir)
    domains = Counter(d.get("domain", "?") for d in docs)
    years = Counter(str(d.get("year", "?")) for d in docs)
    report = {
        "n_docs": len(docs),
        "n_windows": len(windows),
        "n_tokens": sum(len(w) for w in windows),
        "domains": dict(domains),
        "years": dict(years),
        "sources": config.get("sources", []),
        "config": config,
    }
    path = outdir / "v8_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _selftest():
    docs = [
        {"title": "Bees and drought", "text":
         "Bees declined in dry years. Drought reduces floral resources. "
         "Dry plots showed lower visitation. Visitation increased in irrigated plots. "
         "Irrigation buffers drought effects.", "year": 2025, "domain": "eco"},
        {"title": "Frogs and pesticides", "text":
         "Frogs vanished upstream. Pesticide runoff drives declines. "
         "Downstream sites showed lower abundance. Abundance decreased downstream. "
         "Runoff likely contributes.", "year": 2025, "domain": "eco"},
        {"title": "Bees and drought", "text":
         "This is a duplicate abstract with the same title.", "year": 2025, "domain": "eco"},
    ]
    config = {
        "sources": [{"path": "_dummy", "weight": 1.0}],
        "dedup": {"keys": ["pmid", "pmcid", "doi"]},
        "filter": {"min_chars": 100, "min_words": 10},
        "packing": {"eos_token": "<|endoftext|>", "max_seq": 32},
        "tokenizer": "dummy",
        "outdir": tempfile.mkdtemp(prefix="prosa_v8_selftest_"),
    }
    all_docs = [_normalize(d, "selftest", i) for i, d in enumerate(docs)]
    all_docs = _dedup(all_docs, tuple(config["dedup"]["keys"]))
    for d in all_docs:
        d["text"] = _boilerplate_filter(_clean_text(d["text"]))
    all_docs = [d for d in all_docs if _quality_heuristic(d["text"], config["filter"])]
    tok = _build_tokenizer(config["tokenizer"], allow_dummy=True)
    windows = list(_pack_with_eos(all_docs, tok, config["packing"]["max_seq"], config["packing"]["eos_token"]))
    _write_jsonl(all_docs, config["outdir"])
    _write_npy(windows, config["outdir"])
    report = _write_report(all_docs, windows, config["outdir"], config)
    print("selftest OK:", json.dumps(report, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/build_prosa_v8.yaml",
                    help="YAML con sources, dedup, filter, packing")
    ap.add_argument("--limit-docs", type=int, default=0,
                    help="limitar a N docs por fuente para test rapido")
    ap.add_argument("--dry", action="store_true",
                    help="no tokenizar, solo armar report y corpus plano")
    ap.add_argument("--selftest", action="store_true",
                    help="corre pipeline con datos dummy")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    with open(a.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Cargar y normalizar
    all_docs = []
    for src in config["sources"]:
        p = src["path"]
        if not os.path.exists(p):
            print(f"[warn] fuente no existe: {p}")
            continue
        name = Path(p).stem
        for idx, doc in _load_jsonl(p, limit=a.limit_docs or None):
            d = _normalize(doc, name, idx)
            d["text"] = _clean_text(d["text"])
            all_docs.append(d)

    print(f"[load] {len(all_docs)} docs crudos")

    # Dedup
    all_docs = _dedup(all_docs, tuple(config["dedup"]["keys"]))
    print(f"[dedup] {len(all_docs)} docs unicos")

    # Boilerplate y filtro de calidad
    for d in all_docs:
        d["text"] = _boilerplate_filter(d["text"])
    all_docs = [d for d in all_docs if _quality_heuristic(d["text"], config["filter"])]
    print(f"[filter] {len(all_docs)} docs")

    # Guardar corpus plano
    _write_jsonl(all_docs, outdir)

    if a.dry:
        print("[dry] sin tokenizar")
        report = _write_report(all_docs, [], outdir, config)
        print(json.dumps(report, indent=2))
        print(f"[done] en {outdir}")
        return 0

    # Tokenizar y packing
    tok = _build_tokenizer(config["tokenizer"])
    max_seq = config["packing"]["max_seq"]
    eos_token = config["packing"]["eos_token"]
    windows = list(_pack_with_eos(all_docs, tok, max_seq, eos_token))
    print(f"[pack] {len(windows)} ventanas, {sum(len(w) for w in windows)} tokens")

    _write_npy(windows, outdir)
    report = _write_report(all_docs, windows, outdir, config)
    print(json.dumps(report, indent=2))
    print(f"[done] en {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
