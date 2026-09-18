#!/usr/bin/env python3
"""consolidate_papers.py — consolida TODOS los papers cosechados en una DB
canonica con estructura esqueleto+fulltext.

Stage collect  : un pase streaming por las fuentes en orden de prioridad
                 (fulltext primero). Dedupe por clave pmid/pmcid/arxiv o
                 hash de texto. Limpieza ligera inline. Emite
                 papers_db.jsonl + skeleton_db.jsonl (esqueletos ya
                 existentes de corpus_v4_en se reusan por pid).
Stage skeleton : corre build_skeleton.extract sobre papers sin esqueleto
                 (fulltext/abstract largos) y completa skeleton_db.
Stage stats    : reporte de cobertura/dedupe/calidad.

Uso:
  python3 consolidate_papers.py collect  --out data/papersdb/
  python3 consolidate_papers.py skeleton --out data/papersdb/ [--workers N]
  python3 consolidate_papers.py stats    --out data/papersdb/
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

BASE = "/beegfs/a474r867/ecoreasoner"
DATA = f"{BASE}/data"

# (path, kind, id_fields) en orden de prioridad: first-seen gana.
SOURCES = [
    (f"{DATA}/train_corpus_phys.jsonl",        "fulltext", ["arxiv_id", "pmid"]),
    (f"{DATA}/fulltext_corpus_all.jsonl",      "fulltext", ["pmcid", "pmid"]),
    (f"{DATA}/train_corpus_v6.jsonl",          "abstract", ["pmid"]),
    (f"{DATA}/train_corpus_v5.jsonl",          "abstract", ["pmid"]),
    (f"{DATA}/train_corpus_v4.jsonl",          "abstract", ["pmid"]),
    (f"{DATA}/eco_corpus_v2.jsonl",            "abstract", ["pmid"]),
    (f"{DATA}/ecoevorxiv_fulltext_c0.jsonl",   "fulltext", ["pmid", "doi"]),
    (f"{DATA}/ecoevorxiv_fulltext_c1.jsonl",   "fulltext", ["pmid", "doi"]),
    (f"{DATA}/ecoevorxiv_fulltext_c2.jsonl",   "fulltext", ["pmid", "doi"]),
    (f"{DATA}/ecoevorxiv_fulltext_c3.jsonl",   "fulltext", ["pmid", "doi"]),
]
# corpus_v4_en aporta esqueletos ya extraidos (reuso) + docs unam-en/synth.
SKELETON_SRC = f"{DATA}/corpus_v4_en.jsonl"

MIN_CHARS = 200          # descarta docs triviales/vacios
MAX_ALPHA_BAD = 0.35     # si <35% caracteres alfabeticos -> boilerplate/binario
_JUNK_RE = re.compile(r"(\\documentclass|\\usepackage\{|\\begin\{document\})")


def norm_text_key(text):
    t = unicodedata.normalize("NFKD", text[:4000]).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:20]


def doc_key(r):
    for f in ("pmcid", "pmid", "arxiv_id", "doi"):
        v = r.get(f)
        if v:
            v = str(v).strip()
            if v and v.lower() not in ("none", "null", "0"):
                pref = {"pmcid": "pmc", "pmid": "pmid",
                        "arxiv_id": "arxiv", "doi": "doi"}[f]
                return f"{pref}:{v}"
    return f"h:{norm_text_key(r.get('text', ''))}"


def clean_ok(text):
    if not text or len(text) < MIN_CHARS:
        return False
    sample = text[:2000]
    alpha = sum(c.isalpha() for c in sample)
    if alpha / max(len(sample), 1) < MAX_ALPHA_BAD:
        return False
    return True


def detect_lang(text):
    head = text[:1500].lower()
    es = sum(head.count(w) for w in (" el ", " la ", " de ", " que ", " los "))
    en = sum(head.count(w) for w in (" the ", " of ", " and ", " in ", " a "))
    return "es" if es > en * 1.5 else "en"


def stage_collect(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    papers_out = open(f"{out_dir}/papers_db.jsonl", "w")
    skel_out = open(f"{out_dir}/skeleton_db.jsonl", "w")
    seen = set()
    stats = Counter()
    skel_by_key = {}

    # 1) fuentes de texto en orden de prioridad
    for path, kind, _ in SOURCES:
        if not os.path.exists(path):
            stats[f"missing:{os.path.basename(path)}"] += 1
            continue
        for line in open(path, errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            text = r.get("text", "")
            if not clean_ok(text):
                stats["dropped_short_or_junk"] += 1
                continue
            k = doc_key(r)
            if k in seen:
                stats["dup_skipped"] += 1
                continue
            seen.add(k)
            rec = {
                "key": k, "kind": kind, "text": text,
                "domain": r.get("domain"), "year": r.get("year"),
                "source": r.get("source") or os.path.basename(path),
                "pmid": r.get("pmid"), "pmcid": r.get("pmcid"),
                "arxiv_id": r.get("arxiv_id"), "doi": r.get("doi"),
                "license": r.get("license"), "title": r.get("title"),
                "lang": detect_lang(text),
            }
            papers_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[f"kept_{kind}"] += 1
        stats[f"scanned"] += 1
        print(f"[collect] {os.path.basename(path)} -> seen={len(seen)}",
              flush=True)

    # 2) esqueletos existentes de corpus_v4_en (reuso por pid/pid-key)
    #    + docs unam-en/synth como papers propios
    for line in open(SKELETON_SRC, errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = r.get("pid")
        src = r.get("src") or r.get("source", "")
        if pid and str(pid).startswith("PMC"):
            key = f"pmc:{pid}"
        elif pid:
            key = f"pmid:{pid}"
        else:
            key = f"h:{norm_text_key(r.get('text',''))}"
        if src == "skeleton" and r.get("text"):
            if key not in skel_by_key:
                skel_by_key[key] = {
                    "key": key, "skeleton": r["text"],
                    "etapas": r.get("etapas"), "domain": r.get("domain"),
                    "src": "reused_v4en"}
                stats["skel_reused"] += 1
        elif src in ("unam-en", "synth"):
            k2 = f"{src}:{norm_text_key(r.get('text',''))}"
            if k2 not in seen:
                seen.add(k2)
                papers_out.write(json.dumps({
                    "key": k2, "kind": src, "text": r["text"],
                    "domain": r.get("domain"), "year": None,
                    "source": src, "pmid": pid, "pmcid": None,
                    "arxiv_id": None, "doi": None, "license": None,
                    "title": None, "lang": r.get("lang", "en"),
                }, ensure_ascii=False) + "\n")
                stats[f"kept_{src}"] += 1
    papers_out.close()
    for v in skel_by_key.values():
        skel_out.write(json.dumps(v, ensure_ascii=False) + "\n")
    skel_out.close()
    stats["papers_total"] = len(seen)
    stats["skeletons_total"] = len(skel_by_key)
    with open(f"{out_dir}/collect_stats.json", "w") as f:
        json.dump(dict(stats), f, indent=2)
    print("[collect] DONE", dict(stats))


_BS = None


def _skel_work(rec):
    """Worker picklable: mismo orden que build_skeleton.main
    (abstract -> imrad -> phrases -> need_obs)."""
    global _BS
    if _BS is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import build_skeleton as _m
        _BS = _m
    try:
        text = rec["text"]
        stages = {}
        ab = _BS.abstract_structured(text)
        if ab:
            stages.update(ab)
        if len({k for k in _BS._STAGES if k in stages}) < 3:
            im = _BS.imrad_sections(text)
            if im:
                stages.update(im)
        if len({k for k in _BS._STAGES if k in stages}) < 3:
            for k, v in _BS.phrases_fallback(text).items():
                stages.setdefault(k, v)
        stages = _BS.need_obs(text, stages)
        nst = len([k for k in _BS._STAGES if stages.get(k)])
        if nst >= 3:
            return rec["key"], _BS.serialize(stages), nst
    except Exception:
        pass
    return None


def stage_skeleton(out_dir, workers=8):
    """Completa skeleton_db con esqueletos extraidos de papers que no tienen."""
    from multiprocessing import Pool

    skel_path = f"{out_dir}/skeleton_db.jsonl"
    have = set()
    for line in open(skel_path, errors="replace"):
        try:
            have.add(json.loads(line)["key"])
        except json.JSONDecodeError:
            pass
    print(f"[skeleton] ya existen {len(have)} esqueletos")

    def gen():
        for line in open(f"{out_dir}/papers_db.jsonl", errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in have or r["kind"] in ("unam-en", "synth"):
                continue
            if len(r["text"]) < 800:      # sin masa para extraer
                continue
            yield {"key": r["key"], "text": r["text"], "domain": r.get("domain")}

    added = 0; seen_n = 0
    with Pool(workers) as pool, open(skel_path, "a") as out:
        for res in pool.imap_unordered(_skel_work, gen(), chunksize=64):
            seen_n += 1
            if res:
                k, sk, ne = res
                out.write(json.dumps({"key": k, "skeleton": sk,
                                      "etapas": ne, "src": "extracted"},
                                     ensure_ascii=False) + "\n")
                added += 1
            if seen_n % 20000 == 0:
                out.flush()
                print(f"[skeleton] {seen_n} seen, added={added}", flush=True)
    print(f"[skeleton] DONE seen={seen_n} added={added} total={len(have)+added}")


def stage_stats(out_dir):
    n = Counter(); sk = Counter()
    for line in open(f"{out_dir}/skeleton_db.jsonl", errors="replace"):
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        sk["total"] += 1
        sk[f"src:{s.get('src')}"] += 1
        sk[f"etapas:{s.get('etapas')}"] += 1
    tok_est = 0
    for line in open(f"{out_dir}/papers_db.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n[f"kind:{r['kind']}"] += 1
        n[f"lang:{r.get('lang')}"] += 1
        n[f"dom:{r.get('domain')}"] += 1
        tok_est += len(r["text"]) // 4   # ~4 chars/token aprox
    rep = {"papers": dict(n), "skeletons": dict(sk),
           "tokens_est": tok_est}
    print(json.dumps(rep, indent=1)[:6000])
    with open(f"{out_dir}/corpus_stats.json", "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "skeleton", "stats"])
    ap.add_argument("--out", default=f"{DATA}/papersdb")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "collect":
        stage_collect(a.out)
    elif a.stage == "skeleton":
        stage_skeleton(a.out, a.workers)
    else:
        stage_stats(a.out)
