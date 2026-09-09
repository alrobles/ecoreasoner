#!/usr/bin/env python3
"""match_ecoevorxiv_license.py — empareja los preprints OSF de EcoEvoRxiv (con licencia)
contra los article_id del corpus, por titulo normalizado. Genera ecoevorxiv_license_map.jsonl
con {article_id, title, osf_id, license_id, license_name, matched_by}.

Uso: python3 match_ecoevorxiv_license.py
"""
import json, re, unicodedata, sys
from pathlib import Path

BASE = Path("/beegfs/a474r867/ecoreasoner/data")

def norm(t):
    if not t: return ""
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

def load_jsonl(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except: pass
    return out

# 1) OSF preprints con licencia
osf = load_jsonl(BASE / "ecoevorxiv_osf_map.jsonl")
osf_by_norm = {}
for o in osf:
    osf_by_norm.setdefault(norm(o.get("title")), []).append(o)
print(f"[{__import__('time').strftime('%H:%M:%S')}] OSF preprints con lic: {len(osf)}")

# 2) corpora article_id (+title) de ecoevorxiv
meta = load_jsonl(BASE / "ecoevorxiv_meta.jsonl")
fulltext = []
for f in sorted(BASE.glob("ecoevorxiv_fulltext_c*.jsonl")):
    fulltext += load_jsonl(f)
print(f"[{__import__('time').strftime('%H:%M:%S')}] meta ids: {len(meta)} | fulltext docs: {len(fulltext)}")

# unificar por article_id: tomar titulo (meta) y si hay fulltext, fuente
corpus = {}
for m in meta:
    aid = str(m.get("article_id"))
    corpus.setdefault(aid, {"article_id": aid, "title": m.get("title"), "fulltext": False})
for f in fulltext:
    aid = str(f.get("article_id"))
    if aid in corpus:
        corpus[aid]["fulltext"] = True
    else:
        corpus.setdefault(aid, {"article_id": aid, "title": f.get("title"), "fulltext": True})
print(f"[{__import__('time').strftime('%H:%M:%S')}] corpus ecoevorxiv unico: {len(corpus)}")

# 3) match por titulo
matched = 0
out = []
for aid, c in corpus.items():
    nt = norm(c.get("title"))
    cand = osf_by_norm.get(nt, [])
    rec = {**c, "license_id": None, "license_name": None,
           "osf_id": None, "matched_by": None}
    if cand:
        best = cand[0]
        rec["license_id"] = best.get("license_id")
        rec["license_name"] = best.get("license_name")
        rec["osf_id"] = best.get("osf_id")
        rec["matched_by"] = "title_exact"
        matched += 1
    out.append(rec)
    print(f"[{__import__('time').strftime('%H:%M:%S')}] matched={matched}/{len(corpus)}", flush=True) if matched % 500 == 0 else None

with open(BASE / "ecoevorxiv_license_map.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"DONE matched {matched}/{len(corpus)} -> {BASE/'ecoevorxiv_license_map.jsonl'}")