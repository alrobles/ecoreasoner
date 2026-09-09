#!/usr/bin/env python3
"""fetch_ecoevorxiv.py — descarga full-text de los preprints EcoEvoRxiv.

Lee ecoevorxiv_meta.jsonl (del discover), para cada article_id:
  1. GET /repository/view/<id>/  -> halla file_id (/repository/object/<id>/download/<file_id>/)
  2. DESCARGAR el PDF (streaming a disco temporal)
  3. Extraer texto con pdfplumber -> {"text", "article_id", "title", "source":"ecoevorxiv-pdf"}
  (fallback: si solo hay abstract en la vista, usar abstract con source ecovevorxiv-abstract)

Sharding por SLURM_ARRAY_TASK_ID para paralelizar. Escribe
  ecoevorxiv_fulltext_c<chunk>.jsonl
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error

BASE = "https://ecoevorxiv.org"
UA = {"User-Agent": "Mozilla/5.0 (research corpus miner; contact @EcoEvoRxiv)"}
DL_RE = re.compile(r'/repository/object/(\d+)/download/(\d+)/')
ABS_RE = re.compile(r'<h2[^>]*class="title"[^>]*>.*?</h2>\s*<div[^>]*class="(?:abstract|description)"[^>]*>(.*?)</div>', re.S)

def fetch(u, timeout=30, retries=3):
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            time.sleep(1.5)
    return b""

def strip_tags(h):
    return re.sub(r'<[^>]+>', '', h or '').replace('&amp;','&').replace('&#x27;',"'") \
        .replace('&quot;','"').replace('&lt;','<').replace('&gt;','>').strip()

def download_pdf(file_id, article_id, tmp):
    u = f"{BASE}/repository/object/{article_id}/download/{file_id}/"
    data = fetch(u, timeout=90)
    if not data or not data[:5].startswith(b'%PDF'):
        return None
    p = Path(tmp) / f"eev_{article_id}.pdf"
    p.write_bytes(data)
    return p

def pdf_to_text(p):
    try:
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    except Exception as e:
        return ""

def extract_file_id(html_text):
    m = DL_RE.search(html_text)
    # DL_RE = object/(\d+)/download/(\d+)/ -> group(2) es el file_id
    return m.group(2) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="/beegfs/a474r867/ecoreasoner/data/ecoevorxiv_meta.jsonl")
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=4)
    ap.add_argument("--out_dir", default="/beegfs/a474r867/ecoreasoner/data")
    ap.add_argument("--tmp", default="/beegfs/a474r867/ecoreasoner/tmp_eev")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(args.meta) if l.strip()]
    total = len(items)
    start = (total * args.chunk_id) // args.n_chunks
    end = (total * (args.chunk_id + 1)) // args.n_chunks
    mine = items[start:end]
    print(f"[{time.strftime('%H:%M:%S')}] chunk {args.chunk_id}: {len(mine)} items (of {total})", flush=True)

    out = Path(args.out_dir) / f"ecoevorxiv_fulltext_c{args.chunk_id}.jsonl"
    n_pdf = n_abs = n_fail = 0
    t0 = time.time()

    def process(it):
        aid = it["article_id"]
        # 1) vista -> file id
        view = fetch(f"{BASE}/repository/view/{aid}/").decode("utf-8", "replace")
        fid = extract_file_id(view)
        if fid:
            p = download_pdf(fid, aid, args.tmp)
            if p:
                txt = pdf_to_text(p)
                if txt and len(txt.strip()) > 200:
                    return {"text": txt.strip(), "article_id": aid,
                            "title": it.get("title"), "source": "ecoevorxiv-pdf"}
                p.unlink(missing_ok=True)
        # 2) fallback: abstract de la vista
        m = ABS_RE.search(view)
        if m:
            ab = strip_tags(m.group(1))
            if len(ab) > 100:
                return {"text": ab, "article_id": aid,
                        "title": it.get("title"), "source": "ecoevorxiv-abstract"}
        return None

    with open(out, "w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, it): it for it in mine}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_pdf += 1 if r["source"] == "ecoevorxiv-pdf" else 0
                n_abs += 1 if r["source"] == "ecoevorxiv-abstract" else 0
            else:
                n_fail += 1
            if (n_pdf + n_abs + n_fail) % 50 == 0:
                print(f"  {(n_pdf+n_abs+n_fail)}/{len(mine)} pdf={n_pdf} abs={n_abs} fail={n_fail} {time.time()-t0:.0f}s", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] chunk {args.chunk_id} DONE: pdf={n_pdf} abs={n_abs} fail={n_fail} -> {out}", flush=True)

if __name__ == "__main__":
    main()