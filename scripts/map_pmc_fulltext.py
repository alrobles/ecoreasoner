#!/usr/bin/env python3
"""map_pmc_fulltext.py — PMID -> PMCID -> PMC Cloud (S3) full-text downloader.

Construye un corpus de TEXTO COMPLETO desde el PMC Open Access via AWS Cloud
Service (sin FTP deprecado, sin captcha, acceso anonimo via HTTPS).

Fuente de mapeo:  PMC-ids.csv.gz (PMID<->PMCID<->DOI, ~11.4M records)
Fuente de texto:   https://pmc-oa-opendata.s3.amazonaws.com/PMC{id}.{v}/PMC{id}.{v}.txt
                   (plain text extrado del XML JATS; tambien .xml/.json disponibles)

Pipeline:
  1. leer eco_corpus.jsonl (N docs con pmid) o lista de pmids
  2. cargar PMC-ids.csv.gz -> dict pmid->pmcid (solo filas con PMCID)
  3. para cada pmid con pmcid (>0) y pmcid numerico: intentar texto completo
     (versiones 1..max_ver; el .txt de la version con contenido)
  4. escribir fulltext_corpus.jsonl {"text","pmid","pmcid","domain","year","source":"pmc-full"}

Uso: python3 map_pmc_fulltext.py --pmids_corpus eco_corpus.jsonl \
        --ids /beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz \
        --out fulltext_corpus.jsonl --limit 5000 [--max_workers 8] [--dry]
"""
import argparse, csv, gzip, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"

def load_pmid_pmcid(ids_gz_path, prefer_comm=True):
    """Lee PMC-ids.csv.gz -> dict {pmid: pmcid}. Prefiere PMC de columnas 9,10."""
    d = {}
    with gzip.open(ids_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 10:
                continue
            pmcid = row[8].strip()
            pmid = row[9].strip()
            if pmcid.startswith("PMC") and pmid.isdigit():
                d.setdefault(pmid, pmcid)  # primer PMCID (suele ser el comm)
    return d

def get_fulltext(pmcid, max_ver=3, timeout=25):
    """Descarga el .txt del PMC via S3 HTTPS. Devuelve texto o ''."""
    num = pmcid.replace("PMC", "")
    for v in range(1, max_ver+1):
        url = f"{S3_BASE}/PMC{num}.{v}/PMC{num}.{v}.txt"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                if r.status == 200:
                    return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # probar siguiente version
            return ""
        except Exception:
            return ""
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids_corpus", help="jsonl con campo 'pmid' (los que NO tengan full-text se saltan; si no, usar --pmids_stdin)")
    ap.add_argument("--ids", default="/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/fulltext_corpus.jsonl")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] cargando PMC-ids ...", flush=True)
    pm2pmc = load_pmid_pmcid(args.ids)
    print(f"  pmids->pmcid: {len(pm2pmc)}", flush=True)

    # pmids a procesar
    pmids = []
    if args.pmids_corpus:
        with open(args.pmids_corpus) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("pmid"): pmids.append(str(d["pmid"]))
                except Exception: pass
    pmids = list(dict.fromkeys(pmids))  # dedup
    print(f"  pmids en corpus: {len(pmids)}", flush=True)

    # filtrar los que tienen pmcid
    wanted = [(p, pm2pmc[p]) for p in pmids if p in pm2pmc]
    print(f"  con PMC: {len(wanted)} (sin PMC: {len(pmids)-len(wanted)})", flush=True)
    wanted = wanted[: args.limit]
    print(f"  a descargar: {len(wanted)} (dry={args.dry})", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = n_tok = 0
    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(get_fulltext, pmc): (p, pmc) for p, pmc in wanted}
        for fut in as_completed(futs):
            p, pmc = futs[fut]
            txt = fut.result() if not args.dry else "DUMMY"
            if args.dry:
                n_ok += 1; continue
            if txt and len(txt.strip()) > 200:
                f.write(json.dumps({"text": txt.strip(), "pmid": int(p),
                                    "pmcid": pmc, "source": "pmc-full"}, ensure_ascii=False) + "\n")
                n_ok += 1; n_tok += len(txt)//4  # ~4 chars/token
            else:
                n_fail += 1
            if (n_ok+n_fail) % 250 == 0:
                print(f"  {n_ok+n_fail}/{len(wanted)} (ok {n_ok}, fail {n_fail}, {time.time()-t0:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE: ok={n_ok} fail={n_fail} ~{n_tok/1e6:.1f}M tokens -> {out_path}", flush=True)

if __name__ == "__main__":
    main()