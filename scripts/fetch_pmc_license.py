#!/usr/bin/env python3
"""fetch_pmc_license.py — descarga metadata/*.json de PMC-OA-S3 y extrae license_code.

Lee una lista de rutas 'metadata/PMC{n}.{v}.json' (stdin o archivo), descarga el
.json con ThreadPoolExecutor, y escribe <OUT>.jsonl con:
  {"pmcid","version","pmid","doi","license_code","is_pmc_openaccess","is_manuscript","is_retracted","fetch_ok"}

Uso: python3 fetch_pmc_license.py --paths <file> --out <out.jsonl> [--workers N]
"""
import argparse, concurrent.futures as cf, json, urllib.request, time, sys, os

BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"

def load_paths_file(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]

def fetch(rel_path):
    url = BASE + rel_path
    rec = {"pmcid_path": rel_path, "fetch_ok": False, "license_code": None,
           "pmid": None, "doi": None, "version": None,
           "is_pmc_openaccess": None, "is_manuscript": None, "is_retracted": None}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ecoreasoner-provenance/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        rec["fetch_ok"] = True
        rec["pmcid"] = d.get("pmcid")
        rec["version"] = d.get("version")
        rec["pmid"] = d.get("pmid")
        rec["doi"] = d.get("doi")
        rec["license_code"] = d.get("license_code")
        rec["is_pmc_openaccess"] = d.get("is_pmc_openaccess")
        rec["is_manuscript"] = d.get("is_manuscript")
        rec["is_retracted"] = d.get("is_retracted")
        # pmcid may need normalization (some json use 'PMC...' or int/id only)
        if not rec["pmcid"]:
            # derive from path metadata/PMC123.1.json
            core = rel_path[len("metadata/"):]
            rec["pmcid"] = "PMC" + core.split(".")[0]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = load_paths_file(args.paths)
    if args.limit:
        paths = paths[: args.limit]
    total = len(paths)
    t0 = time.time()
    ok = 0
    with open(args.out, "w") as f:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, rec in enumerate(ex.map(fetch, paths), 1):
                f.write(json.dumps(rec) + "\n")
                if rec["fetch_ok"]:
                    ok += 1
                if i % 5000 == 0:
                    el = time.time() - t0
                    rate = i / el
                    eta = (total - i) / rate
                    print(f"[{time.strftime('%H:%M:%S')}] {i}/{total} ok={ok} "
                          f"{rate:.0f}/s ETA {eta/60:.0f}min", flush=True)
    el = time.time() - t0
    print(f"DONE {total} ok={ok} in {el/60:.1f}min -> {args.out}", flush=True)

if __name__ == "__main__":
    main()