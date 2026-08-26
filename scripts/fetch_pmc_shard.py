#!/usr/bin/env python3
"""fetch_pmc_shard.py — descarga full-text PMC por shard (SBATCH array friendly).

Lee un archivo plano de PMIDs, cada tarea array toma SU slice
[offset:offset+chunk_size], y descarga el full-text de cada pmid vía
pmc-oa-opendata S3 (como map_pmc_fulltext.py pero con sharding para paralelizar).
Usa el pkl _cand_pmc.pkl (pmid->pmcid ya cruzado) para no recargar PMC-ids.

Uso (slurm array):
  sbatch --array=0-3 pmc_fetch_array.slurm
  -> cada SLURM_ARRAY_TASK_ID procesa chunk pmids[start:end]

Escribe: data/fulltext_corpus_c{id}.jsonl  (parcial, se unen luego)
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error
import pickle

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"

def get_fulltext(pmcid, max_ver=3, timeout=30):
    num = pmcid.replace("PMC", "")
    for v in range(1, max_ver + 1):
        url = f"{S3_BASE}/PMC{num}.{v}/PMC{num}.{v}.txt"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                if r.status == 200:
                    return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return ""
        except Exception:
            return ""
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids_file", default="/beegfs/a474r867/ecoreasoner/data/_new_pmids.txt")
    ap.add_argument("--pmc_pkl", default="/beegfs/a474r867/ecoreasoner/data/_cand_pmc.pkl")
    ap.add_argument("--chunk_id", type=int, default=0, help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--n_chunks", type=int, default=4)
    ap.add_argument("--out_dir", default="/beegfs/a474r867/ecoreasoner/data")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    pmids = [l.strip() for l in open(args.pmids_file) if l.strip()]
    total = len(pmids)
    # shard bounds
    start = (total * args.chunk_id) // args.n_chunks
    end = (total * (args.chunk_id + 1)) // args.n_chunks
    mine = pmids[start:end]
    print(f"[{time.strftime('%H:%M:%S')}] chunk {args.chunk_id}: {len(mine)} pmids (of {total}, slice {start}:{end})", flush=True)

    with open(args.pmc_pkl, "rb") as f:
        pm2pmc = pickle.load(f)
    wanted = [(p, pm2pmc[p]) for p in mine if p in pm2pmc]
    print(f"  con PMCID en slice: {len(wanted)}", flush=True)

    out_path = Path(args.out_dir) / f"fulltext_corpus_c{args.chunk_id}.jsonl"
    t0 = time.time(); n_ok = n_fail = n_tok = 0
    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(get_fulltext, pmc): (p, pmc) for p, pmc in wanted}
        for fut in as_completed(futs):
            p, pmc = futs[fut]
            txt = fut.result() if not args.dry else "DUMMY"
            if args.dry:
                n_ok += 1; continue
            if txt and len(txt.strip()) > 200:
                f.write(json.dumps({"text": txt.strip(), "pmid": int(p),
                                    "pmcid": pmc, "source": "pmc-full"},
                                   ensure_ascii=False) + "\n")
                n_ok += 1; n_tok += len(txt) // 4
            else:
                n_fail += 1
            if (n_ok + n_fail) % 500 == 0:
                print(f"  {n_ok+n_fail}/{len(wanted)} (ok {n_ok} fail {n_fail} {time.time()-t0:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] chunk {args.chunk_id} DONE: ok={n_ok} fail={n_fail} ~{n_tok/1e6:.1f}M tok -> {out_path}", flush=True)

if __name__ == "__main__":
    main()