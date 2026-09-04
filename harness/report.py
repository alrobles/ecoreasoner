#!/usr/bin/env python3
"""report.py — ensambla el report.json estándar de un micro-run (Fase 3, F0).

Recoge: sha256 del config, seed, timestamps, tag, discriminación, generación,
fluidez y estado. Además lleva el índice runs/index.jsonl (append por línea)
para que compare.py construya tablas A/B/N.

Uso:
  python harness/report.py --config harness/configs/run.yaml \
      --suite runs/f0-smoke/report.suite.json --out runs/f0-smoke/report.json
"""
import argparse, hashlib, json, time as _time
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def append_index(index_path: str, rec: dict) -> None:
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True, help="salida de suite_smoke.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="runs/index.jsonl")
    ap.add_argument("--job-id", default="", help="SlurmJobID si vino de sbatch")
    ap.add_argument("--status", default="complete")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    suite = json.loads(Path(args.suite).read_text())

    rec = {
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_sha256": sha256_file(args.config),
        "job_id": args.job_id,
        "status": args.status,
        "seed": cfg["seed"],
        "tag": cfg["out"]["tag"],
        "job": cfg["job"],
        "model": cfg["model"],
        "eval": cfg["eval"],
        "suite": suite,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    append_index(args.index, {k: rec[k] for k in
                  ("ts", "config_sha256", "job_id", "status", "seed", "tag")}
                 | {"pairwise_acc": suite.get("discrimination", {}).get("pairwise_acc"),
                    "mean_delta": suite.get("discrimination", {}).get("mean_delta")})
    print(json.dumps({"out": args.out, "indexed": args.index}))


if __name__ == "__main__":
    main()