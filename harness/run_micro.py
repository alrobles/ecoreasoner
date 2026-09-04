#!/usr/bin/env python3
"""run_micro.py — valida un run.yaml, arma el job slurm 1-GPU y lo ejecuta
(Fase 3, F0). NUNCA toca el trainer vivo: el job solo corre suite_smoke.py
sobre un checkpoint ya entrenado.

Modos:
  --dry          imprime el slurm y el comando sin enviar
  --submit       genera el .slurm y llama sbatch (default)
  --local        ejecuta suite_smoke.py localmente (sin slurm)

Uso:
  python harness/run_micro.py --config harness/configs/run.yaml \
      --ckpt /beegfs/.../checkpoint-best.pt --submit
"""
import argparse, os, subprocess, sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parent
TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --export=ALL

set -euo pipefail
module purge
source /etc/profile.d/modules.sh 2>/dev/null || true

{launch}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint a evaluar")
    ap.add_argument("--pairs", default="", help="jsonl de pares (opcional; si no, self-test)")
    ap.add_argument("--out", default="", help="dir de salida; default = out.dir de run.yaml")
    ap.add_argument("--mode", choices=["dry", "submit", "local"], default="submit")
    ap.add_argument("--extra-args", default="", help="flags extra para suite_smoke.py")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    jcfg, ecfg, mcfg, ocfg = cfg["job"], cfg["eval"], cfg["model"], cfg["out"]

    out_dir = Path(args.out or ocfg["dir"]) / ocfg["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[fatal] checkpoint no existe: {ckpt}", file=sys.stderr)
        sys.exit(2)

    suite_out = out_dir / "report.suite.json"
    cmd = [
        "python", str(HARNESS / "suite_smoke.py"),
        "--ckpt", str(ckpt),
        "--config", str(Path(args.config).resolve()),
        "--out", str(suite_out),
        "--device", "cuda",
    ]
    if args.pairs:
        cmd += ["--pairs", args.pairs]
    if args.extra_args:
        cmd += args.extra_args.split()
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    if args.mode == "local":
        print(f"[local] {cmd_str}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(rc)
        # ensamblar report estándar
        rep = subprocess.run(
            ["python", str(HARNESS / "report.py"),
             "--config", str(Path(args.config).resolve()),
             "--suite", str(suite_out),
             "--out", str(out_dir / "report.json")],
            check=False).returncode
        sys.exit(rep)

    # slurm
    venv = jcfg.get("venv", "/bb/bwvenv/bin/python")
    launch = cmd_str.replace("python", venv, 1)
    slurm = TEMPLATE.format(
        name=ocfg["tag"], partition=jcfg["partition"], gres=jcfg["gres"],
        cpus=jcfg["cpus"], mem=jcfg["mem"], time=jcfg["time"],
        output=str(out_dir / "job-%j.out"), launch=launch,
    )
    slurm_path = out_dir / f"{ocfg['tag']}.slurm"
    slurm_path.write_text(slurm)
    print(f"[slurm] {slurm_path}")

    if args.mode == "dry":
        print(slurm)
        # también registrar el comando report de ensamblaje
        print(f"# luego: python {HARNESS}/report.py --config ... --suite {suite_out} "
              f"--out {out_dir}/report.json --job-id $SLURM_JOB_ID")
        return

    r = subprocess.run(["sbatch", str(slurm_path)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()