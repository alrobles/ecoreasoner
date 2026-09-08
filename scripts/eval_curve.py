#!/usr/bin/env python3
"""eval_curve.py — curva de discriminación pairwise sobre los checkpoints de un run.

Recorre <run-dir>/checkpoint-g<N>/model.pt (ordenados por N), ejecuta
harness/suite_smoke.py en cada uno vía subprocess y vuelca una línea jsonl por
checkpoint: {"step": N, "pairwise_acc": X, "mean_delta": Y}.

Solo stdlib (json, subprocess, argparse, pathlib, re). Pensado para correr
DENTRO del contenedor apptainer (ver scripts/eval_curve.slurm), donde torch
está disponible para suite_smoke.py.

Uso:
  python3 scripts/eval_curve.py \
      --run-dir /beegfs/a474r867/ecoreasoner/runs/f1-hetero \
      --pairs /beegfs/a474r867/ecoreasoner/runs/pairs.jsonl \
      --config /beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml \
      --every 4 --min-step 100
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# suite_smoke.py del mismo repo (scripts/../harness), i.e. en el cluster:
# /beegfs/a474r867/ecoreasoner/harness/suite_smoke.py
SUITE_DEFAULT = Path(__file__).resolve().parents[1] / "harness" / "suite_smoke.py"

CKPT_RE = re.compile(r"^checkpoint-g(\d+)$")


def find_checkpoints(run_dir):
    """Lista (step, model.pt) de los subdirs checkpoint-g<N> con model.pt."""
    out = []
    for d in run_dir.iterdir():
        m = CKPT_RE.match(d.name)
        if m and d.is_dir() and (d / "model.pt").is_file():
            out.append((int(m.group(1)), d / "model.pt"))
    return sorted(out)


def load_existing_rows(out_path):
    """Carga eval_curve.jsonl existente en un dict {step: row}, conservando la última fila de cada step.

    Acepta filas con campo 'error' y las mantiene si son la última para ese step.
    Líneas malformadas o sin 'step' se ignoran con advertencia.
    """
    by_step = {}
    if not out_path.exists():
        return by_step
    for i, line in enumerate(out_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            step = row.get("step")
            if step is not None:
                by_step[int(step)] = row
        except Exception as e:
            print(f"[eval_curve] ignorando línea {i} de {out_path}: {e}", file=sys.stderr)
    return by_step


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="dir del run (contiene checkpoint-g<N>/model.pt)")
    ap.add_argument("--pairs", required=True, help="jsonl de pares tokenizados")
    ap.add_argument("--config", required=True, help="run.yaml del harness")
    ap.add_argument("--out", default=None,
                    help="salida jsonl (default <run-dir>/eval_curve.jsonl)")
    ap.add_argument("--every", type=int, default=1,
                    help="evaluar cada N checkpoints de la lista ordenada (default 1 = todos); "
                         "el último checkpoint siempre se incluye")
    ap.add_argument("--min-step", type=int, default=0,
                    help="saltar checkpoints con step < N (default 0)")
    ap.add_argument("--suite", default=str(SUITE_DEFAULT),
                    help="ruta a suite_smoke.py (default: <repo>/harness/suite_smoke.py)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-gen", action="store_true",
                    help="propagar --no-gen a suite_smoke (solo discriminación, más rápido)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "eval_curve.jsonl"
    tmp_path = run_dir / ".eval_tmp.json"  # tmp DENTRO del run-dir (no /tmp)
    existing = load_existing_rows(out_path)

    ckpts = [(n, p) for n, p in find_checkpoints(run_dir) if n >= args.min_step]
    every = max(1, args.every)
    sel = [c for i, c in enumerate(ckpts) if i % every == 0]
    if ckpts and ckpts[-1] not in sel:
        sel.append(ckpts[-1])
    if not sel:
        print(f"[fatal] sin checkpoints checkpoint-g<N>/model.pt en {run_dir} "
              f"(min-step={args.min_step})", file=sys.stderr)
        sys.exit(2)
    print(f"[eval_curve] {len(sel)}/{len(ckpts)} checkpoints -> {out_path}")

    rows = []
    for step, ckpt in sel:
        tmp_path.unlink(missing_ok=True)  # no releer un report viejo si falla
        cmd = [sys.executable, args.suite,
               "--ckpt", str(ckpt), "--config", args.config,
               "--pairs", args.pairs, "--out", str(tmp_path),
               "--device", args.device]
        if args.no_gen:
            cmd.append("--no-gen")
        print(f"[eval_curve] step={step} ckpt={ckpt}", flush=True)
        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"suite_smoke exit {proc.returncode}")
            rep = json.loads(tmp_path.read_text())
            disc = rep["discrimination"]
            row = {"step": step,
                   "pairwise_acc": disc["pairwise_acc"],
                   "mean_delta": disc["mean_delta"]}
        except Exception as e:  # report ausente/roto o sin 'discrimination'
            row = {"step": step, "pairwise_acc": None, "mean_delta": None,
                   "error": str(e)}
            print(f"[eval_curve] step={step} ERROR: {e}", file=sys.stderr)
        rows.append(row)

    tmp_path.unlink(missing_ok=True)

    # Reescribir la curva completa: deduplicar por step conservando la última fila.
    all_rows = dict(existing)
    for r in rows:
        all_rows[r["step"]] = r
    final_rows = [all_rows[s] for s in sorted(all_rows)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = out_path.with_name(out_path.name + ".tmp")
    with out_tmp.open("w") as f:
        for r in final_rows:
            f.write(json.dumps(r) + "\n")
    os.replace(out_tmp, out_path)

    print(f"\n{'step':>8} {'pairwise_acc':>12} {'mean_delta':>10}")
    for r in rows:
        pa = f"{r['pairwise_acc']:.4f}" if isinstance(r["pairwise_acc"], (int, float)) else "-"
        md = f"{r['mean_delta']:.5f}" if isinstance(r["mean_delta"], (int, float)) else "-"
        print(f"{r['step']:>8} {pa:>12} {md:>10}")
    print(f"[eval_curve] escrito {out_path}")


if __name__ == "__main__":
    main()
