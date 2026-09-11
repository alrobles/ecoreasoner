#!/usr/bin/env python3
"""launch_v2_train.py — lanza un run v2 a partir de un config YAML.

Lee un config v2 y genera el sbatch --export adecuado para
scripts/moe_v4_micro_v2.slurm. Evita errores de comas/escapes
manejando el --export con separador ',' (valores sin comas).

Uso:
  python scripts/launch_v2_train.py harness/configs/f0-span-v2-rope.yaml
  python scripts/launch_v2_train.py harness/configs/f0-span-v2-50m.yaml --steps 5000 --dry-run
"""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path

import yaml

BASE = "/beegfs/a474r867/ecoreasoner"
SLURM = f"{BASE}/scripts/moe_v4_micro_v2.slurm"


def _b2i(v):
    return 1 if v else 0


def build_export(cfg, target_steps, batch, grad_accum, pairs_dir):
    m = cfg["model"]
    t = cfg["train"]
    o = cfg["out"]
    tag = o["tag"]
    # valores simples sin comas, separador seguro
    pairs_dir = pairs_dir or f"{BASE}/runs/pairs_hard_v2"
    return {
        "TAG": tag,
        "DATA_CACHE": t["data_cache"],
        "CONFIG": str(Path(cfg["_path"]).resolve()),
        "TARGET_STEPS": str(target_steps),
        "PAIRS_DIR": pairs_dir,
        "MASK_TYPE": t["mask_type"],
        "MASK_SCHEDULE": t["mask_schedule"],
        "SPAN_LEN": str(t.get("span_len", 64)),
        "WHOLE_STAGE": str(_b2i(t.get("whole_stage", True))),
        "LR": str(t["lr"]),
        "LR_DECAY": t["lr_decay"],
        "LR_MIN_RATIO": str(t["lr_min_ratio"]),
        "WARMUP": str(t["warmup"]),
        "GRAD_CLIP": str(t["grad_clip"]),
        "EMA_DECAY": str(t.get("ema_decay", 0.0)),
        "WEIGHT_TYING": str(_b2i(m.get("weight_tying", False))),
        "USE_ROPE": str(_b2i(m.get("use_rope", False))),
        "BATCH_SIZE": str(batch or t.get("batch_size", 8)),
        "GRAD_ACCUM": str(grad_accum or t.get("grad_accum", 2)),
        "VOCAB": str(m["vocab"]),
        "HIDDEN": str(m["hidden"]),
        "LAYERS": str(m["layers"]),
        "HEADS": str(m["heads"]),
        "FF_MULT": str(m["ff_mult"]),
        "N_EXPERTS": str(m.get("n_experts", 1)),
        "EXPERT_K": str(m.get("k", 1)),
        "SEQ_LEN": str(m["seq_len"]),
        "CURRICULUM": str(_b2i(t.get("curriculum", False))),
        "CUR_STAGES": json.dumps(t.get("cur_stages", [])),
        "ROLE_MASK": str(_b2i(t.get("role_mask", False))),
        "ROLE_CONFIG": json.dumps(t.get("role_config", {})),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="run.yaml v2")
    ap.add_argument("--steps", type=int, default=10000, help="target steps")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--sbatch-args", default="", help="extra args para sbatch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[fatal] config no existe: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg_path_str = str(cfg_path.resolve())
    # si se edita en local, el config de ejecucion esta en beegfs
    cfg_path_str = cfg_path_str.replace("/home/reumanlab/ecoreasoner", BASE)
    cfg["_path"] = cfg_path_str

    exports = build_export(cfg, args.steps, args.batch, args.grad_accum, args.pairs_dir)
    export_str = ",".join(f"{k}={v}" for k, v in exports.items())

    cmd = ["ssh", "-o", "BatchMode=yes", "kuhpc",
           f"cd {BASE} && sbatch --parsable {args.sbatch_args} --export={shlex.quote(export_str)} {SLURM}"]

    print("=" * 60)
    print(" TAG:", exports["TAG"])
    print(" DATA_CACHE:", exports["DATA_CACHE"])
    print(" COMMAND:", " ".join(cmd))
    print("=" * 60)

    if args.dry_run:
        print("[dry-run] no se lanzó")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
