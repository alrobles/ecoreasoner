#!/usr/bin/env bash
# battery_point.sh <gstep> — snapshot del ckpt y lanza la bateria de 3 evals.
# Corre EN EL CLUSTER (lo invoca watch_v5_battery.sh via ssh).
#
# Por que snapshot: el trainer retiene ~2 ckpts y crea el dir ANTES de
# escribir model.pt (~2min de save). Evals lanzados contra el dir vivo
# mueren con FileNotFoundError (carrera vista en g313001) o miden un ckpt
# distinto al de la etiqueta. Snapshot = hardlink (mismo inode, gratis;
# el rmtree del trainer solo baja el link count) con fallback a cp.
# Outdirs: $CURVE/g<G>/{dense,cons_inf,cons_ctxnec}; ckpt: $CURVE/ckpts/g<G>/.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
OUT=${RUN_OUT:-runs/v5-1b}                # moefine: RUN_OUT=runs/v5-1b-moefine
EVAL_CFG=${EVAL_CFG:-eval-moe-v5.yaml}    # moefine: EVAL_CFG=eval-moe-v5-moefine.yaml
CURVE=$OUT/battery_curve
G=${1:?uso: battery_point.sh <gstep>}
SRC=$BASE/$OUT/checkpoint-g$G
SNAP=$BASE/$CURVE/ckpts/g$G

if [ ! -f "$SRC/model.pt" ]; then
    echo "ckpt incompleto o inexistente: $SRC (model.pt ausente)" >&2
    exit 1
fi
mkdir -p "$SNAP"
for f in model.pt ema_model.pt; do
    [ -f "$SRC/$f" ] || continue
    ln "$SRC/$f" "$SNAP/$f" 2>/dev/null || cp -f "$SRC/$f" "$SNAP/$f"
done
echo "snap g$G -> $SNAP ($(ls "$SNAP" | tr '\n' ' '))"

cd "$BASE"
sbatch --export="ALL,RUNDIR=$BASE/$OUT,CKPT_DIR=$SNAP,EVAL_CFG=$EVAL_CFG,OUTDIR=$BASE/$CURVE/g$G/dense" \
    scripts/g_eval_holdout_v5.slurm
sbatch --partition=sixhour --gres=gpu:1 --job-name=c-eval \
    --export="ALL,RUNDIR=$BASE/$OUT,CKPT_DIR=$SNAP,EVAL_CFG=$EVAL_CFG,PAIRS=$BASE/runs/pairs_l3_inf,MODES=consistency consistency_profile,OUTDIR=$BASE/$CURVE/g$G/cons_inf" \
    scripts/c_contrast.slurm
sbatch --partition=sixhour --gres=gpu:1 --job-name=c-eval \
    --export="ALL,RUNDIR=$BASE/$OUT,CKPT_DIR=$SNAP,EVAL_CFG=$EVAL_CFG,PAIRS=$BASE/runs/pairs_l3_ctxnec,MODES=consistency consistency_profile,OUTDIR=$BASE/$CURVE/g$G/cons_ctxnec" \
    scripts/c_contrast.slurm
