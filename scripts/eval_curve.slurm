#!/usr/bin/env bash
#SBATCH --job-name=eval-curve
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_curve_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_curve_%j.err
# =============================================================================
# eval_curve.slurm — curva de discriminación (pairwise_acc / mean_delta) sobre
# los checkpoints checkpoint-g<N> de un run. Llama a scripts/eval_curve.py, que
# lanza harness/suite_smoke.py por checkpoint y escribe <run-dir>/eval_curve.jsonl.
#
# SIF base cu121 SIN overlay blackwell: el python del SIF base es
# /opt/conda/bin/python (/bb/bwvenv solo existe dentro de
# blackwell-torch.overlay, que no hace falta aquí).
#
# Vars (sobrescribibles con sbatch --export):
#   BASE      repo en el cluster    (default /beegfs/a474r867/ecoreasoner)
#   RUN_DIR   run a evaluar         (default $BASE/runs/f1-hetero)
#   PAIRS     jsonl de pares        (default $BASE/runs/pairs.jsonl)
#   CONFIG    run.yaml del harness  (default $BASE/harness/configs/f0-span-esqueleto.yaml)
#   EVERY     eval cada N ckpts     (default 1 = todos)
#   MIN_STEP  saltar step < N       (default 0)
#   OUT       salida jsonl          (default <run-dir>/eval_curve.jsonl)
#   EXTRA     args extra            (p.ej. "--no-gen" para saltar generación)
#   SIF / PY  imagen / intérprete   (defaults: $BASE/pytorch-cuda.sif, /opt/conda/bin/python)
#
# Uso:
#   sbatch scripts/eval_curve.slurm
#   sbatch --export="RUN_DIR=/beegfs/a474r867/ecoreasoner/runs/f1-hetero,EVERY=4,EXTRA=--no-gen" \
#       scripts/eval_curve.slurm
# =============================================================================
set -uo pipefail
BASE="${BASE:-/beegfs/a474r867/ecoreasoner}"
SIF="${SIF:-$BASE/pytorch-cuda.sif}"
OVL="${OVL:-$BASE/blackwell-torch.overlay}"

# PY depende de la familia: pro6000 (Blackwell, sm_120) necesita OVERLAY + /bb/bwvenv
# (el SIF base cu121 no tiene kernel para sm_120 -> "no kernel image"); para cu121
# (L40/Q6000/A100) basta SIF base + /opt/conda/bin/python.
FAMILY="${FAMILY:-}"
APP_CUR="--nv"
APP_PY="/opt/conda/bin/python"
# deteccion por el --gres del job (p.ej. "gpu:pro6000:1")
if [ -z "$FAMILY" ] && [ -n "${SLURM_STEP_GRES:-}" ]; then
    FAMILY=$(echo "$SLURM_STEP_GRES" | grep -oE 'pro6000|a100|l40|q6000' | head -1 || true)
fi
if [ -z "$FAMILY" ]; then
    case "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'gres=[^ ]*' | head -1)" in
        *pro6000*) FAMILY=pro6000;;
    esac
fi
if [ "$FAMILY" = "pro6000" ]; then
    APP_CUR="--nv --overlay $OVL:ro"
    APP_PY="/bb/bwvenv/bin/python"
fi
PY="${PY:-$APP_PY}"
RUN_DIR="${RUN_DIR:-$BASE/runs/f1-hetero}"
PAIRS="${PAIRS:-$BASE/runs/pairs.jsonl}"
CONFIG="${CONFIG:-$BASE/harness/configs/f0-span-esqueleto.yaml}"
EVERY="${EVERY:-1}"
MIN_STEP="${MIN_STEP:-0}"
OUT="${OUT:-$RUN_DIR/eval_curve.jsonl}"
EXTRA="${EXTRA:-}"

echo "=== eval_curve RUN_DIR=$RUN_DIR EVERY=$EVERY MIN_STEP=$MIN_STEP FAMILY=${FAMILY:-auto} PY=$PY $(date -u) ==="
apptainer exec $APP_CUR "$SIF" "$PY" \
    "$BASE/scripts/eval_curve.py" \
    --run-dir "$RUN_DIR" --pairs "$PAIRS" --config "$CONFIG" \
    --out "$OUT" --every "$EVERY" --min-step "$MIN_STEP" $EXTRA
rc=$?
echo "=== eval_curve exit $rc $(date -u) ==="
exit $rc
