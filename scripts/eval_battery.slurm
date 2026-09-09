#!/usr/bin/env bash
#SBATCH --job-name=eval-battery
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_battery_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_battery_%j.err
# =============================================================================
# eval_battery.slurm — corre la batería de negativos graduados (L0-L3) sobre UN
# checkpoint de un run YA ENTRENADO y produce el veredicto por nivel.
#
# Responde la pregunta del postmortem F2: ¿hay señal escondida o falsación
# fuerte? L0 coherencia | L1 dominio | L2 orden de etapas | L3 inferencia.
#
# NO toca el run vivo: solo LEE checkpoint-g*/model.pt. ~15-30 min de 1 GPU
# (4 niveles x 512 pares x 1 ckpt, --no-gen).
#
# Vars (sbatch --export):
#   BASE       repo en el cluster      (default /beegfs/a474r867/ecoreasoner)
#   RUN_DIR    run a evaluar           (default $BASE/runs/f2-spanes)
#   CKPT       model.pt concreto       (default: último checkpoint-gN del run)
#   PAIRS_DIR  dir con pairs_L*.jsonl  (default $BASE/runs/pairs_hard)
#   CONFIG     run.yaml del harness    (default f0-span-esqueleto.yaml)
#   OUT_DIR    salida                  (default $RUN_DIR/battery_L0L3)
#   FAMILY     pro6000|a100|l40|q6000  (default: autodetect por gres)
#
# Uso:
#   sbatch --export="RUN_DIR=$BASE/runs/f2-spanes" scripts/eval_battery.slurm
# =============================================================================
set -uo pipefail
BASE="${BASE:-/beegfs/a474r867/ecoreasoner}"
SIF="${SIF:-$BASE/pytorch-cuda.sif}"
OVL="${OVL:-$BASE/blackwell-torch.overlay}"

# familia -> intérprete (pro6000/sm_120 necesita overlay + /bb/bwvenv)
FAMILY="${FAMILY:-}"
APP_CUR="--nv"; APP_PY="/opt/conda/bin/python"
if [ -z "$FAMILY" ] && [ -n "${SLURM_STEP_GRES:-}" ]; then
    FAMILY=$(echo "$SLURM_STEP_GRES" | grep -oE 'pro6000|a100|l40|q6000' | head -1 || true)
fi
if [ -z "$FAMILY" ]; then
    case "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oiE 'gres=[^ ]*' | head -1)" in
        *pro6000*|*Pro6000*|*PRO6000*) FAMILY=pro6000;;
    esac
fi
if [ "$FAMILY" = "pro6000" ]; then
    APP_CUR="--nv --overlay $OVL:ro"; APP_PY="/bb/bwvenv/bin/python"
fi
PY="${PY:-$APP_PY}"

RUN_DIR="${RUN_DIR:-$BASE/runs/f2-spanes}"
PAIRS_DIR="${PAIRS_DIR:-$BASE/runs/pairs_hard}"
CONFIG="${CONFIG:-$BASE/harness/configs/f0-span-esqueleto.yaml}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/battery_L0L3}"
mkdir -p "$OUT_DIR"

# checkpoint: explícito o el mayor step (mismo patrón que los watchers)
CKPT="${CKPT:-}"
if [ -z "$CKPT" ]; then
    CKPT_STEP=$(ls -d "$RUN_DIR"/checkpoint-g* 2>/dev/null \
        | grep -oE 'g[0-9]+$' | tr -d g | sort -n | tail -1)
    [ -z "$CKPT_STEP" ] && { echo "sin checkpoints en $RUN_DIR"; exit 1; }
    CKPT="$RUN_DIR/checkpoint-g$CKPT_STEP/model.pt"
fi
echo "=== eval_battery CKPT=$CKPT PAIRS_DIR=$PAIRS_DIR FAMILY=${FAMILY:-auto} $(date -u) ==="

rc_all=0
for LVL in 0 1 2 3; do
    PAIRS="$PAIRS_DIR/pairs_L${LVL}.jsonl"
    OUT="$OUT_DIR/battery_L${LVL}.json"
    if [ ! -f "$PAIRS" ]; then echo "[L$LVL] falta $PAIRS — skip"; continue; fi
    echo "=== [L$LVL] $PAIRS ==="
    apptainer exec $APP_CUR "$SIF" "$PY" \
        "$BASE/harness/suite_smoke.py" \
        --ckpt "$CKPT" --pairs "$PAIRS" --config "$CONFIG" \
        --out "$OUT" --no-gen
    rc=$?
    [ $rc -ne 0 ] && { echo "[L$LVL] eval rc=$rc"; rc_all=$rc; }
done

echo "=== veredicto batería ==="
apptainer exec $APP_CUR "$SIF" "$PY" \
    "$BASE/scripts/verdict.py" --battery "$OUT_DIR" \
    --out "$OUT_DIR/battery_verdict.json"
vrc=$?
[ $vrc -ne 0 ] && rc_all=$vrc
echo "=== eval_battery exit $rc_all $(date -u) ==="
exit $rc_all
