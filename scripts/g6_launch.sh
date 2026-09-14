#!/usr/bin/env bash
# g6_launch.sh — generacion G6: cruces del ganador G5 + gen nuevo mras
# (masking adaptativo por EMA de CE por token — "Mask Is What DLLM Needs").
#
# Uso: BASE=hinrcf10 bash g6_launch.sh
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=${BASE:-hinrcf10}
DATA=/beegfs/a474r867/ecoreasoner/data
AUG2=$DATA/train_ids_b4aug2.npz
HI='{"b_l":0.30,"b_h":0.99}'

EXTRA=""
if [ "$BASE" = "hinrcf10" ]; then EXTRA="ROLE_MASK=0"; fi
run() { # tag seed extra_env...
    local tag=$1 seed=$2; shift 2
    env TAG="$tag" MASK_TYPE=random CANDIDATE_FOCUS=1.0 \
        MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=$seed $EXTRA "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

# --- gen nuevo: mras (masking adaptativo; bajo CF10 pondera la eleccion
#     de etapa por dificultad media; en el fallback pondera posiciones) ---
run g6-mras-s1    11 MRAS_GAMMA=0.5 MRAS_FLOOR=0.3
run g6-mras-s2    12 MRAS_GAMMA=0.5 MRAS_FLOOR=0.3
# --- corrective a dosis baja: p=0.10 fue bimodal (0.279/0.581) ---
run g6-corrlo-s1  21 CORRECTIVE_P=0.05
run g6-corrlo-s2  22 CORRECTIVE_P=0.05
# --- interacciones: mras estabiliza corrective? numw+mras mueven num? ---
run g6-cormras-s1 31 CORRECTIVE_P=0.10 MRAS_GAMMA=0.5 MRAS_FLOOR=0.3
run g6-mrashi-s1  41 MRAS_GAMMA=1.0 MRAS_FLOOR=0.3
# --- control interno de la generacion ---
run g6-base-s1    51
run g6-base-s2    52

echo "G6 lanzada (base=$BASE, 8 jobs)"
