#!/usr/bin/env bash
# g5_launch.sh — generacion G5: genes de la revision de literatura sobre
# el backbone ganador de G4 (BASE=hicf10|hinrcf10).
#
# Genes (todos con seeds explicitas):
#   corr    corrective_p=0.10 — mutaciones visibles + correccion supervisada
#           (Corrective Diffusion LM; el gen de mayor alineacion con el eval)
#   corrhi  corrective_p=0.20 — dosis alta
#   numw    loss_num_w=3.0 loss_neg_w=2.5 — DSFT-style loss weighting
#   corrnw  corrective + numw (interaccion)
#   cos     mask_schedule=cosine SIN curriculum — hipotesis contraria de la
#           literatura de likelihood (ruido bajo-medio) vs nuestro denso
#   mrd     curriculum INVERSO: b_h decae 0.99->0.30 (masking-ratio decay,
#           direccion opuesta a la rampa que heredo la familia hi)
#
# Uso: BASE=hicf10 bash g5_launch.sh
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=${BASE:-hicf10}
DATA=/beegfs/a474r867/ecoreasoner/data
AUG2=$DATA/train_ids_b4aug2.npz
HI='{"b_l":0.30,"b_h":0.99}'

# backbone: mismos defaults del runner que la familia g4-hi*
# (whole_stage=1, curriculum=1 rampa, role_mask=1 salvo hinrcf10)
EXTRA=""
if [ "$BASE" = "hinrcf10" ]; then EXTRA="ROLE_MASK=0"; fi
run() { # tag seed extra_env...
    local tag=$1 seed=$2; shift 2
    env TAG="$tag" MASK_TYPE=random CANDIDATE_FOCUS=1.0 \
        MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=$seed $EXTRA "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

run g5-corr-s1    1 CORRECTIVE_P=0.10
run g5-corr-s2    2 CORRECTIVE_P=0.10
run g5-corrhi-s1  1 CORRECTIVE_P=0.20
run g5-numw-s1    1 LOSS_NUM_W=3.0 LOSS_NEG_W=2.5
run g5-numw-s2    2 LOSS_NUM_W=3.0 LOSS_NEG_W=2.5
run g5-corrnw-s1  1 CORRECTIVE_P=0.10 LOSS_NUM_W=3.0 LOSS_NEG_W=2.5
run g5-cos-s1     1 CURRICULUM=0 MASK_SCHEDULE=cosine MASK_SCHEDULE_ARGS='{}'
run g5-mrd-s1     1 MASK_SCHEDULE_ARGS='{"b_l":0.05,"b_h":0.99}' \
    CUR_STAGES='[[0,2000,16,0.99],[2000,6000,16,0.60],[6000,10000,16,0.30]]'

echo "G5 lanzada (base=$BASE, 8 jobs)"
