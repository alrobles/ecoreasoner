#!/usr/bin/env bash
# g7_launch.sh — G7: consolidacion del ganador G6 (3 seeds) + probe residual.
# Uso: GENE="<env k=v>" bash g7_launch.sh
#   GENE se inyecta como env extra en los 3 seeds del campeon.
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
DATA=/beegfs/a474r867/ecoreasoner/data
AUG2=$DATA/train_ids_b4aug2.npz
HI='{"b_l":0.30,"b_h":0.99}'
GENE=${GENE:-}

run() { local tag=$1 seed=$2; shift 2
    env TAG="$tag" MASK_TYPE=random CANDIDATE_FOCUS=1.0 ROLE_MASK=0 \
        MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=$seed $GENE "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

run g7-champ-s1 61
run g7-champ-s2 62
run g7-champ-s3 63
echo "G7 lanzada (GENE='$GENE')"
