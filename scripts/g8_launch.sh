#!/usr/bin/env bash
# g8_launch.sh — G8: receta campeon (hinrcf10 = random + hi-curriculum +
# nr + cf10) sobre corpus nuevo. Ablacion de corpus, no de receta:
#   c1  = corpus v3 completo (344,837 docs; incluye 40K unam ES)
#   c1e = corpus v3 solo-EN (304,585 docs; sin UNAM) — aisla dilucion ES
# Mismos seeds 91/92 para comparacion pareada con g8-c1.
# Uso: CORPUS=c1|c1e [DEP="--dependency=afterok:JOBID"] bash g8_launch.sh
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
DATA=/beegfs/a474r867/ecoreasoner/data
HI='{"b_l":0.30,"b_h":0.99}'
CORPUS=${CORPUS:-c1}
DEP=${DEP:-}
case "$CORPUS" in
  c1)  NPZ=$DATA/train_ids_c1.npz;;
  c1e) NPZ=$DATA/train_ids_c1e.npz;;
  *) echo "CORPUS desconocido: $CORPUS"; exit 1;;
esac

run() { local tag=$1 seed=$2
    env TAG="$tag" MASK_TYPE=random CANDIDATE_FOCUS=1.0 ROLE_MASK=0 \
        MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$NPZ SEED=$seed \
        sbatch --export=ALL --job-name="$tag" $DEP g0_run.slurm
}

run g8-$CORPUS-s91 91
run g8-$CORPUS-s92 92
echo "G8 corpus=$CORPUS lanzada (NPZ=$NPZ DEP='$DEP')"
