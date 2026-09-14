#!/usr/bin/env bash
# g4_launch.sh — generacion G4: mutantes del lider dev g3-hi
# (MASK_TYPE=random + CF0.7 + aug2 + mask_p DENSO 0.30-0.99, FIT 0.492
# vs campeon 0.446 sd 0.012). Cruza densidad con los ejes que movieron
# negacion (ep2=steps, ws0) + replicas para sigma + probes de mask_p.
# Protocolo: configs candidatas a campeon llevan 2 seeds (-s1/-s2).
# Uso: bash g4_launch.sh
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
DATA=/beegfs/a474r867/ecoreasoner/data
AUG2=$DATA/train_ids_b4aug2.npz
AUG2L=$DATA/train_ids_b4aug2logic.npz
HI='{"b_l":0.30,"b_h":0.99}'

# --- replicas del lider (sigma propia; g3-hi tiene solo 1 seed) ------
TAG=g4-hi-s2    MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=2 sbatch --export=ALL --job-name=g4-hi-s2   g0_run.slurm
TAG=g4-hi-s3    MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=3 sbatch --export=ALL --job-name=g4-hi-s3   g0_run.slurm
# --- cruce con ejes que movieron negacion: steps y whole_stage -------
TAG=g4-hiep2-s1 MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 TARGET_STEPS=20000 SEED=1 sbatch --export=ALL --job-name=g4-hiep2-s1 g0_run.slurm
TAG=g4-hiep2-s2 MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 TARGET_STEPS=20000 SEED=2 sbatch --export=ALL --job-name=g4-hiep2-s2 g0_run.slurm
TAG=g4-hiws0-s1 MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 WHOLE_STAGE=0 SEED=1 sbatch --export=ALL --job-name=g4-hiws0-s1  g0_run.slurm
TAG=g4-hiws0-s2 MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 WHOLE_STAGE=0 SEED=2 sbatch --export=ALL --job-name=g4-hiws0-s2  g0_run.slurm
# --- probes de densidad mask_p --------------------------------------
TAG=g4-himax    MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS='{"b_l":0.45,"b_h":0.99}' DATA_CACHE=$AUG2 SEED=1 sbatch --export=ALL --job-name=g4-himax   g0_run.slurm
TAG=g4-himid    MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS='{"b_l":0.20,"b_h":0.90}' DATA_CACHE=$AUG2 SEED=1 sbatch --export=ALL --job-name=g4-himid   g0_run.slurm
# --- interacciones pendientes bajo densidad -------------------------
TAG=g4-hicf05   MASK_TYPE=random CANDIDATE_FOCUS=0.50 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 SEED=1 sbatch --export=ALL --job-name=g4-hicf05  g0_run.slurm
TAG=g4-hilogic  MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2L SEED=1 sbatch --export=ALL --job-name=g4-hilogic g0_run.slurm
TAG=g4-hilr1    MASK_TYPE=random CANDIDATE_FOCUS=0.70 MASK_SCHEDULE_ARGS="$HI" DATA_CACHE=$AUG2 LR=1e-4 SEED=1 sbatch --export=ALL --job-name=g4-hilr1   g0_run.slurm

# --- watcher ---------------------------------------------------------
GEN=g4 TAGS="g4-hi-s2 g4-hi-s3 g4-hiep2-s1 g4-hiep2-s2 g4-hiws0-s1 g4-hiws0-s2 g4-himax g4-himid g4-hicf05 g4-hilogic g4-hilr1" \
    sbatch --export=ALL --job-name=g4-watch g0_watch.slurm

echo "G4 lanzada (11 jobs + watcher)"
