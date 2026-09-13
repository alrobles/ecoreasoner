#!/usr/bin/env bash
# g2_launch.sh — lanza la generacion G2 (8 jobs + watcher).
# Basada en G1: g1-cf05rand (FIT 0.495, num 0.385, neg 0.372 — breakout del
# mask_type=random bajo candidate-focus) + genes complementarios aug/cf07.
# Uso: bash g2_launch.sh   (ajustar slots numbias-dependientes si numbias falla)
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
DATA=/beegfs/a474r867/ecoreasoner/data
NB_ROLE='{"connectives":["because","despite","however","therefore","although","thus","since","while","whereas","yet","but","so","if","not","no","never","cannot","without","fails"],"relation_verbs":["increases","reduces","decreases","inhibits","promotes","correlates","depends","drives","affects","influences","regulates","enhances","suppresses","causes","leads","results","associated","linked","modulates","triggers"],"species_keywords":["panthera","quercus","apis","daphnia","mytilus","giraffa","cervus","populus","saguaro","tigris","ilex","suber"],"number_weight":6.0,"connective_weight":3.0,"relation_weight":3.0,"entity_weight":2.0,"uppercase_bonus":0.5}'

# --- 6 mutantes del ganador g1-cf05rand (random + CF0.5) ----------------------
TAG=g2-randaug   MASK_TYPE=random CANDIDATE_FOCUS=0.50 DATA_CACHE=$DATA/train_ids_b4aug.npz   sbatch --export=ALL --job-name=g2-randaug   g0_run.slurm
TAG=g2-randcf07  MASK_TYPE=random CANDIDATE_FOCUS=0.70                                        sbatch --export=ALL --job-name=g2-randcf07  g0_run.slurm
TAG=g2-randnumb  MASK_TYPE=random CANDIDATE_FOCUS=0.50 ROLE_CONFIG="$NB_ROLE"                 sbatch --export=ALL --job-name=g2-randnumb  g0_run.slurm
TAG=g2-randaug5x MASK_TYPE=random CANDIDATE_FOCUS=0.50 DATA_CACHE=$DATA/train_ids_aug5xmix.npz sbatch --export=ALL --job-name=g2-randaug5x g0_run.slurm
TAG=g2-randws0   MASK_TYPE=random CANDIDATE_FOCUS=0.50 WHOLE_STAGE=0                          sbatch --export=ALL --job-name=g2-randws0   g0_run.slurm
TAG=g2-randnr    MASK_TYPE=random CANDIDATE_FOCUS=0.50 ROLE_MASK=0                            sbatch --export=ALL --job-name=g2-randnr    g0_run.slurm

# --- 2 probes frescos --------------------------------------------------------
# mask_p alto bajo random (denoising denso ~ curriculum endpoint sin span)
TAG=g2-randhi    MASK_TYPE=random CANDIDATE_FOCUS=0.50 MASK_SCHEDULE_ARGS='{"b_l":0.30,"b_h":0.99}' sbatch --export=ALL --job-name=g2-randhi g0_run.slurm
# intensificacion total: random + CF0.7 + aug (los 3 ganadores)
TAG=g2-cf07aug   MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$DATA/train_ids_b4aug.npz   sbatch --export=ALL --job-name=g2-cf07aug   g0_run.slurm

# --- watcher -----------------------------------------------------------------
GEN=g2 TAGS="g2-randaug g2-randcf07 g2-randnumb g2-randaug5x g2-randws0 g2-randnr g2-randhi g2-cf07aug" \
    sbatch --export=ALL --job-name=g2-watch g0_watch.slurm

echo "G2 lanzada (8 jobs + watcher)"
