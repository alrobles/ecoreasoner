#!/usr/bin/env bash
# g3_launch.sh — generacion G3: mutantes del campeon holdout g2-cf07aug
# (MASK_TYPE=random + CANDIDATE_FOCUS=0.70 + corpus aug). G3 sube el corpus
# aug a b4aug2 (11,964 docs reales, 13x) y satura el pool (~13 jobs).
# Uso: bash g3_launch.sh
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
DATA=/beegfs/a474r867/ecoreasoner/data
AUG2=$DATA/train_ids_b4aug2.npz
AUG1=$DATA/train_ids_b4aug.npz
NB_ROLE='{"connectives":["because","despite","however","therefore","although","thus","since","while","whereas","yet","but","so","if","not","no","never","cannot","without","fails"],"relation_verbs":["increases","reduces","decreases","inhibits","promotes","correlates","depends","drives","affects","influences","regulates","enhances","suppresses","causes","leads","results","associated","linked","modulates","triggers"],"species_keywords":["panthera","quercus","apis","daphnia","mytilus","giraffa","cervus","populus","saguaro","tigris","ilex","suber"],"number_weight":6.0,"connective_weight":3.0,"relation_weight":3.0,"entity_weight":2.0,"uppercase_bonus":0.5}'

# baseline comun de la generacion: random + CF0.7 + aug2 (campeon + dato 13x)
B="MASK_TYPE=random CANDIDATE_FOCUS=0.70"

# --- ejes principales (GPUs rapidas donde se pueda) ---------------------------
# replica del campeon con corpus nuevo = varianza de corrida + gen "mas aug"
TAG=g3-champ     MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 sbatch --export=ALL --job-name=g3-champ --gres=gpu:l40:1   g0_run.slurm
# campeon con corpus viejo (902 docs) = aisla el gen "cantidad de aug"
TAG=g3-champv1   MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG1 sbatch --export=ALL --job-name=g3-champv1 --gres=gpu:l40:1 g0_run.slurm
# barrido de candidate_focus
TAG=g3-cf085     MASK_TYPE=random CANDIDATE_FOCUS=0.85 DATA_CACHE=$AUG2 sbatch --export=ALL --job-name=g3-cf085 --gres=gpu:l40:1   g0_run.slurm
TAG=g3-cf10      MASK_TYPE=random CANDIDATE_FOCUS=1.00 DATA_CACHE=$AUG2 sbatch --export=ALL --job-name=g3-cf10  --gres=gpu:l40:1   g0_run.slurm
TAG=g3-cf05      MASK_TYPE=random CANDIDATE_FOCUS=0.50 DATA_CACHE=$AUG2 sbatch --export=ALL --job-name=g3-cf05  --gres=gpu:a40:1   g0_run.slurm
# mecanismo: whole_stage / role / numbias bajo la receta ganadora
TAG=g3-ws0       MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 WHOLE_STAGE=0    sbatch --export=ALL --job-name=g3-ws0   --gres=gpu:a40:1   g0_run.slurm
TAG=g3-nr        MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 ROLE_MASK=0      sbatch --export=ALL --job-name=g3-nr    --gres=gpu:a40:1   g0_run.slurm
TAG=g3-numb      MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 ROLE_CONFIG="$NB_ROLE" sbatch --export=ALL --job-name=g3-numb --gres=gpu:a40:1 g0_run.slurm
# densidad de masking bajo random (extremos)
TAG=g3-hi        MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 MASK_SCHEDULE_ARGS='{"b_l":0.30,"b_h":0.99}' sbatch --export=ALL --job-name=g3-hi --gres=gpu:q8000:1 g0_run.slurm
TAG=g3-lo        MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 MASK_SCHEDULE_ARGS='{"b_l":0.02,"b_h":0.50}' sbatch --export=ALL --job-name=g3-lo --gres=gpu:q8000:1 g0_run.slurm
# receta: lr / epocas / capacidad bajo la ganadora
TAG=g3-lr4       MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 LR=4e-4          sbatch --export=ALL --job-name=g3-lr4   g0_run.slurm
TAG=g3-ep2       MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 TARGET_STEPS=20000 sbatch --export=ALL --job-name=g3-ep2 --gres=gpu:pro6000:1 g0_run.slurm
TAG=g3-cap       MASK_TYPE=random CANDIDATE_FOCUS=0.70 DATA_CACHE=$AUG2 HIDDEN=768 LAYERS=8 sbatch --export=ALL --job-name=g3-cap --gres=gpu:pro6000:1 g0_run.slurm

# --- watcher ------------------------------------------------------------------
GEN=g3 TAGS="g3-champ g3-champv1 g3-cf085 g3-cf10 g3-cf05 g3-ws0 g3-nr g3-numb g3-hi g3-lo g3-lr4 g3-ep2 g3-cap" \
    sbatch --export=ALL --job-name=g3-watch g0_watch.slurm

echo "G3 lanzada (13 jobs + watcher)"
