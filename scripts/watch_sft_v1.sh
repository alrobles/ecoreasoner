#!/usr/bin/env bash
# watch_sft_v1.sh — vigia del SFT sft-v1 (init v5-1b@g94k, 143647 docs).
#
# sft_mdlm.py no auto-resume y una epoca (~17955 steps a bs8) no cabe en una
# ventana sixhour => el trainer NUNCA escribe training_complete.flag.
# Este watcher encadena olas: si no hay sft-mdlm en cola y faltan steps,
# resubmit con INIT=ultimo ckpt + OFFSET=ultimo_gN (numeracion acumulativa
# via --step_offset). Al llegar a TARGET_GN cancela la ola viva y lanza
# AMBAS evals sobre el ckpt mas alto (EMA):
#   (a) eval_qa_sft.slurm   — generativa, eval_devin_hard 3000 held-out
#   (b) g_eval_holdout_v5.slurm — battery discriminacion holdout_clean L0-L3
#       (gate de regresion: el SFT no debe erosionar la senal estructural)
#
# TARGET: 2 epocas ~= 35910 steps. Se dispara en el primer ckpt >=35500.
#
# Flag local:  data/SFT_V1_DONE.flag
# Log:         data/watch_sft_v1.log
# Uso:         nohup bash scripts/watch_sft_v1.sh > /dev/null 2>&1 &
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=runs/sft-v1
DATA_CACHE=$BASE/data/sft_ids_v1.npz
TARGET_GN=${TARGET_GN:-35500}
BATT_DIR=$OUT/battery_logicdiff_v4holdout
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/SFT_V1_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_sft_v1.log
INTERVAL=${INTERVAL:-600}
EV_SENT=0; BATT_SENT=0

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher sft-v1 start (target_gN=$TARGET_GN)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g* 2>/dev/null | sed 's/.*-g//' | sort -n | tail -1); \
     sftj=\$(squeue -u a474r867 -h -n sft-mdlm | wc -l); \
     evj=\$(squeue -u a474r867 -h -n evalqa-sft | wc -l); \
     btj=\$(squeue -u a474r867 -h -n holdout-v5 | wc -l); \
     evl=\$(wc -l < $OUT/eval_devin_hard.gen.jsonl 2>/dev/null || echo 0); \
     batt=0; [ -f $BATT_DIR/dense/battery.json ] && batt=1; \
     echo \"latest=\${latest:-0} sftj=\$sftj evj=\$evj btj=\$btj evl=\$evl batt=\$batt\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  latest=$(echo "$st" | sed -n 's/.*latest=\([0-9]*\).*/\1/p')
  sftj=$(echo "$st"   | sed -n 's/.*sftj=\([0-9]*\).*/\1/p')
  evj=$(echo "$st"    | sed -n 's/.*evj=\([0-9]*\).*/\1/p')
  btj=$(echo "$st"    | sed -n 's/.*btj=\([0-9]*\).*/\1/p')
  evl=$(echo "$st"    | sed -n 's/.*evl=\([0-9]*\).*/\1/p')
  batt=$(echo "$st"   | sed -n 's/.*batt=\([0-9]*\).*/\1/p')
  latest=${latest:-0}; sftj=${sftj:-0}; evj=${evj:-0}; btj=${btj:-0}
  evl=${evl:-0}; batt=${batt:-0}

  # done = ambas evals terminadas
  if [ "$evl" -ge 3000 ] && [ "$batt" = "1" ]; then
    log "EVAL COMPLETA: gens=$evl + battery lista -> SFT_V1_DONE"
    date > "$FLAG_LOCAL"; exit 0
  fi

  if [ "$latest" -ge "$TARGET_GN" ]; then
    if [ "$sftj" -gt 0 ]; then
      log "target alcanzado (g$latest) — cancelando ola sft viva"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "scancel -u a474r867 -n sft-mdlm" 2>/dev/null
      continue
    fi
    if [ "$EV_SENT" = "0" ] && [ "$evj" = "0" ] && [ "$evl" -lt 3000 ]; then
      log "lanzando eval_qa_sft sobre checkpoint-g$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,CKPT=$BASE/$OUT scripts/eval_qa_sft.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
      EV_SENT=1
    fi
    if [ "$BATT_SENT" = "0" ] && [ "$btj" = "0" ] && [ "$batt" = "0" ]; then
      log "lanzando battery holdout_v5 sobre checkpoint-g$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,RUNDIR=$BASE/$OUT scripts/g_eval_holdout_v5.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
      BATT_SENT=1
    fi
  else
    EV_SENT=0; BATT_SENT=0
    if [ "$sftj" = "0" ]; then
      log "sin ola viva y latest=g$latest < $TARGET_GN — resubmit (offset=$latest)"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,INIT=$BASE/$OUT/checkpoint-g$latest,DATA=$DATA_CACHE,OUT=$BASE/$OUT,EPOCHS=2,OFFSET=$latest scripts/sft_mdlm.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
