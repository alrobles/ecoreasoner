#!/usr/bin/env bash
# watch_sft_v1.sh — vigia del SFT sft-v1 (init v5-1b@g94k, 143647 docs).
#
# sft_mdlm.py no auto-resume y una epoca (~17955 steps a bs8) no cabe en una
# ventana sixhour => el trainer NUNCA escribe training_complete.flag.
# Este watcher encadena olas: si no hay sft-mdlm en cola y faltan steps,
# resubmit con INIT=ultimo ckpt + OFFSET=ultimo_gN (numeracion acumulativa
# via --step_offset). Al llegar a TARGET_GN cancela la ola viva y lanza
# eval_qa_sft.slurm sobre el ckpt mas alto (EMA).
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
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/SFT_V1_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_sft_v1.log
INTERVAL=${INTERVAL:-600}
EVAL_SENT=0

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher sft-v1 start (target_gN=$TARGET_GN)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g* 2>/dev/null | sed 's/.*-g//' | sort -n | tail -1); \
     sftj=\$(squeue -u a474r867 -h -n sft-mdlm | wc -l); \
     evj=\$(squeue -u a474r867 -h -n evalqa-sft | wc -l); \
     evl=\$(wc -l < $OUT/eval_devin_hard.gen.jsonl 2>/dev/null || echo 0); \
     echo \"latest=\${latest:-0} sftj=\$sftj evj=\$evj evl=\$evl\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  latest=$(echo "$st" | sed -n 's/.*latest=\([0-9]*\).*/\1/p')
  sftj=$(echo "$st"   | sed -n 's/.*sftj=\([0-9]*\).*/\1/p')
  evj=$(echo "$st"    | sed -n 's/.*evj=\([0-9]*\).*/\1/p')
  evl=$(echo "$st"    | sed -n 's/.*evl=\([0-9]*\).*/\1/p')
  latest=${latest:-0}; sftj=${sftj:-0}; evj=${evj:-0}; evl=${evl:-0}

  if [ "$evl" -ge 3000 ]; then
    log "EVAL COMPLETA ($evl gens) -> SFT_V1_DONE"
    date > "$FLAG_LOCAL"; exit 0
  fi

  if [ "$latest" -ge "$TARGET_GN" ]; then
    if [ "$sftj" -gt 0 ]; then
      log "target alcanzado (g$latest) — cancelando ola sft viva"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "scancel -u a474r867 -n sft-mdlm" 2>/dev/null
      continue
    fi
    if [ "$EVAL_SENT" = "0" ] && [ "$evj" = "0" ]; then
      log "lanzando eval_qa_sft sobre checkpoint-g$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,CKPT=$BASE/$OUT scripts/eval_qa_sft.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
      EVAL_SENT=1
    fi
  else
    EVAL_SENT=0
    if [ "$sftj" = "0" ]; then
      log "sin ola viva y latest=g$latest < $TARGET_GN — resubmit (offset=$latest)"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,INIT=$BASE/$OUT/checkpoint-g$latest,DATA=$DATA_CACHE,OUT=$BASE/$OUT,EPOCHS=2,OFFSET=$latest scripts/sft_mdlm.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
