#!/usr/bin/env bash
# watch_v5_battery.sh — curva de discriminacion durante el pretrain de v5-1b.
# Cada INTERVAL: si el ckpt mas alto COMPLETO (model.pt ya renombrado) supera
# al ultimo evaluado en >= STEP_GAP y no hay eval en cola, lanza via
# scripts/battery_point.sh (snapshot hardlink + 3 sbatch con CKPT_DIR pineado):
#   (a) g_eval_holdout_v5.slurm  -> $CURVE/g<N>/dense          (L0-L3 dense)
#   (b) c_contrast.slurm inf     -> $CURVE/g<N>/cons_inf       (consistency+profile)
#   (c) c_contrast.slurm ctxnec  -> $CURVE/g<N>/cons_ctxnec    (binding probe)
# La curva (b)+(c) responde la pregunta abierta #1 del progress report:
# si la exposicion mueve las metricas corregidas (no solo dense).
#
# Log: data/watch_v5_battery.log
# Uso: nohup bash scripts/watch_v5_battery.sh > /dev/null 2>&1 &
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=runs/v5-1b
CURVE=$OUT/battery_curve
STEP_GAP=${STEP_GAP:-25000}
INTERVAL=${INTERVAL:-3600}
LOG=/home/reumanlab/ecoreasoner/data/watch_v5_battery.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher v5-battery start (gap=$STEP_GAP, metrics=dense+consistency+ctxnec)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g*/model.pt 2>/dev/null | grep -v incomplete | sed 's|.*/checkpoint-g||; s|/model.pt||' | sort -n | tail -1); \
     evaled=\$(ls -d $CURVE/g* 2>/dev/null | sed 's|.*/g||; s/-.*//' | sort -n | tail -1); \
     btj=\$(squeue -u a474r867 -h -n holdout-v5,c-eval | wc -l); \
     echo \"latest=\${latest:-0} evaled=\${evaled:-0} btj=\$btj\"" \
    2>/dev/null | tail -1)
  latest=$(echo "$st" | sed -n 's/.*latest=\([0-9]*\).*/\1/p')
  evaled=$(echo "$st" | sed -n 's/.*evaled=\([0-9]*\).*/\1/p')
  btj=$(echo "$st"    | sed -n 's/.*btj=\([0-9]*\).*/\1/p')
  latest=${latest:-0}; evaled=${evaled:-0}; btj=${btj:-0}
  log "estado: latest=g$latest evaled=g$evaled btj=$btj"

  if [ "$btj" = "0" ] && [ $((latest - evaled)) -ge "$STEP_GAP" ]; then
    log "lanzando battery en checkpoint-g$latest"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && bash scripts/battery_point.sh $latest" \
      2>/dev/null | tail -4 | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
