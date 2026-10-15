#!/usr/bin/env bash
# watch_v6_battery.sh — curva de discriminacion durante el pretrain de v6-2b
# (brazo de capacidad en MI210). Misma logica que watch_v5_battery.sh:
# cada INTERVAL, si el ckpt mas alto supera al ultimo evaluado en >=STEP_GAP
# (o el run termino con ckpt sin evaluar) y no hay eval en cola, lanza
# scripts/battery_point.sh con RUN_OUT=runs/v6-2b + EVAL_CFG=eval-moe-v6.yaml.
#
# Uso: nohup bash scripts/watch_v6_battery.sh > /dev/null 2>&1 &
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=runs/v6-2b
CURVE=$OUT/battery_curve
STEP_GAP=${STEP_GAP:-25000}
INTERVAL=${INTERVAL:-3600}
LOG=/home/reumanlab/ecoreasoner/data/watch_v6_battery.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher v6-battery start (gap=$STEP_GAP, eval_cfg=eval-moe-v6.yaml)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g*/model.pt 2>/dev/null | grep -v incomplete | sed 's|.*/checkpoint-g||; s|/model.pt||' | sort -n | tail -1); \
     evaled=\$(ls -d $CURVE/g* 2>/dev/null | sed 's|.*/g||; s/-.*//' | sort -n | tail -1); \
     btj=\$(squeue -u a474r867 -h -n holdout-v5,c-eval | wc -l); \
     [ -f $OUT/training_complete.flag ] && done=1 || done=0; \
     echo \"latest=\${latest:-0} evaled=\${evaled:-0} btj=\$btj done=\$done\"" \
    2>/dev/null | tail -1)
  latest=$(echo "$st" | sed -n 's/.*latest=\([0-9]*\).*/\1/p')
  evaled=$(echo "$st" | sed -n 's/.*evaled=\([0-9]*\).*/\1/p')
  btj=$(echo "$st"    | sed -n 's/.*btj=\([0-9]*\).*/\1/p')
  donef=$(echo "$st"  | sed -n 's/.*done=\([0-9]*\).*/\1/p')
  latest=${latest:-0}; evaled=${evaled:-0}; btj=${btj:-0}; donef=${donef:-0}
  log "estado: latest=g$latest evaled=g$evaled btj=$btj done=$donef"

  if [ "$btj" = "0" ] && { [ $((latest - evaled)) -ge "$STEP_GAP" ] || { [ "$donef" = "1" ] && [ "$latest" -gt "$evaled" ]; }; }; then
    log "lanzando battery en checkpoint-g$latest (v6-2b, eval-moe-v6)"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && RUN_OUT=runs/v6-2b EVAL_CFG=eval-moe-v6.yaml bash scripts/battery_point.sh $latest" \
      2>/dev/null | tail -4 | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
