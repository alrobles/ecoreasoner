#!/usr/bin/env bash
# watch_v5_battery.sh — curva de discriminacion L0-L3 durante el pretrain
# de v5-1b. Cada INTERVAL: si el ckpt mas alto supera al ultimo evaluado
# en >= STEP_GAP y no hay holdout-v5 en cola, lanza g_eval_holdout_v5.slurm
# con OUTDIR=runs/v5-1b/battery_curve/g<N> (EMA, holdout_clean).
#
# La curva responde: ¿la discriminacion inferencial (L2/L3) emerge con
# escala sola, o hace falta ataque de datos (trazas logicas)? — insumo
# directo del frente "incremento de razonamiento".
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

log "watcher v5-battery start (gap=$STEP_GAP)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g* 2>/dev/null | sed 's/.*-g//' | sort -n | tail -1); \
     evaled=\$(ls -d $CURVE/g* 2>/dev/null | sed 's/.*-g//' | sort -n | tail -1); \
     btj=\$(squeue -u a474r867 -h -n holdout-v5 | wc -l); \
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
      "cd $BASE && sbatch --export=ALL,RUNDIR=$BASE/$OUT,OUTDIR=$BASE/$CURVE/g$latest scripts/g_eval_holdout_v5.slurm" \
      2>/dev/null | tail -1 | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
