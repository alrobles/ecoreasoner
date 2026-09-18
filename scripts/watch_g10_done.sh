#!/usr/bin/env bash
# watch_g10_done.sh — vigia de la generacion G10 (5 runs g10-*).
# Mismo protocolo que watch_g9_done.sh: espera flag + battery_L3.json de
# los 5 brazos (eval dev sobre ema_model.pt via EVAL_EMA=1), luego corre
# g9_fitness.py --glob 'g10-*' y escribe G10_DONE.flag.
#
# Flag: /home/reumanlab/ecoreasoner/data/G10_DONE.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_g10_done.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/G10_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_g10_done.log
INTERVAL=900

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher g10 start"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     flags=\$(ls runs/g10-*/training_complete.flag 2>/dev/null | wc -l); \
     evals=\$(ls runs/g10-*/battery_logicdiff/dense/battery_L3.json 2>/dev/null | wc -l); \
     jobs=\$(squeue -u a474r867 -h -o '%j' | grep -c '^g10-' || true); \
     dead=''; \
     for d in runs/g10-*/; do \
       t=\$(basename \$d); \
       [ -f \$d/battery_logicdiff/dense/battery_L3.json ] && continue; \
       squeue -u a474r867 -h -n \$t | grep -q . || dead=\"\$dead \$t\"; \
     done; \
     echo \"flags=\$flags evals=\$evals jobs=\$jobs dead=\$dead\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  evals=$(echo "$st" | sed -n 's/.*evals=\([0-9]*\).*/\1/p')
  if [ "${evals:-0}" = "5" ]; then
    log "G10 COMPLETA — tabla de fitness:"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && python3 scripts/g9_fitness.py --glob 'g10-*'" \
      2>/dev/null | tee -a "$LOG"
    {
      date
      echo "$st"
    } > "$FLAG_LOCAL"
    log "FLAG escrita: $FLAG_LOCAL — fin del watcher g10"
    exit 0
  fi
  sleep "$INTERVAL"
done
