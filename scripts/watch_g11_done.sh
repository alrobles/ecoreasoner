#!/usr/bin/env bash
# watch_g11_done.sh — vigia de la generacion G11 (3 runs g11-*).
# Mismo protocolo que watch_g10_done.sh: espera flag + battery_L3.json de
# los 3 brazos (eval dev sobre ema_model.pt via EVAL_EMA=1), luego corre
# g9_fitness.py --glob 'g11-*' y escribe G11_DONE.flag.
#
# Flag: /home/reumanlab/ecoreasoner/data/G11_DONE.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_g11_done.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/G11_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_g11_done.log
INTERVAL=900
NEVALS=${NEVALS:-3}

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher g11 start (nevals=$NEVALS)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     flags=\$(ls runs/g11-*/training_complete.flag 2>/dev/null | wc -l); \
     evals=\$(ls runs/g11-*/battery_logicdiff/dense/battery_L3.json 2>/dev/null | wc -l); \
     jobs=\$(squeue -u a474r867 -h -o '%j' | grep -c '^g11-' || true); \
     dead=''; \
     for d in runs/g11-*/; do \
       t=\$(basename \$d); \
       [ -f \$d/battery_logicdiff/dense/battery_L3.json ] && continue; \
       squeue -u a474r867 -h -n \$t | grep -q . || dead=\"\$dead \$t\"; \
     done; \
     echo \"flags=\$flags evals=\$evals jobs=\$jobs dead=\$dead\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  evals=$(echo "$st" | sed -n 's/.*evals=\([0-9]*\).*/\1/p')
  if [ "${evals:-0}" = "$NEVALS" ]; then
    log "G11 COMPLETA — tabla de fitness:"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && python3 scripts/g9_fitness.py --glob 'g11-*'" \
      2>/dev/null | tee -a "$LOG"
    {
      date
      echo "$st"
    } > "$FLAG_LOCAL"
    log "FLAG escrita: $FLAG_LOCAL — fin del watcher g11"
    exit 0
  fi
  sleep "$INTERVAL"
done
