#!/usr/bin/env bash
# watch_g9_done.sh — vigia de la generacion G9 (8 runs g9-*).
#
# Loop cada ~15min (ssh kuhpc):
#   - cuenta runs g9-* con training_complete.flag
#   - si un run no tiene flag NI job vivo en cola -> reporta (los
#     resubmit de ola se auto-gestionan; un fallo duro queda visible)
#   - cuando los 8 tienen flag **y** battery_L3.json (el eval corre
#     despues del flag) -> corre scripts/g9_fitness.py en el
#     cluster, escribe flag local y sale.
# El eval dev v3_eval ya lo hace g0_run.slurm al COMPLETE
# (battery_logicdiff/dense con PAIRS_DIR default = pairs_hard_v3_eval).
#
# Flag: /home/reumanlab/ecoreasoner/data/G9_DONE.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_g9_done.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/G9_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_g9_done.log
INTERVAL=900

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher g9 start"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     flags=\$(ls runs/g9-*/training_complete.flag 2>/dev/null | wc -l); \
     evals=\$(ls runs/g9-*/battery_logicdiff/dense/battery_L3.json 2>/dev/null | wc -l); \
     jobs=\$(squeue -u a474r867 -h -o '%j' | grep -c '^g9-' || true); \
     dead=''; \
     for d in runs/g9-*/; do \
       t=\$(basename \$d); \
       [ -f \$d/battery_logicdiff/dense/battery_L3.json ] && continue; \
       squeue -u a474r867 -h -n \$t | grep -q . || dead=\"\$dead \$t\"; \
     done; \
     echo \"flags=\$flags evals=\$evals jobs=\$jobs dead=\$dead\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  evals=$(echo "$st" | sed -n 's/.*evals=\([0-9]*\).*/\1/p')
  if [ "${evals:-0}" = "8" ]; then
    log "G9 COMPLETA — tabla de fitness:"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && python3 scripts/g9_fitness.py" 2>/dev/null | tee -a "$LOG"
    {
      date
      echo "$st"
    } > "$FLAG_LOCAL"
    log "FLAG escrita: $FLAG_LOCAL — fin del watcher g9"
    exit 0
  fi
  sleep "$INTERVAL"
done
