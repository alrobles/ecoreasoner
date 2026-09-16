#!/usr/bin/env bash
# watch_v4_done.sh — vigía del retrain_v4 -> dispara eval holdout limpio.
#
# Loop cada ~10min (ssh kuhpc):
#   - flag ausente y 0 jobs mdlm-retrain-v4 -> relanza el retrain
#     (AUTO_RESUBMIT=1 cubre ventanas sixhour; esto cubre falls duros).
#   - training_complete.flag presente -> sbatch g_eval_holdout con
#     RUN=retrain_v4 (symlink runs/retrain_v4 -> outputs/retrain_v4),
#     MODEL_OV con la arquitectura MoE real, PAIRS=holdout_clean.
#     Escribe flag local y sale.
# Flag: /home/reumanlab/ecoreasoner/data/V4_EVAL_SUBMITTED.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_v4_done.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=$BASE/outputs/retrain_v4
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/V4_EVAL_SUBMITTED.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_v4_done.log
INTERVAL=600
MODEL_OV='{"hidden":768,"layers":12,"heads":12,"n_experts":8,"k":1}'

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher v4 start"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "if [ -f $OUT/training_complete.flag ]; then echo DONE; \
     else squeue -u a474r867 -n mdlm-retrain-v4 -h | wc -l; fi" \
    2>/dev/null | tail -1)
  log "estado=$st"
  if [ "$st" = "DONE" ]; then
    log "RETRAIN COMPLETO — disparando eval holdout_clean"
    break
  fi
  if [ "$st" = "0" ]; then
    log "sin job de retrain y sin flag -> relanzando"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && sbatch --parsable --export=TARGET_STEPS=11000,AUTO_RESUBMIT=1 \
       scripts/mdlm_retrain_v4.slurm" 2>/dev/null | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done

# === flag caido: eval ===
ssh -o BatchMode=yes "$REMOTE" \
  "ln -sfn $OUT $BASE/runs/retrain_v4"
JID=$(ssh -o BatchMode=yes "$REMOTE" \
  "cd $BASE && sbatch --parsable \
     --export=ALL,RUN=retrain_v4,PAIRS=$BASE/runs/pairs_hard_v4_holdout_clean,MODEL_OV='$MODEL_OV' \
     scripts/g_eval_holdout.slurm" 2>/dev/null | tr -d '[:space:]')
log "eval job=$JID (out -> runs/retrain_v4/battery_logicdiff_v4holdout)"

date > "$FLAG_LOCAL"
echo "eval_jid=$JID" >> "$FLAG_LOCAL"
log "FLAG escrita: $FLAG_LOCAL — fin del watcher v4"
