#!/usr/bin/env bash
# watch_v4s_done.sh — vigia del port retrain_v4s (receta hinrcf10 -> MoE v4).
#
# Loop cada ~10min (ssh kuhpc):
#   - flag ausente y 0 jobs retrain_v4s -> relanza v4s_launch.sh
#     (AUTO_RESUBMIT cubre olas sixhour; esto cubre falls duros tipo OOM).
#   - training_complete.flag presente -> sbatch g_eval_generic.slurm con
#     PAIRS=pairs_hard_v3_eval (dev, comparabilidad leaderboard vs
#     hinrcf10 L3 0.626) -> OUTDIR battery_logicdiff_v3eval.
#     El eval holdout_clean ya lo hace g0_run.slurm al COMPLETE.
#     Escribe flag local y sale.
# Flag: /home/reumanlab/ecoreasoner/data/V4S_DEV_EVAL_SUBMITTED.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_v4s_done.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=$BASE/runs/retrain_v4s
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/V4S_DEV_EVAL_SUBMITTED.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_v4s_done.log
INTERVAL=600

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher v4s start"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "if [ -f $OUT/training_complete.flag ]; then echo DONE; \
     else squeue -u a474r867 -n retrain_v4s -h | wc -l; fi" \
    2>/dev/null | tail -1)
  log "estado=$st"
  if [ "$st" = "DONE" ]; then
    log "V4S COMPLETO — disparando eval dev v3_eval"
    jid=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE/scripts && env RUNDIR=$OUT \
        PAIRS=$BASE/runs/pairs_hard_v3_eval \
        CFG=$BASE/harness/configs/eval-moe-v4.yaml \
        OUTDIR_NAME=battery_logicdiff_v3eval MODES=dense,base \
        sbatch --export=ALL --parsable --job-name=eval-v4s-dev g_eval_generic.slurm" \
      2>/dev/null | tail -1)
    log "eval dev job=$jid (out -> runs/retrain_v4s/battery_logicdiff_v3eval)"
    {
      date
      echo "eval_dev_jid=$jid"
    } > "$FLAG_LOCAL"
    log "FLAG escrita: $FLAG_LOCAL — fin del watcher v4s"
    exit 0
  fi
  if [ "$st" = "0" ]; then
    log "sin job retrain_v4s y sin flag -> relanzando"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE/scripts && bash v4s_launch.sh" 2>/dev/null | tail -2
  fi
  sleep "$INTERVAL"
done
