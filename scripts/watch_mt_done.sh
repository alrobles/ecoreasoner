#!/usr/bin/env bash
# watch_mt_done.sh — vigía del batch UNAM-EN -> dispara pipeline EcoReasoner.
#
# Loop cada ~5min (ssh kuhpc):
#   - pendientes>0 y 0 tasks mt* vivas -> relanza arrays (ventanas sixhour).
#   - pendientes==0 -> BANDERA: merge beegfs<->local, sbatch corpus_v4en_build,
#     retrain_v4 encadenado por dependencia, escribe flag, sale.
# Flag: /home/reumanlab/ecoreasoner/data/UNAM_EN_DONE.flag
# Log:  /home/reumanlab/ecoreasoner/data/watch_mt_done.log
set -uo pipefail

REMOTE="kuhpc"
BIN=/beegfs/a474r867/unam_mt_in
BOUT=/beegfs/a474r867/unam_md_en
BSCR=/beegfs/a474r867/ecoreasoner/scripts
LDIR=/home/reumanlab/tesis_unam_scraper/curada_v2/md_en
FLAG=/home/reumanlab/ecoreasoner/data/UNAM_EN_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_mt_done.log
INTERVAL=300

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

pending_count(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" '
    n=0; for f in '"$BIN"'/*.md; do
      b=$(basename "$f" .md)
      [ -s "'"$BOUT"'/$b.en.md" ] || n=$((n+1))
    done; echo $n' 2>/dev/null
}

running_tasks(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "squeue -u a474r867 -h | grep -cE 'mtvllm|mt7b'" 2>/dev/null
}

resubmit(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" "cd /beegfs/a474r867/ecoreasoner && \
    NSHARDS=44 SHARD_BASE=0 MAXDOCS=64 PYLIBS=pylibs_vllm2 sbatch --gres=gpu:pro6000:1 --array=0-23 scripts/mt_vllm.slurm; \
    NSHARDS=44 SHARD_BASE=24 DTYPE=fp16 TOKBUDGET=6000 sbatch --gres=gpu:q6000:1 --array=0-19 scripts/mt_batch.slurm" \
    2>/dev/null | grep -o "job [0-9]*"
}

log "watcher start"
while :; do
  pend=$(pending_count); run=$(running_tasks)
  pend=${pend:-?}; run=${run:-0}
  log "pendientes=$pend tasks_vivas=$run"
  if [ "$pend" = "0" ]; then
    log "TRADUCCIÓN COMPLETA — disparando pipeline"
    break
  fi
  if [ "$run" = "0" ] && [ "$pend" != "?" ]; then
    log "flota muerta con pendientes -> relanzando"
    resubmit | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done

# === BANDERA CAÍDA: merge + corpus + retrain ===
log "merge beegfs -> local"
rsync -a "$REMOTE:$BOUT/" "$LDIR/" >> "$LOG" 2>&1
log "merge local -> beegfs (docs OR/locales que faltan arriba)"
rsync -a "$LDIR"/ "$REMOTE:$BOUT/" >> "$LOG" 2>&1

log "submit corpus_v4en_build"
BUILD_JID=$(ssh -o BatchMode=yes "$REMOTE" \
  "sbatch --parsable $BSCR/corpus_v4en_build.slurm" 2>/dev/null | tr -d '[:space:]')
log "build job=$BUILD_JID"

log "submit retrain_v4 (dependencia afterok:$BUILD_JID)"
RT_JID=$(ssh -o BatchMode=yes "$REMOTE" \
  "sbatch --parsable --dependency=afterok:$BUILD_JID \
    --export=TARGET_STEPS=11000,AUTO_RESUBMIT=1 \
    $BSCR/mdlm_retrain_v4.slurm" 2>/dev/null | tr -d '[:space:]')
log "retrain_v4 job=$RT_JID (espera a que build termine; auto-relanza olas)"

date > "$FLAG"
echo "build_jid=$BUILD_JID retrain_jid=$RT_JID" >> "$FLAG"
log "FLAG escrita: $FLAG — fin del watcher"
