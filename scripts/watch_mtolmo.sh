#!/usr/bin/env bash
# watch_mtolmo.sh — vigia del array mt_olmo (UNAM-EN curada_v3 -> unam_md_en).
#
# Loop cada INTERVAL (ssh kuhpc):
#   - pendientes>0 y 0 tasks mtolmo vivas -> relanza array 0-15
#     (resume por .en.md: seguro tras cada ventana sixhour).
#   - pendientes==0 -> flag UNAM_EN_MT_DONE.flag y sale.
#     El merge a unam_en.jsonl y la decision de integracion
#     (dosis post-1.3M o QA->SFT) quedan como paso aparte.
#
# Uso:  nohup bash scripts/watch_mtolmo.sh > /dev/null 2>&1 &
# Flag: data/UNAM_EN_MT_DONE.flag   Log: data/watch_mtolmo.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867
IN=$BASE/tesis_unam/curada_v3/science
OUTD=$BASE/unam_md_en
FLAG=/home/reumanlab/ecoreasoner/data/UNAM_EN_MT_DONE.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_mtolmo.log
INTERVAL=${INTERVAL:-600}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

pending_count(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" '
    n=0; for f in '"$IN"'/*.md; do
      b=$(basename "$f" .md)
      [ -s "'"$OUTD"'/$b.en.md" ] || n=$((n+1))
    done; echo $n' 2>/dev/null
}

running_tasks(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "squeue -u a474r867 -h -n mtolmo | wc -l" 2>/dev/null
}

resubmit(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE/ecoreasoner && sbatch --array=0-15 --export=ALL,INDIR=$IN,OUTDIR=$OUTD,NSHARDS=16 scripts/mt_olmo.slurm" \
    2>/dev/null | tail -1
}

log "watcher mtolmo start (in=$IN out=$OUTD)"
while :; do
  pend=$(pending_count); run=$(running_tasks)
  pend=${pend:-?}; run=${run:-0}
  log "pendientes=$pend tasks_vivas=$run"
  if [ "$pend" = "0" ]; then
    log "TRADUCCION COMPLETA — flag escrita (merge a unam_en.jsonl = decision aparte)"
    date > "$FLAG"; exit 0
  fi
  if [ "$run" = "0" ] && [ "$pend" != "?" ]; then
    log "flota muerta con pendientes -> relanzando array 0-15"
    resubmit | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
