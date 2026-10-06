#!/usr/bin/env bash
# watch_unam_qa.sh — cadena UNAM-EN -> corpus QA-SFT, tras la traduccion.
#
# Fases (idempotentes, seguras tras reinicio):
#   0. espera data/UNAM_EN_MT_DONE.flag (lo escribe watch_mtolmo.sh)
#   1. merge: unam_en_to_jsonl.py -> data/unam_en.jsonl   (remoto)
#   2. flota: relanza array qa_gen_mt (0-15) mientras haya pids sin dialogo
#   3. finalize: sbatch unam_qa_finalize.slurm (filter+concat+pretok CPU)
#   4. flag UNAM_QA_SFT_READY.flag con counts -> sale
#
# Uso:    nohup bash scripts/watch_unam_qa.sh > /dev/null 2>&1 &
# Flags:  data/UNAM_QA_SFT_READY.flag   Log: data/watch_unam_qa.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
MT_FLAG=/home/reumanlab/ecoreasoner/data/UNAM_EN_MT_DONE.flag
DONE_FLAG=/home/reumanlab/ecoreasoner/data/UNAM_QA_SFT_READY.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_unam_qa.log
INTERVAL=${INTERVAL:-600}
NSHARDS=${NSHARDS:-16}
INDOCS=$BASE/data/unam_en.jsonl
QADIR=$BASE/data/qa_mt_unam
OUT_NPZ=$BASE/data/sft_ids_v3_unam.npz

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

qa_pending(){
  # pids de entrada sin registro en qa_mt_shard*.jsonl (resume por pid)
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && python3 - <<'PY'
import glob, json, os
try:
    inp = {json.loads(l)['pid'] for l in open('$INDOCS', encoding='utf-8') if l.strip()}
except Exception:
    print(-1); raise SystemExit
done = set()
for f in glob.glob('$QADIR/qa_mt_shard*.jsonl'):
    for l in open(f, encoding='utf-8'):
        try: done.add(json.loads(l)['pid'])
        except Exception: pass
print(len(inp - done))
PY" 2>/dev/null | tail -1
}

gen_running(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "squeue -u a474r867 -h -n qagen_mt | wc -l" 2>/dev/null
}

log "watcher unam-qa start (esperando MT flag)"

# --- fase 0: esperar traduccion ---
while [ ! -f "$MT_FLAG" ]; do sleep "$INTERVAL"; done
log "MT flag detectado -> merge + QA-gen"

# --- fase 1: merge (idempotente) ---
if ssh -o BatchMode=yes "$REMOTE" "[ ! -s $INDOCS ]" 2>/dev/null; then
  ssh -o ConnectTimeout=60 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && python3 scripts/unam_en_to_jsonl.py --indir /beegfs/a474r867/unam_md_en --out $INDOCS" \
    2>/dev/null | tail -2 | tee -a "$LOG"
else
  log "unam_en.jsonl ya existe — merge saltado"
fi

# --- fase 2: flota QA hasta 0 pendientes ---
while :; do
  pend=$(qa_pending); run=$(gen_running)
  pend=${pend:-?}; run=${run:-0}
  log "qa pendientes=$pend tasks_vivas=$run"
  if [ "$pend" = "0" ]; then break; fi
  if [ "$run" = "0" ] && [ "$pend" != "?" ]; then
    log "flota qagen muerta con pendientes -> relanzando array 0-$((NSHARDS-1))"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && sbatch --array=0-$((NSHARDS-1)) --export=ALL,INDIR=$INDOCS,OUTDIR=$QADIR,NSHARDS=$NSHARDS,K=4 scripts/qa_gen_mt.slurm" \
      2>/dev/null | tail -1 | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done

# --- fase 3: finalize (filter + concat + pretok) ---
if ssh -o BatchMode=yes "$REMOTE" "[ ! -s $OUT_NPZ ]" 2>/dev/null; then
  log "lanzando unam_qa_finalize"
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && sbatch --export=ALL,IN_QA=$QADIR,OUT_NPZ=$OUT_NPZ scripts/unam_qa_finalize.slurm" \
    2>/dev/null | tail -1 | tee -a "$LOG"
fi
while :; do
  ok=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "[ -s $OUT_NPZ ] && [ -s $OUT_NPZ.meta.json ] && echo 1 || echo 0" 2>/dev/null | tail -1)
  [ "$ok" = "1" ] && break
  sleep "$INTERVAL"
done

meta=$(ssh -o BatchMode=yes "$REMOTE" "cat $OUT_NPZ.meta.json" 2>/dev/null | tr -d '\n ')
log "NPZ listo: $meta"
date > "$DONE_FLAG"
echo "npz=$OUT_NPZ $meta" >> "$DONE_FLAG"
log "UNAM_QA_SFT_READY — lanzamiento de SFT-v3 queda como decision"
