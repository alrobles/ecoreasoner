#!/usr/bin/env bash
# watch_mt_v4.sh — pipeline multi-turno a escala (v4): flota qa_gen_mt
# sobre mt_passages_v4.jsonl (muestra de corpus_v5_pdb) -> qa_mt_v4/
# -> finalize -> sft_ids_v4_mt.npz. Clon de watch_unam_qa.sh fases 2-3
# (sin espera de flag de traduccion: las fuentes ya existen).
#
# Mezcla v4 (via envs del finalize): OUT_UNAM = dialogos filtrados nuevos;
# V1JSONL apunta a v1_comb (UNAM-mt filtrado + subsample 40K de v1) para
# que el npz tenga fraccion multi-turno real (~50%), no 3.5%.
#
# Uso:    nohup bash scripts/watch_mt_v4.sh > /dev/null 2>&1 &
# Flag:   data/SFT_V4_MT_READY.flag   Log: data/watch_mt_v4.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
DONE_FLAG=/home/reumanlab/ecoreasoner/data/SFT_V4_MT_READY.flag
LOG=/home/reumanlab/ecoreasoner/data/watch_mt_v4.log
INTERVAL=${INTERVAL:-600}
NSHARDS=${NSHARDS:-32}
INDOCS=$BASE/data/mt_passages_v4.jsonl
QADIR=$BASE/data/qa_mt_v4
OUT_NPZ=$BASE/data/sft_ids_v4_mt.npz
V1COMB=$BASE/data/qa_sft_v1_comb.jsonl   # unam_mt filtrado + v1 subsample

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

qa_pending(){
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && python3 - <<'PY'
import glob, json
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

log "watcher mt-v4 start (indocs=$INDOCS nshards=$NSHARDS)"

# --- preparar mezcla v1_comb (idempotente) ---
if ssh -o BatchMode=yes "$REMOTE" "[ ! -s $V1COMB ]" 2>/dev/null; then
  log "construyendo $V1COMB = qa_sft_unam_mt.jsonl + 40K subsample qa_sft_v1"
  ssh -o ConnectTimeout=30 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && cat data/qa_sft_unam_mt.jsonl > $V1COMB.tmp && \
     shuf -n 40000 data/qa_sft_v1.jsonl >> $V1COMB.tmp && \
     mv $V1COMB.tmp $V1COMB && wc -l $V1COMB" \
    2>/dev/null | tail -1 | tee -a "$LOG"
fi

# --- fase 1: flota QA hasta 0 pendientes ---
while :; do
  pend=$(qa_pending); run=$(gen_running)
  pend=${pend:-?}; run=${run:-0}
  log "qa pendientes=$pend tasks_vivas=$run"
  if [ "$pend" = "0" ]; then break; fi
  if [ "$run" = "0" ] && [ "$pend" != "?" ] && [ "$pend" != "-1" ] && [ -n "$pend" ]; then
    log "flota qagen muerta con pendientes -> relanzando array 0-$((NSHARDS-1))"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
      "cd $BASE && sbatch --array=0-$((NSHARDS-1)) --export=ALL,INDIR=$INDOCS,OUTDIR=$QADIR,NSHARDS=$NSHARDS,K=4 scripts/qa_gen_mt.slurm" \
      2>/dev/null | tail -1 | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done

# --- fase 2: finalize (filter + concat + pretok) ---
if ssh -o BatchMode=yes "$REMOTE" "[ ! -s $OUT_NPZ ]" 2>/dev/null; then
  log "lanzando finalize v4-mt"
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && sbatch --export=ALL,IN_QA=$QADIR,OUT_UNAM=$BASE/data/qa_sft_mt_v4.jsonl,OUT_ALL=$BASE/data/qa_sft_v4_mt.jsonl,V1JSONL=$V1COMB,OUT_NPZ=$OUT_NPZ scripts/unam_qa_finalize.slurm" \
    2>/dev/null | tail -1 | tee -a "$LOG"
fi
while :; do
  ok=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "[ -s $OUT_NPZ ] && [ -s $OUT_NPZ.meta.json ] && echo 1 || echo 0" 2>/dev/null | tail -1)
  [ "$ok" = "1" ] && break
  sleep "$INTERVAL"
done

meta=$(ssh -o BatchMode=yes "$REMOTE" "cat $OUT_NPZ.meta.json" 2>/dev/null | tr -d '\n ')
log "NPZ v4-mt listo: $meta"
date > "$DONE_FLAG"
echo "npz=$OUT_NPZ $meta" >> "$DONE_FLAG"
log "SFT_V4_MT_READY — lanzamiento de SFT-v4 queda como decision"
