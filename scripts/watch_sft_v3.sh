#!/usr/bin/env bash
# watch_sft_v3.sh — vigia del SFT sft-v3 (backbone v5-1b maduro, gate >=g700K).
#
# Igual que watch_sft_v1.sh: sft_mdlm.py no auto-resume y una epoca
# (~17955 steps a bs8 sobre 143647 docs) no cabe en una ventana sixhour =>
# encadena olas con INIT=ultimo ckpt + OFFSET=ultimo gN acumulativo.
# Primera ola: INIT se toma de data/sft_v3_init.txt (escrito al lanzar el
# watcher con SFT_INIT=<dir ckpt EMA del pretrain>).
#
# Al llegar a TARGET_GN cancela la ola viva y lanza:
#   (a) eval_qa_sft.slurm  — generativa, eval_devin_hard 3000 held-out
#   (b) c_contrast.slurm   — REGRESION con instrumentos corregidos:
#       PAIRS=pairs_l3_ctxnec MODES="consistency remask"
#       (el SFT no debe erosionar el binding a contexto; comparar contra
#        la curva mainline del ckpt de init)
#
# TARGET: 2 epocas ~= 35910 steps. Se dispara en el primer ckpt >=35500.
#
# Uso:    SFT_INIT=$BASE/runs/v5-1b/checkpoint-gNNNNNN \
#           nohup bash scripts/watch_sft_v3.sh > /dev/null 2>&1 &
# Flag:   data/SFT_V3_DONE.flag   Log: data/watch_sft_v3.log
set -uo pipefail

REMOTE="kuhpc"
BASE=/beegfs/a474r867/ecoreasoner
OUT=runs/sft-v3
DATA_CACHE=$BASE/data/sft_ids_v3_unam.npz
TARGET_GN=${TARGET_GN:-35500}
REG_DIR=$BASE/runs/contrast/pairs_l3_ctxnec_sft-v3
FLAG_LOCAL=/home/reumanlab/ecoreasoner/data/SFT_V3_DONE.flag
INIT_FILE=/home/reumanlab/ecoreasoner/data/sft_v3_init.txt
LOG=/home/reumanlab/ecoreasoner/data/watch_sft_v3.log
INTERVAL=${INTERVAL:-600}
EV_SENT=0; BATT_SENT=0

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# persistir el init de la primera ola (sobrevive reinicios del watcher)
if [ -n "${SFT_INIT:-}" ]; then echo "$SFT_INIT" > "$INIT_FILE"; fi
SFT_INIT=$(cat "$INIT_FILE" 2>/dev/null || true)
if [ -z "$SFT_INIT" ]; then
  log "ERROR: sin SFT_INIT (ni env ni $INIT_FILE) — salgo"; exit 1
fi

log "watcher sft-v3 start (target_gN=$TARGET_GN init0=$SFT_INIT)"
while :; do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
    "cd $BASE && \
     latest=\$(ls -d $OUT/checkpoint-g* 2>/dev/null | sed 's/.*-g//' | sort -n | tail -1); \
     sftj=\$(squeue -u a474r867 -h -n sft-v3 | wc -l); \
     evj=\$(squeue -u a474r867 -h -n evalqa-sft | wc -l); \
     btj=\$(squeue -u a474r867 -h -n c-eval | wc -l); \
     evl=\$(wc -l < $OUT/eval_devin_hard.gen.jsonl 2>/dev/null || echo 0); \
     batt=0; [ -f $REG_DIR/logicdiff_summary.json ] && batt=1; \
     echo \"latest=\${latest:-0} sftj=\$sftj evj=\$evj btj=\$btj evl=\$evl batt=\$batt\"" \
    2>/dev/null | tail -1)
  log "estado: $st"
  latest=$(echo "$st" | sed -n 's/.*latest=\([0-9]*\).*/\1/p')
  sftj=$(echo "$st"   | sed -n 's/.*sftj=\([0-9]*\).*/\1/p')
  evj=$(echo "$st"    | sed -n 's/.*evj=\([0-9]*\).*/\1/p')
  btj=$(echo "$st"    | sed -n 's/.*btj=\([0-9]*\).*/\1/p')
  evl=$(echo "$st"    | sed -n 's/.*evl=\([0-9]*\).*/\1/p')
  batt=$(echo "$st"   | sed -n 's/.*batt=\([0-9]*\).*/\1/p')
  latest=${latest:-0}; sftj=${sftj:-0}; evj=${evj:-0}; btj=${btj:-0}
  evl=${evl:-0}; batt=${batt:-0}

  # done = ambas evals terminadas
  if [ "$evl" -ge 3000 ] && [ "$batt" = "1" ]; then
    log "EVAL COMPLETA: gens=$evl + regresion ctxnec lista -> SFT_V3_DONE"
    date > "$FLAG_LOCAL"; exit 0
  fi

  if [ "$latest" -ge "$TARGET_GN" ]; then
    if [ "$sftj" -gt 0 ]; then
      log "target alcanzado (g$latest) — cancelando ola sft viva"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "scancel -u a474r867 -n sft-v3" 2>/dev/null
      continue
    fi
    if [ "$EV_SENT" = "0" ] && [ "$evj" = "0" ] && [ "$evl" -lt 3000 ]; then
      log "lanzando eval_qa_sft sobre checkpoint-g$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --export=ALL,CKPT=$BASE/$OUT scripts/eval_qa_sft.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
      EV_SENT=1
    fi
    if [ "$BATT_SENT" = "0" ] && [ "$btj" = "0" ] && [ "$batt" = "0" ]; then
      log "lanzando regresion ctxnec (consistency+remask) sobre checkpoint-g$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --partition=sixhour --export=\"ALL,RUNDIR=$BASE/$OUT,PAIRS=$BASE/runs/pairs_l3_ctxnec,MODES=consistency remask,OUTDIR=$REG_DIR\" scripts/c_contrast.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
      BATT_SENT=1
    fi
  else
    EV_SENT=0; BATT_SENT=0
    if [ "$sftj" = "0" ]; then
      if [ "$latest" = "0" ]; then INIT_CKPT="$SFT_INIT"; else INIT_CKPT="$BASE/$OUT/checkpoint-g$latest"; fi
      log "sin ola viva y latest=g$latest < $TARGET_GN — resubmit init=$INIT_CKPT offset=$latest"
      ssh -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE" \
        "cd $BASE && sbatch --job-name=sft-v3 --export=ALL,INIT=$INIT_CKPT,DATA=$DATA_CACHE,OUT=$BASE/$OUT,EPOCHS=2,OFFSET=$latest scripts/sft_mdlm_v5.slurm" \
        2>/dev/null | tail -1 | tee -a "$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
