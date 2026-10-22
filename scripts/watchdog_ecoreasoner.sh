#!/usr/bin/env bash
# watchdog_ecoreasoner.sh — mantiene vivos los watchers locales de ecoreasoner.
#
# Para cada watcher: si el proceso no existe y su flag de DONE no existe,
# lo relanza con nohup. Idempotente — seguro en cron cada N min y @reboot.
#
#   watcher                    done-flag (si existe, ya no relanza)
#   watch_v5_battery.sh        (sin flag — curva vive hasta g1.3M)
#   watch_mtolmo.sh            data/UNAM_EN_MT_DONE.flag
#   watch_unam_qa.sh           data/UNAM_QA_SFT_READY.flag
#
# Cron:  */15 * * * * /home/reumanlab/ecoreasoner/scripts/watchdog_ecoreasoner.sh
#        @reboot   /home/reumanlab/ecoreasoner/scripts/watchdog_ecoreasoner.sh
# Log:   data/watchdog.log
set -uo pipefail

DIR=/home/reumanlab/ecoreasoner
LOG=$DIR/data/watchdog.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

guard(){
  local script=$1 flag=$2
  if pgrep -f "$script" >/dev/null; then return; fi
  if [ -n "$flag" ] && [ -f "$DIR/data/$flag" ]; then return; fi
  log "watcher $script muerto y sin flag ${flag:-NONE} -> relanzando"
  cd "$DIR" && nohup bash "scripts/$script" > /dev/null 2>&1 &
}

guard watch_v5_battery.sh ""
guard watch_v6_battery.sh ""
guard watch_mtolmo.sh    "UNAM_EN_MT_DONE.flag"
guard watch_unam_qa.sh   "UNAM_QA_SFT_READY.flag"
guard watch_sft_v3.sh    "SFT_V3_DONE.flag"
guard watch_sft_v4.sh    "SFT_V4_DONE.flag"
guard watch_mt_v4.sh     "SFT_V4_MT_READY.flag"

# bootstrap sft-v4 (aprobado 10-22): al caer SFT_V3_DONE, sembrar init con el
# ckpt terminal de sft-v3 y arrancar el watcher v4 (TARGET_GN=16000 ~= 2ep).
if [ -f "$DIR/data/SFT_V3_DONE.flag" ] && [ ! -f "$DIR/data/sft_v4_init.txt" ] \
   && [ ! -f "$DIR/data/SFT_V4_DONE.flag" ]; then
  V3LAST=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
    "ls -d /beegfs/a474r867/ecoreasoner/runs/sft-v3/checkpoint-g*/model.pt 2>/dev/null | grep -v incomplete | sort -V | tail -1 | xargs -r dirname" 2>/dev/null)
  if [ -n "$V3LAST" ]; then
    echo "$V3LAST" > "$DIR/data/sft_v4_init.txt"
    log "bootstrap sft-v4: init=$V3LAST (TARGET_GN=16000)"
    cd "$DIR" && TARGET_GN=16000 nohup bash scripts/watch_sft_v4.sh > /dev/null 2>&1 &
  fi
fi

# Guard mainline: la cadena AUTO_RESUBMIT puede morir si el sbatch dentro de
# finalize() falla (visto 10-14: ola TIMEOUT, sbatch silencioso -> sin job
# durante 2h a 8.5K pasos del terminal). Si no hay job v5-1b en cola/activo
# y no hay training_complete.flag, resubmit.
mainline_guard(){
  local name=$1 rundir=$2 slurm=$3 extra=$4 opts=${5:-}
  local r=/beegfs/a474r867/ecoreasoner
  local nj flagd
  nj=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "squeue -u a474r867 -h -n $name 2>/dev/null | wc -l" 2>/dev/null) \
    || { log "$name: ssh fallo, skip"; return; }
  [ "${nj:-0}" -gt 0 ] && return
  flagd=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "[ -f $r/runs/$rundir/training_complete.flag ] && echo 1 || echo 0" 2>/dev/null)
  [ "$flagd" = "1" ] && return
  # No resubmitir si state.json se escribio hace <40min: la ola puede estar
  # a medio transicionar (sbatch ya emitido, job en PD bajo otro nombre).
  # Y si nunca existio state.json, el run no arranco -> no es responsable
  # de este guard (arranque inicial es manual).
  stale=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "[ -f $r/runs/$rundir/state.json ] && [ \$(( \$(date +%s) - \$(stat -c %Y $r/runs/$rundir/state.json) )) -gt 2400 ] && echo 1 || echo 0" 2>/dev/null)
  [ "$stale" != "1" ] && return
  log "$name SIN job, SIN flag, state.json stale -> resubmit"
  ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
    "cd $r && sbatch --export=\"$extra\" $opts $slurm" >> "$LOG" 2>&1
}
mainline_guard v5-1b v5-1b scripts/v5_1b_ddp.slurm \
  "AUTO_RESUBMIT=1,WAVE_ID=v5-1b,WAVE_OUT=/beegfs/a474r867/ecoreasoner/runs/v5-1b,TARGET_STEPS=1300000,DATA_CACHE=/beegfs/a474r867/ecoreasoner/data/train_ids_v5pdb.npz,N_EXPERTS=16,SHAPE_N=3,SHAPE_G=gpu:3" \
  "--ntasks=3 --ntasks-per-node=3 --gres=gpu:3"
mainline_guard v6-2b-nv v6-2b scripts/v6_2b_nv.slurm \
  "AUTO_RESUBMIT=1,WAVE_ID=v6-2b,WAVE_OUT=/beegfs/a474r867/ecoreasoner/runs/v6-2b,TARGET_STEPS=1300000,N_EXPERTS=16,LAYERS=24,SHAPE_N=3,SHAPE_G=gpu:3" \
  "--ntasks=3 --ntasks-per-node=3 --gres=gpu:3"
