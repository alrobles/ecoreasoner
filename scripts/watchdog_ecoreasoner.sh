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
guard watch_mtolmo.sh    "UNAM_EN_MT_DONE.flag"
guard watch_unam_qa.sh   "UNAM_QA_SFT_READY.flag"

# Guard mainline: la cadena AUTO_RESUBMIT puede morir si el sbatch dentro de
# finalize() falla (visto 10-14: ola TIMEOUT, sbatch silencioso -> sin job
# durante 2h a 8.5K pasos del terminal). Si no hay job v5-1b en cola/activo
# y no hay training_complete.flag, resubmit.
mainline_guard(){
  local r=/beegfs/a474r867/ecoreasoner
  local nj flagd
  nj=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "squeue -u a474r867 -h -n v5-1b 2>/dev/null | wc -l" 2>/dev/null) \
    || { log "mainline: ssh fallo, skip"; return; }
  [ "${nj:-0}" -gt 0 ] && return
  flagd=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "[ -f $r/runs/v5-1b/training_complete.flag ] && echo 1 || echo 0" 2>/dev/null)
  [ "$flagd" = "1" ] && return
  # No resubmitir si state.json se escribio hace <40min: la ola puede estar
  # a medio transicionar (sbatch ya emitido, job en PD bajo otro nombre).
  stale=$(ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
       "[ -f $r/runs/v5-1b/state.json ] && [ \$(( \$(date +%s) - \$(stat -c %Y $r/runs/v5-1b/state.json) )) -gt 2400 ] && echo 1 || echo 0" 2>/dev/null)
  [ "$stale" != "1" ] && return
  log "mainline SIN job, SIN flag, state.json stale -> resubmit v5-1b"
  ssh -o ConnectTimeout=15 -o BatchMode=yes kuhpc \
    "cd $r && sbatch --ntasks=3 --ntasks-per-node=3 --gres=gpu:3 \
     --export=AUTO_RESUBMIT=1,WAVE_ID=v5-1b,WAVE_OUT=$r/runs/v5-1b,TARGET_STEPS=1300000,DATA_CACHE=$r/data/train_ids_v5pdb.npz,N_EXPERTS=16,SHAPE_N=3,SHAPE_G=gpu:3 \
     scripts/v5_1b_ddp.slurm" >> "$LOG" 2>&1
}
mainline_guard
