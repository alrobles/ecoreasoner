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
