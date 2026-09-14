#!/usr/bin/env bash
# gen_wait.sh — espera a que una generacion tenga verdicts (o muerte
# declarada por gen_overseer.py --write-dead). Uso:
#   bash gen_wait.sh "tag1 tag2 ..." [poll_s] [max_h]
# Imprime progreso cada poll; sale 0 cuando todos tienen verdict.
set -u
RUNS=/beegfs/a474r867/ecoreasoner/runs
SCR=/beegfs/a474r867/ecoreasoner/scripts
TAGS=$1
POLL=${2:-120}
MAXH=${3:-10}
t0=$(date +%s)
while true; do
    missing=""
    for t in $TAGS; do
        [ -f "$RUNS/$t/battery_verdict_dense.json" ] || missing="$missing $t"
    done
    now=$(date +%s); el=$(( (now - t0) / 60 ))
    if [ -z "$missing" ]; then
        echo "[gen_wait] TODOS los verdicts listos (${el}min)"; exit 0
    fi
    echo "[gen_wait] ${el}min — faltan:$missing"
    # overseer: marca como muertas las corridas sin job vivo ni log fresco
    python3 "$SCR/gen_overseer.py" --runs "$RUNS" --tags "$missing" \
        --stale-min 30 --write-dead || true
    if [ $el -gt $((MAXH * 60)) ]; then
        echo "[gen_wait] TIMEOUT ${MAXH}h — faltan:$missing"; exit 1
    fi
    sleep "$POLL"
done
