#!/usr/bin/env bash
#SBATCH --job-name=g0-watch
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/g0_watch_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/g0_watch_%j.err
# Piloto automatico EvoG0: espera los 8 verdicts dense, corre leaderboard,
# escribe ranking + propuesta G1 en runs/g0_next_gen_proposal.txt.
# Si se acaba la ola sin completar, se re-encola (patron waves).
set -u
BASE=/beegfs/a474r867/ecoreasoner
RUNS=$BASE/runs
SIF=$BASE/pytorch-cuda.sif
TAGS="g0-cf05 g0-nocur g0-lr4 g0-norole g0-cap g0-ep2 g0-marked g0-aug"
DONE_FILE=$RUNS/g0_generation_done.flag

missing() {
    for t in $TAGS; do
        [ -f "$RUNS/$t/battery_verdict_dense.json" ] || echo "$t"
    done
}

if [ -f "$DONE_FILE" ]; then
    echo "generacion ya cerrada ($DONE_FILE existe); saliendo"; exit 0
fi

for i in $(seq 1 64); do   # 64 x 5min = 5.3h por ola
    m=$(missing)
    if [ -z "$m" ]; then break; fi
    echo "[$(date -u +%H:%M)] esperando: $m"
    sleep 300
done

m=$(missing)
if [ -n "$m" ]; then
    echo "ola agotada, faltan: $m -> re-encolo"
    sbatch --export=ALL "$0"
    exit 42
fi

echo "=== G0 COMPLETA $(date -u) — ranking ==="
apptainer exec "$SIF" python3 "$BASE/scripts/leaderboard.py" \
    --runs "$RUNS" --glob 'g0-*/battery_logicdiff/dense' \
    --out "$RUNS/g0_leaderboard.jsonl" | tee "$RUNS/g0_leaderboard.txt"

# Propuesta G1: top-2 por fitness -> mutantes (regla pre-registrada ROADMAP)
apptainer exec "$SIF" python3 - "$RUNS/g0_leaderboard.jsonl" "$RUNS/g0_next_gen_proposal.txt" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
top = [r for r in rows if r.get("fitness") is not None][:2]
lines = ["# Propuesta G1 (auto, g0_watch)", "",
         f"top-1: {top[0]['run'] if len(top)>0 else '?'} fit={top[0].get('fitness') if len(top)>0 else '?'}",
         f"top-2: {top[1]['run'] if len(top)>1 else '?'} fit={top[1].get('fitness') if len(top)>1 else '?'}", "",
         "Regla: 6 mutantes de top-2 (combinar genes si compatibles +",
         "intensificar el gen ganador) + 2 probes nuevos. Confirmar y",
         "lanzar con: TAG=g1-<x> <knobs> sbatch --export=ALL --job-name=$TAG g0_run.slurm"]
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

touch "$DONE_FILE"
echo "=== G0 WATCH DONE $(date -u) ==="
