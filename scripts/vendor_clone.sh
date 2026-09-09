#!/usr/bin/env bash
# scripts/vendor_clone.sh — descarga los repos de referencia para el plan
# prosa-dllm. Se ejecuta en el cluster (o localmente) y deja el código bajo
# third_party/ (no trackeado en git por .gitignore).

set -euo pipefail

REPOS=(
    "kuleshov-group/mdlm"
    "kuleshov-group/bd3lms"
    "ML-GSAI/RADD"
    "ML-GSAI/LLaDA"
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/third_party"
mkdir -p "$VENDOR"
cd "$VENDOR"

for repo in "${REPOS[@]}"; do
    name="$(basename "$repo")"
    url="https://github.com/$repo.git"
    if [ -d "$name/.git" ]; then
        echo "[update] $name"
        (cd "$name" && git pull --ff-only)
    else
        echo "[clone] $repo -> $name"
        git clone --depth=1 "$url" "$name"
    fi
done

echo "=== third_party listo ==="
ls -1 "$VENDOR"
