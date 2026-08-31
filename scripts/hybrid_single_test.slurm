#!/usr/bin/env bash
#SBATCH --job-name=hybrid-single
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/hybrid_single_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/hybrid_single_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/scripts:$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv "$BASE/pytorch-cuda.sif" env PYTHONPATH="$BASE/scripts:$BASE/venv-extra:${PYTHONPATH:-}" \
  python3 - <<'EOF'
import torch, json
from transformers import AutoModel, AutoTokenizer
from train_hybrid import add_lora, sft_step, tokenize_pairs, build_batches
DIR = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct"
tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True, local_files_only=True)
m = AutoModel.from_pretrained(DIR, trust_remote_code=True, dtype=torch.bfloat16, local_files_only=True).to("cuda")
add_lora(m)
m.train()
items = tokenize_pairs("/beegfs/a474r867/ecoreasoner/data/l1/sft_v3_pairs.jsonl")
bs = build_batches(items, tok, 1536, 8)[:3]
B = len(bs[0]); T = max(len(x[0]) for x in bs[0])
ids = torch.full((B, T), 156892, dtype=torch.long)
rs, rl = [], []
for j, (idj, s, l) in enumerate(bs[0]):
    ids[j, :len(idj)] = torch.tensor(idj, dtype=torch.long); rs.append(s); rl.append(l)
loss, n = sft_step(m, (ids, torch.tensor(rs), torch.tensor(rl)), "cuda")
print("loss:", loss.item(), "| grad_fn:", loss.grad_fn is not None, "| n_mask:", n)
loss.backward()
ng = sum(1 for p in m.parameters() if p.grad is not None)
print("OK backward, params con grad:", ng)
EOF
echo "single test rc=$?"