#!/usr/bin/env bash
#SBATCH --job-name=tf446-test
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/tf446_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/tf446_test_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python - <<'EOF'
import torch, time
from transformers import AutoModel, AutoTokenizer
DIR = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct"
print("TF", __import__("transformers").__version__, "TOK", __import__("tokenizers").__version__)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True, local_files_only=True)
m = AutoModel.from_pretrained(DIR, trust_remote_code=True, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
print(f"LOADED {sum(p.numel() for p in m.parameters())} ({time.time()-t0:.0f}s)")
msgs=[{"role":"user","content":"Download bioclim data for temperature."}]
pr = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
if hasattr(pr, "input_ids"): pr = pr["input_ids"]
pr = pr.to("cuda")
x = torch.full((1, pr.shape[1]+40), 156895, dtype=torch.long, device="cuda"); x[:, :pr.shape[1]] = pr
with torch.no_grad(): out = m(x).logits
lg = out[0, pr.shape[1]-5:]
print("logits mean %.4f std %.4f nan=%s" % (lg.float().mean().item(), lg.float().std().item(), torch.isnan(lg).any().item()))
top = torch.topk(lg[3], 5)
print("top5 pos3:", [(t, round(float(p),3)) for t,p in zip(tok.convert_ids_to_tokens(top.indices.tolist()), top.values.tolist())])
EOF
echo "test done rc=$?"