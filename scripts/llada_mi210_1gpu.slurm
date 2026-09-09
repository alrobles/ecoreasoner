#!/bin/bash
#SBATCH --job-name=llada-mi210
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/llada-mi210-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/llada-mi210-%j.err

set -euo pipefail

echo "[$(date)] LLaDA-8B MI210 starting"
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/beegfs/a474r867/ecoreasoner/scripts:$PYTHONPATH

MODEL_DIR="/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots"
MODEL_DIR=$(ls -d ${MODEL_DIR}/*/ | head -1)
SIF="/beegfs/a474r867/ecoreasoner/qwen35-rocm-v2.sif"
SERVE_PORT=8080

echo "${SLURM_JOB_NODELIST}:${SERVE_PORT}" > /beegfs/a474r867/ecoreasoner/.current_llada_node

# Write serve script
cat > /tmp/llada_serve.py << "PYEOF"
import os, json, time, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "")
PORT = int(os.environ.get("SERVE_PORT", "8080"))

print(f"Loading LLaDA-8B from {MODEL_DIR}...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s on {next(model.parameters()).device}", flush=True)

MASK_ID = tokenizer.mask_token_id or tokenizer.unk_token_id or 0

def llada_generate(messages, max_new_tokens=256, num_steps=128, temperature=0.0):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(model.device)
    prefix_len = input_ids.shape[1]
    masks = torch.full((1, max_new_tokens), MASK_ID, dtype=torch.long, device=model.device)
    input_ids = torch.cat([input_ids, masks], dim=1)
    steps = min(num_steps, max_new_tokens)
    with torch.no_grad():
        for step in range(steps):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            mask_pos = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                break
            n_unmask = max(1, len(mask_pos) // (steps - step))
            if temperature > 0:
                probs = torch.softmax(logits[0, mask_pos] / temperature, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                conf = probs.max(dim=-1)[0]
            else:
                sampled = logits[0, mask_pos].argmax(dim=-1)
                conf = logits[0, mask_pos].max(dim=-1)[0]
            top_k = min(n_unmask, len(mask_pos))
            top_idx = conf.topk(top_k).indices
            for i in top_idx:
                input_ids[0, mask_pos[i]] = sampled[i]
    return tokenizer.decode(input_ids[0, prefix_len:], skip_special_tokens=True).strip()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/v1/chat/completions":
                msgs = body.get("messages", [])
                max_tok = body.get("max_tokens", 256)
                temp = body.get("temperature", 0.0)
                nsteps = body.get("num_steps", 128)
                t0 = time.time()
                resp = llada_generate(msgs, max_new_tokens=max_tok, num_steps=nsteps, temperature=temp)
                t1 = time.time()
                tok = len(tokenizer.encode(resp))
                result = {"model": "llada-8b-instruct", "choices": [{"message": {"role": "assistant", "content": resp}}],
                          "usage": {"total_tokens": tok, "completion_tokens": tok},
                          "timing": {"elapsed_s": round(t1-t0, 2), "tokens_per_s": round(tok/(t1-t0), 1) if t1>t0 else 0}}
            elif self.path == "/v1/models":
                result = {"data": [{"id": "llada-8b-instruct", "object": "model"}]}
            else:
                result = {"error": "unknown"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())
    def log_message(self, format, *args):
        pass

HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
PYEOF

apptainer exec --rocm --nv \
  --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env MODEL_DIR=$MODEL_DIR --env SERVE_PORT=$SERVE_PORT \
  $SIF python3 /tmp/llada_serve.py
