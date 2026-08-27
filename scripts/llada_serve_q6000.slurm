#!/bin/bash
#SBATCH --job-name=llada-serve
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/llada-serve-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/llada-serve-%j.err

# LLaDA-8B-Instruct serving on Q6000 (CUDA) via Apptainer + transformers
# Serves OpenAI-compatible API on port $SERVE_PORT

set -euo pipefail

echo "[$(date)] LLaDA-8B-Instruct serve job starting on Q6000"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"

export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots"
MODEL_DIR=$(ls -d ${MODEL_PATH}/*/ | head -1)
echo "Model dir: $MODEL_DIR"

SIF="/beegfs/a474r867/ecoreasoner/pytorch-cuda.sif"
SERVE_PORT=${SERVE_PORT:-8080}

echo "${SLURM_JOB_NODELIST}:${SERVE_PORT}" > /beegfs/a474r867/ecoreasoner/.current_llada_node

cat > /tmp/llada_serve.py << 'PYEOF'
import os, json, time, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "")
PORT = int(os.environ.get("SERVE_PORT", "8080"))

print(f"Loading LLaDA-8B-Instruct from {MODEL_DIR}...", flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Check if flash attention is available
try:
    import flash_attn
    attn_impl = "flash_attention_2"
    print(f"Using flash_attention_2 (v{flash_attn.__version__})", flush=True)
except ImportError:
    attn_impl = "eager"
    print("Using eager attention (no flash_attn)", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=attn_impl,
)
model.eval()

t1 = time.time()
print(f"Model loaded in {t1-t0:.1f}s on {model.device}", flush=True)

# Get mask token
MASK_ID = tokenizer.mask_token_id
if MASK_ID is None:
    # LLaDA uses a special mask token
    MASK_ID = tokenizer.convert_tokens_to_ids("[MASK]")
    if MASK_ID is None:
        MASK_ID = tokenizer.unk_token_id or 0
print(f"Mask token ID: {MASK_ID} ({tokenizer.decode([MASK_ID])})", flush=True)

def llada_generate(messages, max_new_tokens=256, num_steps=128, temperature=0.0):
    """Generate using LLaDA masked diffusion sampling."""
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(model.device)
    
    prefix_len = input_ids.shape[1]
    response_length = max_new_tokens
    
    # Append mask tokens for the response
    masks = torch.full((1, response_length), MASK_ID, dtype=torch.long, device=model.device)
    input_ids = torch.cat([input_ids, masks], dim=1)
    
    steps = min(num_steps, response_length)
    
    with torch.no_grad():
        for step in range(steps):
            # Forward pass
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            
            # Find masked positions
            mask_positions = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                break
            
            # Number of tokens to unmask this step
            remaining_steps = steps - step
            n_to_unmask = max(1, len(mask_positions) // remaining_steps)
            
            if temperature > 0:
                probs = torch.softmax(logits[0, mask_positions] / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                confidences = probs.max(dim=-1)[0]
            else:
                sampled = logits[0, mask_positions].argmax(dim=-1)
                confidences = logits[0, mask_positions].max(dim=-1)[0]
            
            # Unmask top-k most confident
            top_k = min(n_to_unmask, len(mask_positions))
            top_indices = confidences.topk(top_k).indices
            
            for idx in top_indices:
                pos = mask_positions[idx]
                input_ids[0, pos] = sampled[idx]
    
    # Decode only the response
    response_ids = input_ids[0, prefix_len:prefix_len + response_length]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            
            if self.path == "/v1/chat/completions":
                messages = body.get("messages", [])
                max_tokens = body.get("max_tokens", 256)
                temperature = body.get("temperature", 0.0)
                num_steps = body.get("num_steps", 128)
                
                t0 = time.time()
                response = llada_generate(messages, max_new_tokens=max_tokens,
                                         num_steps=num_steps, temperature=temperature)
                t1 = time.time()
                elapsed = t1 - t0
                tokens = len(tokenizer.encode(response))
                tps = tokens / elapsed if elapsed > 0 else 0
                
                result = {
                    "model": "llada-8b-instruct",
                    "choices": [{"message": {"role": "assistant", "content": response}}],
                    "usage": {"total_tokens": tokens, "completion_tokens": tokens},
                    "timing": {"elapsed_s": round(elapsed, 2), "tokens_per_s": round(tps, 1),
                               "num_steps": num_steps, "max_tokens": max_tokens},
                }
                print(f"  [{time.strftime('%H:%M:%S')}] {tokens} tok / {elapsed:.1f}s = {tps:.1f} tok/s (steps={num_steps})", flush=True)
            elif self.path == "/v1/models":
                result = {"data": [{"id": "llada-8b-instruct", "object": "model"}]}
            else:
                result = {"error": f"unknown endpoint: {self.path}"}
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"ERROR: {e}\n{tb}", flush=True)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e), "traceback": tb}).encode())
    
    def do_GET(self):
        if self.path == "/v1/models":
            result = {"data": [{"id": "llada-8b-instruct", "object": "model"}]}
        elif self.path == "/health":
            result = {"status": "ok", "model": "llada-8b-instruct"}
        else:
            result = {"status": "ok"}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def log_message(self, format, *args):
        pass  # suppress default logging

print(f"Starting server on 0.0.0.0:{PORT}...", flush=True)
server = HTTPServer(("0.0.0.0", PORT), Handler)
print(f"LLaDA-8B-Instruct serving on :{PORT}", flush=True)
sys.stdout.flush()
server.serve_forever()
PYEOF

# Install transformers in the container if needed, then serve
apptainer exec --nv \
  --env HF_HOME=$HF_HOME \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env MODEL_DIR=$MODEL_DIR \
  --env SERVE_PORT=$SERVE_PORT \
  $SIF bash -c '
    pip install --quiet transformers accelerate 2>&1 | tail -3
    echo "Python deps installed"
    python3 /tmp/llada_serve.py
  '

echo "[$(date)] LLaDA serve job ending"
