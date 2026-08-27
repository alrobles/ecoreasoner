#!/bin/bash
#SBATCH --job-name=llada-serve
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=128G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/llada-serve-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/llada-serve-%j.err

# LLaDA-8B-Instruct serving on MI210 via Apptainer + transformers
# Uses native LLaDA generate.py code (no vLLM dependency)
# Serves OpenAI-compatible API on port $SERVE_PORT

set -euo pipefail

echo "[$(date)] LLaDA-8B-Instruct serve job starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"

export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/beegfs/a474r867/ecoreasoner/scripts:${PYTHONPATH:-}

MODEL_PATH="/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots"
# Resolve symlink to actual snapshot dir
MODEL_DIR=$(ls -d ${MODEL_PATH}/*/ | head -1)
echo "Model dir: $MODEL_DIR"

SIF="/beegfs/a474r867/ecoreasoner/qwen35-rocm-v2.sif"
SERVE_PORT=${SERVE_PORT:-8080}

# Write node:port to file for tunnel setup
echo "${SLURM_JOB_NODELIST}:${SERVE_PORT}" > /beegfs/a474r867/ecoreasoner/.current_llada_node

# Generate the serving script
cat > /tmp/llada_serve.py << 'PYEOF'
import os, json, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "")
PORT = int(os.environ.get("SERVE_PORT", "8080"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading LLaDA-8B-Instruct from {MODEL_DIR}...", flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.to(DEVICE)
model.eval()

t1 = time.time()
print(f"Model loaded in {t1-t0:.1f}s", flush=True)
print(f"Device: {DEVICE}", flush=True)

# LLaDA sampling function (from official generate.py)
def llada_generate(messages, max_new_tokens=256, num_steps=128, cfg=3.0, temperature=0.0):
    """Generate using LLaDA masked diffusion sampling."""
    # Format chat
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(model.device)
    
    # Create response mask region
    response_length = max_new_tokens
    # LLaDA mask token: viene del config del modelo (126336 = <|mdm_mask|>),
    # NO del tokenizer (que no lo expone como atributo con remote code).
    mask_token = int(getattr(model.config, "mask_token_id", None) or getattr(tokenizer, "mask_token_id", None))
    if mask_token is None:
        mask_token = tokenizer.unk_token_id or 0
    
    # Append mask tokens for the response
    masks = torch.full((1, response_length), mask_token, dtype=torch.long).to(model.device)
    input_ids = torch.cat([input_ids, masks], dim=1)
    
    # Number of steps for denoising
    steps = min(num_steps, response_length)
    
    # Token positions to denoise (the masked region)
    prefix_len = input_ids.shape[1] - response_length
    
    with torch.no_grad():
        for step in range(steps):
            # Forward pass (una sola vez)
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
            
            # Get predictions for masked positions only
            mask_positions = (input_ids[0] == mask_token).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                break
            
            # Number of tokens to unmask this step
            n_to_unmask = max(1, len(mask_positions) // (steps - step))
            
            if temperature > 0:
                # Sample with temperature
                probs = torch.softmax(logits[0, mask_positions] / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                # Get confidence scores
                confidences = probs.max(dim=-1)[0]
            else:
                # Greedy: pick most confident predictions
                sampled = logits[0, mask_positions].argmax(dim=-1)
                confidences = logits[0, mask_positions].max(dim=-1)[0]
            
            # Unmask top-k most confident
            top_k = min(n_to_unmask, len(mask_positions))
            top_indices = confidences.topk(top_k).indices
            
            for idx in top_indices:
                pos = mask_positions[idx]
                input_ids[0, pos] = sampled[idx]
    
    # Decode only the response
    response = tokenizer.decode(input_ids[0, prefix_len:], skip_special_tokens=True)
    return response

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            
            if self.path == "/v1/chat/completions":
                messages = body.get("messages", [])
                max_tokens = body.get("max_tokens", 256)
                temperature = body.get("temperature", 0.0)
                num_steps = body.get("num_steps", 128)  # LLaDA-specific
                
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
                    "timing": {"elapsed_s": round(elapsed, 2), "tokens_per_s": round(tps, 1)},
                }
                print(f"  Generated {tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)", flush=True)
            elif self.path == "/v1/models":
                result = {"data": [{"id": "llada-8b-instruct", "object": "model"}]}
            else:
                result = {"error": "unknown endpoint"}
            
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
        if self.path == "/v1/models":
            result = {"data": [{"id": "llada-8b-instruct", "object": "model"}]}
        elif self.path == "/health":
            result = {"status": "ok"}
        else:
            result = {"error": "unknown endpoint"}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}", flush=True)

print(f"Starting server on port {PORT}...", flush=True)
server = HTTPServer(("0.0.0.0", PORT), Handler)
print(f"LLaDA-8B-Instruct serving on :{PORT}", flush=True)
server.serve_forever()
PYEOF

# Launch inside Apptainer
apptainer exec --rocm \
  --env HF_HOME=$HF_HOME \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env MODEL_DIR=$MODEL_DIR \
  --env SERVE_PORT=$SERVE_PORT \
  $SIF python3 /tmp/llada_serve.py

echo "[$(date)] LLaDA serve job ending"
