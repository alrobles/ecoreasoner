#!/usr/bin/env python3
"""_debug_tok_mem.py — mide RSS del batch-encoding del tokenizer LLaDA."""
import json, os, sys, time

def rss():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return l.strip()
    return "?"

TOK = "/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
IN = "/beegfs/a474r867/ecoreasoner/data/skeleton/train_skeleton_train.jsonl"

print("[t0]", rss(), flush=True)
if "--with-torch" in sys.argv:
    t0 = time.time()
    import torch  # noqa
    print(f"[t0.5] import torch {time.time()-t0:.1f}s", rss(), flush=True)
from transformers import AutoTokenizer
print("[t1] import transformers", rss(), flush=True)
tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True,
                                  trust_remote_code=True, use_fast=True)
print("[t2] tokenizer loaded fast=", getattr(tok, "is_fast", "?"), rss(), flush=True)

texts = []
with open(IN) as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        d = json.loads(line)
        if d.get("text"):
            texts.append(d["text"])
print("[t3]", len(texts), "texts", rss(), flush=True)

for B in (1, 8, 32, 128, 512):
    chunk = texts[:B]
    t0 = time.time()
    enc = tok(chunk, return_offsets_mapping=True, max_length=768,
              truncation=True, add_special_tokens=False)
    dt = time.time() - t0
    n = sum(len(x) for x in enc["input_ids"])
    print(f"[B={B}] {dt:.2f}s {n} tok", rss(), flush=True)
    del enc

# también probar encode individual en loop
t0 = time.time()
for t in texts[:100]:
    enc = tok(t, return_offsets_mapping=True, max_length=768,
              truncation=True, add_special_tokens=False)
print(f"[loop-100] {time.time()-t0:.2f}s", rss(), flush=True)
