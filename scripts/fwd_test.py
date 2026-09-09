import os, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

MODEL = os.environ["QWEN_MODEL"]
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", rank=local_rank, world_size=2)

print(f"[rank{local_rank}] init FSDP...", flush=True)
mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
wrap = transformer_auto_wrap_policy([Qwen3_5MoeDecoderLayer])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model = FSDP(model, auto_wrap_policy=wrap, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD)
print(f"[rank{local_rank}] FSDP OK", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
inp = tok(["Describe the niche of a species.", "Climate affects range."], return_tensors="pt").input_ids.to(local_rank)
out = model(input_ids=inp, labels=inp)
print(f"[rank{local_rank}] forward OK, loss: {out.loss.item()}", flush=True)
out.loss.backward()
print(f"[rank{local_rank}] backward OK", flush=True)
dist.destroy_process_group()
