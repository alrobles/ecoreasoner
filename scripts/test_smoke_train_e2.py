#!/usr/bin/env python3
"""smoke_train_e2.py — smoke test del trainer E2+E4 con loss aux + LR decay, en CPU.

Usa datos sintéticos diminutos; NO toca datos reales. Verifica que el pipeline
(parse -> build -> aux loss -> scheduler -> ckpt) corre sin crashear con los flags nuevos.
"""
import sys, os, tempfile, numpy as np, torch

d = tempfile.mkdtemp(prefix="e2smoke_")
np.save(os.path.join(d, "ids.npy"), np.random.randint(0, 100, [64, 768], dtype=np.int32))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import train_mdlm_moe_expE2 as T

# tokenizer chico para el test
T._load_tokenizer = lambda: type("_T", (), {"vocab_size": 1000})()

T.ARGS = T.parse()
T.ARGS.data_cache = os.path.join(d, "ids.npy")
T.ARGS.max_steps = 6
T.ARGS.log_every = 2
T.ARGS.aux_coeff = 0.01
T.ARGS.lr_decay = "cosine"
T.ARGS.warmup = 3
T.ARGS.ckpt_every = 4
T.ARGS.hidden = 128
T.ARGS.layers = 2
T.ARGS.heads = 4
T.ARGS.ff_mult = 2
T.ARGS.seq_len = 768
T.ARGS.batch_size = 2
T.ARGS.grad_accum = 2
T.ARGS.n_experts = 8
T.ARGS.out_dir = os.path.join(d, "out")
T.DEVICE = torch.device("cpu")
T.main()
print("SMOKE OK:", os.listdir(T.ARGS.out_dir))