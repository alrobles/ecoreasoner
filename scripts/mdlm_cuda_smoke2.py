import sys
sys.argv = ["train_mdlm_moe.py","--hidden","768","--layers","12","--heads","12",
            "--n_experts","8","--expert_k","1","--ff_mult","4","--seq_len","768",
            "--batch_size","2","--grad_accum","2","--lr","2e-4","--warmup","200",
            "--mask_p","0.15","--max_steps","50","--data","/beegfs/a474r867/ecoreasoner/data/sci_v1.jsonl",
            "--output","/beegfs/a474r867/ecoreasoner/outputs/smoke2"]
import sys as _s; _s.path.insert(0, '/beegfs/a474r867/ecoreasoner/scripts')
import torch, torch.nn.functional as F, time
from train_mdlm_moe import MdLMMoE, ARGS
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
m = MdLMMoE(ARGS.vocab, 768, 12, 12, 4, 768, 8, 1).cuda()
n = sum(p.numel() for p in m.parameters()); print(f'model {n/1e6:.1f}M')
torch.manual_seed(0)
x = torch.randint(0,30000,(2,768)).cuda(); mp = torch.randperm(768)[:120].cuda()
xm = x.clone(); xm[:,mp]=32000
out = m(xm); print('fwd', out.shape)
loss = F.cross_entropy(out[:,mp].reshape(-1,32000), x[:,mp].reshape(-1)); print('loss', loss.item())
loss.backward()
g = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum().item()>0)
print(f'backward OK {g}/{len(list(m.parameters()))} grads')
t0=time.time()
for _ in range(5):
    o=m(xm); F.cross_entropy(o[:,mp].reshape(-1,32000), x[:,mp].reshape(-1)).backward()
print(f'5 fwd+bwd steps: {time.time()-t0:.1f}s')
print('CUDA SMOKE PASS')
