# V100 multi-nodo — smoke validado + lanzamiento 32 ranks (2026-08-29)

## Smoke (job 27500653, 2 nodos × 2 V100 = 4 ranks) — VALIDADO

Config compacta fp32 (~449M params):
- hidden 768, layers 12, heads 12, 4 exp top-1, ff_mult 4, seq 512
- batch 4, grad_accum 4, sync_every 4, clip 1.0
- datos: data_cache train_ids_v5.npy + --data jsonl (argparse lo exige)

Resultado:
- NCCL multi-nodo Init COMPLETE (2 nodos, NET/Socket inter-nodo, SHM intranodo)
- model: total=449.3M, active=279.3M (62%)
- 40 steps en ~57s (~1.4s/step): loss 11.83 → 11.50, checkpoints g1/g40, rc=0
- Sin hang, sin OOM (7.3G/16G VRAM estado fp32)

## Pitfalls encontrados (importantes)

1. **CWD del srun**: el srun arranca desde $HOME del login (no BASE) → usar SIEMPRE
   ruta ABSOLUTA al script (`$BASE/scripts/train_mdlm_moe_lsgd.py`), no relativa.
   Error típico: `can't open file '/kuhpc/home/.../scripts/xxx.py'` exit 2.
2. **--data es required** en train_mdlm_moe_lsgd.py aunque se use --data_cache:
   argparse lo exige. Pasar SIEMPRE `--data <jsonl>` (el cache gana si existe).
3. **--gres en srun rompe** ("Invalid generic resource"): el gres va SOLO en
   #SBATCH (del job), nunca en el srun del step. Para bench por tipo, usar
   `sbatch --gres=gpu:X:1` que sobrescribe el header.

## Ola grande

- Comando: `sbatch --export=WAVE_ID=v100-32,TARGET_STEPS=6000 moe_v4_v100.slurm`
- 16 nodos × 2 V100 = 32 ranks, sync_every 8, clip 1.0, batch 4, grad_accum 4
- Memoria: 7.3G estado + act ~1G → ~8-9G/16G OK
- Comunicación estimada: 1.8G (449M grads) cada 8 pasos → ~3-4s red/paso efectivo