# Sesión autónoma noche 2026-08-29 — registro

## Contexto
Usuario autorizó modo autónomo hasta mañana + sudo. Orden: matar serve v4-flash
(no relanzar), preparar 4 Blackwell para el siguiente training cuando termine
bw0, buscando mejora sobre la última versión. Destilación B1 dejada para mañana.

## Acciones ejecutadas
1. Cancelado job 27500630 (ollama-v4serve) + su relanzado 27500686.
2. **Pausado cron `v4flash-keepalive-15m`** (2d884420a767) — era el único script
   que relanzaba ollama-v4serve en pro6000 (r30r08n01), chocaba con las 4 Blackwell.
3. Auditoría nocturna: governors solo gestionan q6000/qwen36; swarm_watchdog solo
   toca outputs/w\*/m1\* (no bw\*); v4-auto-serve-watch dormido (sin cron); los
   demás watchdogs no tocan pro6000. NADA interfiere con bw1.
4. Destilación B1: proceso moribundo (kill destructivo bloqueado por el sistema);
   1860 trazas seguras; se retoma con --append mañana.

## Pipeline bw0 → bw1 (martes del entrenamiento)
- bw0 (MoE v4, 4 exp top-1, corpus v4, world=2, 1 nodo) entrena hasta 6000.
- Al completar 6000 → flag COMPLETE → job desaparece.
- Cron `ecoreasoner-bw0-bw4-watchdog` (5bd4c38c62e9, 15 min) lanza bw1.
- b1 = moe_v4_bw4.slurm: 2 nodos × 2 pro6000 (4 Blackwell), 8 exp top-2,
  corpus v5 (train_ids_v5.npy), sync_every 4, clip 1, torchrun multi-nodo
  (nnodes=2, rdzv c10d), resubmit solo rank 0.
- Cron `ecoreasoner-bw1-alive-watchdog` (c551ee60cae4) alerta si bw1 muere.
- Cron `ecoreasoner-night-status-log` (38dafd9a390a, 20 min) → /tmp/night_train_status.log.

## Bugs críticos cazados esta noche (evitados antes de lanzar bw1)
1. **Doble AUTO_RESUBMIT multi-nodo**: con 2 nodos, cada nodo ejecuta finalize() →
   fix: resubmit solo rank 0 (SLURM_PROCID==0).
2. **torchrun sin --nnodes**: faltaba rendezvous explícito → fix: --nnodes=2
   --rdzv_backend=c10d --rdzv_id.
3. **RANK vs SLURM_PROCID** (commit 53d22ae): en srun+torchrun los workers heredan
   SLURM_PROCID (0/1 por nodo) pero RANK de torchrun es global (0-3). El trainer
   lisgd leía SLURM_PROCID primero → colisión de ranks. Fix: RANK gana, SLURM
   queda como fallback. Seguro para ambos modos (vanilla srun / torchrun).
   NO afecta a bw0 (usa train_mdlm_moe.py original intacto).

## Transición de ola bw0 (verificada)
- 27496591 recibió SIGUSR1 06:28 UTC, checkpoint g4901, resubmit → 27500699
  (nodo r30r08n01, la 2a Blackwell libre). Resume correcto.

## Merciones TFLOPS (bench)
- V100 PCIe 16GB: FP32 13.5 / FP16 88.5 TFLOPS medidos.
- A100/L40/A40/Q8000/Q6000: benchmarks batch quedaron PD (nodos ocupados);
  specs en docs/MULTINODO-Q6000-TECHO.md.

## Commits de la sesión
- 170f13e bw1 + V100 smoke + bench
- 43afca4 bw4 slurm blindado (resubmit rank0 + torchrun rdzv)
- 53d22ae fix RANK multi-nodo torchrun