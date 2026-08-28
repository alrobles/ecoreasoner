# SESIÓN: Sweet spot MoE v4 + destilación (2026-08-27) — PUNTO DE RETOMA

## ÉXITO CLAVE: MoE v4 single-nodo FUNCIONA
- Job: **27475104** (`moe_v4_single.slurm`), RUNNING, 1 nodo r22 (3 GPU Q6000), DDP world=3 intra-nodo.
- Modelo: baseline `train_mdlm_moe.py`, **MoE 4 expertos top-1**, hidden 1024, 16 layers, 863.6M total / ~595M activo.
- Dataset: `train_ids_v4.npy` (1.7B tokens, 2.75M docs, corpus v4 = v3 + 1.74M PMC full-text).
- **step 0 loss 12.01, 0 OOM, produciendo steps** = el sweet spot del mejor modelo estable en Q6000.

## El hallazgo técnico que desbloqueó todo (CRÍTICO para retomar)
**El MoE-DDP top-k se cuelga en MULTI-NODO** (12 ranks) pero **funciona en UN solo nodo** (3 GPU intra-nodo):
- Causa: `find_unused_parameters=True` → OOM por buffers; con `False` + expertos no-siempre-usados → all-reduce se bloquea entre ranks en multi-nodo.
- Fix aplicado al baseline (scripts/train_mdlm_moe.py):
  1. `MoEMLP.forward` guarda `_fcount`/`_gate_probs`; método `balance_loss()` (Switch aux: alpha*n*sum(f·P)) que toca TODOS los expertos cada iteración.
  2. En el loop: `raw = glob_model.module if ddp else glob_model; aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks); (loss/ga + aux).backward()`.
  3. `find_unused_parameters=False` (ya no hay params no-usados).
- RESULTADO: funciona en 1 GPU (fix-test 27474922, 50 steps COMPLETE) y en 1 nodo 3 GPU (27475104). NO en multi-nodo.

## Prueba y error de configs MoE (memtest sweep_mem)
| Config | Total | Activos | Peak | ¿Cabe? |
|---|---|---|---|---|
| h768 l12 ne8 (e2-v3) | 733M | 336M | 15.1GB | OK |
| h1024 l16 ne4 | 998M | 595M | 20.4GB | OK (1 nodo) |
| h1024 l16 ne3 | 864M | 595M | 17.7GB | OK |
| h1024 l16 ne2 | 729M | 595M | 15.0GB | OK |
- **El sweet spot elegido: h1024 l16 ne4 (998M/595M act)** en 1 nodo.

## El expE2 (train_mdlm_moe_expE2.py) NO lo usamos para DDP — se congela
- El expE2 (timestep-router + shared) se congela en DDP (deadlock 1ª iteración), aunque tiene Plan B.
- Usar el **baseline train_mdlm_moe.py** (con el fix arriba) para MoE en DDP 1 nodo.

## Otros frentes (activos al cierre)
- **Destilación round4**: COMPLETÓ 88 trayectorias → `distill_v4_round4.jsonl` (teacher deepseek Blackwell r30r08n01:49525, job ollama-v4serve 27473010 vivo).
- **Dense control v4**: completó 6000 steps (loss 0.66) en outputs/moe_dense_v4_ctl/e2_final.pt — baseline de comparación.
- **e2-v3**: loss 0.595, Plan B validado.

## PENDIENTE (próxima sesión)
1. ⚠️ **PROBLEMA ABIERTO (el que hay que resolver)**: el MoE-DDP con batch pequeño deja expertos inactivos SIN grad → DDP exige `find_unused_parameters=True` (OOM) o da RuntimeError. El `balance_loss` con `P.detach()` balancea el gate pero NO da grad a los MLP de expertos inactivos.
   - Error exacto: `Expected to have finished reduction... Parameter indices which did not receive grad for rank 0: 90 91 ... 397` (los MoE experts inactivos del mini-batch).
   - Solución pendiente (elegir): (a) `find_unused_parameters=True` pero en single-nodo donde quizá quepa VRAM (el OOM era multi-nodo), (b) forzar activar todos los expertos cada iteración (token de guardia / routing forzado), (c) hacer que el aux dé grad real a todos los expertos (p.ej. `aux = sum(exp_e(x))` dummy por cada experto), (d) aceptar dense-DDP (que SÍ funciona).
2. Seguir monitorizando/relanzando MoE v4 single-node una vez resuelto el punto 1.
3. "Usar la arquitectura a nuestro favor": investigar cómo apalancar el MoE (especialización por dominio).
4. El usuario pidió guardar y reiniciar para que no se cuelgue el agente.
