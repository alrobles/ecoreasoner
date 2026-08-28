# BRANCH moe-v4-blackwell — RETOMAR MAÑANA (2026-08-28)

Objetivo: mover el entrenamiento del MoE v4 dLLM del Q6000 (24GB, batallando) a las
Blackwell RTX PRO 6000 (96GB VRAM), como pidió el usuario, para avanzar hacia el
MÍNIMO PROTOTIPO VIABLE.

## POR QUÉ Blackwell
- Q6000 = 24GB, sweet spot encontrado pero apretado (MoE 4 exp, batch 1).
- Blackwell pro6000 = 97.9GB VRAM, driver 580.82.07 / CUDA 13.0, capability (12,0)=sm_120.
- Nodos: r30r24n01 (2 GPU pro6000 LIBRES), r30r08n01 (2, ocupadas x ollama-v4serve teacher),
  r23r09n01 (1, ocupada).
- Beneficio: batch GRANDE -> más expertos activos/iter -> ataca la raíz del deadlock de grad
  (además del probe). Comparación limpia vs dense-control.

## ¡BLOQUEANTE CRÍTICO detectado (ya justificado por qué antes no corría)!
El SIF `pytorch-cuda.sif` trae **PyTorch 2.4.1+cu121 que NO soporta sm_120** (solo hasta sm_90).
El smoke 27476186 falló con: "NVIDIA RTX PRO 6000 ... sm_120 is not compatible with the
current PyTorch installation. PyTorch supports sm_50..sm_90". Esa es la razón de fondo.

## Progreso hoy (commits en moe-v4-blackwell)
1. `2309649` slurm + smoke Blackwell; detectado bloqueante sm_120.
2. `545fd5c` bootstrap overlay + torch cu128.
3. `e6de1b7` fix unbound var OV->OVL (3 intentos del mismo typo; reescrito limpio).
- Scripts: `moe_v4_blackwell.slurm` (batch 4), `moe_v4_blackwell_smoke.slurm` (30 steps),
  `setup_blackwell_torch.slurm` (crea overlay + instala torch 2.7.1 cu128 + valida sm_120).

## Estado EXACTO de los jobs (28-08 00:35 CDT)
- `27476261` (torch install) FALLÓ: overlay 20GB creado OK (`/beegfs/a474r867/ecoreasoner/
  blackwell-torch.overlay`, 21GB), pero `pip install` murió:
  `ERROR: ... OSError [Errno 39] Directory not empty: /opt/conda/.../triton-3.0.0.dist-info/`
  (desinstalar triton de la capa base inmutable dentro del overlay).
- `27475827` (Q6000 moe_v4_single): RUNNING en r22r15n01, pero train.log SIN pasos nuevos
  desde "Resumed checkpoint-g1" ~25min. Sospechoso (lentitud o cuelgue) — pendiente revisar:
  NO hay OOM ni error DDP en su .err, solo FutureWarnings torch.load.

## CÓMO CONTINUAR MAÑANA (orden)
1. Fix del `pip install`: installar torch en LUGAR LIMPIO, no sobre la capa base del SIF.
   Opción robusta: dentro del overlay, crear venv `--overlay OVL python3 -m venv /bb/bwvenv`
   y `BWenv/bin/pip install torch==2.7.1 cu128 --ignore-installed`, luego correr training con
   `--overlay` + `PYTHONPATH`/`PATH` del venv. O `pip install --overlay --upgrade --force-reinstall`.
   Alternativo más simple: NO actualizar en el SIF; `pip install --no-deps --ignore-installed
   torch==2.7.1` dentro del overlay para que añada SIN desinstalar triton de base.
2. Tras torch OK: lanzar `moe_v4_blackwell_smoke.slurm` (20 steps, DDP world=2) para validar
   sm_120 + batch 4 + fix de grad en Blackwell real.
3. Sí smoke OK: lanzar `moe_v4_blackwell.slurm` (6000 steps, batch 4) — el job productivo.
4. Ajustar slurm de training para montar `--overlay blackwell-torch.overlay`.
5. Q6000 (27475814): decidir si sigue o se cancela (porque Blackwell lo reemplaza como prod).

## Decisión de diseño (validar mañana)
Batch 4 quiere explotar 96GB y reducir expertos inactivos. Si se quiere más capacidad, usar
`n_experts 8` (ahora cabe). Empezaré con ne=4 idéntico al Q6000 para comparación limpia, luego upscale.