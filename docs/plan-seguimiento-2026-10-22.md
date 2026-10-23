# Plan de seguimiento — EcoReasoner (2026-10-22)

Horizonte: ~3 semanas. Base: V6-2B-RCCL-CRASHLOOP-2026-10-22 +
HANDOFF-2026-10-15. Estado al escribir: SFT-v3/v4 cerrados hoy;
v6-2b migrado a NVIDIA (ola 32262353 en cola); sciev #4 en espera.

Convención: **[AUTO]** corre solo · **[MANUAL]** acción nuestra ·
**[DECISIÓN]** requiere a Angel.

---

## 0. Objetivos (lista maestra)

| # | objetivo | estado | tipo |
|---|----------|--------|------|
| O1 | v6-2b entrena en NVIDIA sin crash-loop | ola en cola | AUTO |
| O2 | Verificar 1ª ola NV (steps reales + ckpt g13001) | espera arranque | MANUAL |
| O3 | Curva v6-2b con baterías c/25K hasta señal legible (~g100-300K) | watcher vivo | AUTO |
| O4 | Veredicto capacidad-vs-receta (gen_exact v6 vs curva v5-1b) | pendiente O3 | DECISIÓN |
| O5 | Probe chat multi-turno sobre sft-v4-g16000 | pendiente | MANUAL |
| O6 | sciev 4º punto sobre g1300000 | bloqueado externo | DECISIÓN |
| O7 | P8 (systems+harness+falsaciones) — redacción | todo el material listo | MANUAL |
| O8 | Integración UNAM-EN al corpus (dosis vs solo-QA) | abierto desde 10-04 | DECISIÓN |
| O9 | Destino pool MI210 (reportar nodos / NRANKS=1 / abandonar DDP) | pendiente | DECISIÓN |
| O10 | Veredicto final tesis (exposición→deriva; falsaciones) | tras O4+O6 | DECISIÓN |
| O11 | Push de 30+ commits locales a origin | pendiente | MANUAL |

## 1. Cadena automática viva

- `watch_v6_battery.sh` — dispara batería (dense+cons_inf+ctxnec+
  remask) en cada ckpt con gap ≥25K y en el terminal. Ya vio el
  run hasta g12501; retoma solo al llegar ckpts nuevos.
- `watchdog_ecoreasoner.sh` (cron */15 + @reboot) — mantiene
  watchers, mainline_guard v6-2b-nv (resubmit si state.json >40min
  stale sin job) y v5-1b (inerte, flag puesto).
- AUTO_RESUBMIT dentro de cada ola NV (exit 42 graceful + ola nueva).

## 2. Secuencia esperada

| cuándo | evento | tipo |
|---|---|---|
| ~10-23 mañana | ola 32262353 arranca (estimado 05:17; puede ser antes) | AUTO |
| +5min arranque | **verificación O2**: `step` lines en train.log, VRAM ok | MANUAL |
| +2-4h | ckpt g13001 → confirma pipeline completo (resume+save NV) | AUTO |
| ~c/1-2 días | batería automática c/25K si la ola vive | AUTO |
| ~1-2 sem | señal O4 legible comparando gen_exact/remask vs curva v5-1b | DECISIÓN |
| libre | probe chat sft-v4 (O5) — barato, 1 GPU, ~30min | MANUAL |
| libre | push git + empezar P8 (O7/O11) | MANUAL |
| cuando respondan | sciev #4 (O6) — set comparable del addendum | DECISIÓN |

## 3. Contingencias v6-2b

| señal | causa | acción |
|---|---|---|
| OOM otra vez en 44-48GB | tier aún justo | bajar a `bs1 acc4 + EMA-CPU` (editar tier, redeploy, cancel+resubmit para hornear) |
| Job PD >24h | islas saturadas | variante SHAPE_N=2 (2 GPUs) o canary 1-GPU en OUT aparte |
| Steps pero ~sin progreso | EMA-CPU@8 lento | medir tok/s vs l40 (~9.7K v5); si <50% considerar islas ≥60G solas |
| `step` en log pero loss NaN | ckpt ROCm malo | re-resume desde g12001 (ocultar g12501) |
| allreduce hang en NVIDIA | no debería (NCCL estable en v5) | escalar a Angel — sería bug nuevo |

## 4. O5 — probe chat sft-v4 (el test barato pendiente)

sft-v4-g16000 ya tiene los datos multi-turno encima (66K docs, 39.7%
multi-span). Falta la sonda de chat: generar K diálogos 4-turnos y
medir coherencia/formato. Es el primer dato de si el cuello de chat
es datos o capacidad — pista independiente de la curva v6.

## 5. O6 — sciev (sin cambios)

- Condición intacta: no lanzar mientras su dev esté activo. Nota en
  `/beegfs/a474r867/sciev-code/COORDINACION-ECOREASONER-2026-10-11.md`
  sigue sin respuesta (revisar semanal en monitoreo).
- Set: `ev5g1300k_{choice,noul,score}` + jev_gpqa_main + jev_scifact
  (versión comparable a puntos 1-3).

## 6. O9 — pool MI210 (decisión de fondo)

Evidencia: deadlock RCCL/SHM ~95% de olas en 7+ nodos, kernel
`queue evicted` + kworkers pegados. Opciones:
- **reportar**: mail a crc@ku.edu con jobs/firma (los nodos seguirán
  degradando — ya hay julia OOMs ajenos en dmesg).
- **NRANKS=1**: si se quiere rescatar el pool (24K tok/s bs12 > DDP
  3-rank 13.8K — sin colectivas no hay deadlock). Requiere script
  con `gres=gpu:mi210:1` + ACCUM=3.
- **abandonar**: v6-2b ya en NVIDIA; MI210 solo para smokes futuros.

## 7. Monitoreo (cheat-sheet actualizado)

```bash
# v6-2b NV: arranque + steps
ssh kuhpc 'squeue -u a474r867 -h -n v6-2b-nv; tail -5 /beegfs/a474r867/ecoreasoner/runs/v6-2b/train.log'

# baterías v6 (cuando haya ckpts nuevos)
tail -5 ~/ecoreasoner/data/watch_v6_battery.log

# flags
ls ~/ecoreasoner/data/*.flag
```

Cadencia: 1 chequeo al arrancar la ola (crítico), luego diario.
Regla de oro vigente: conteo congelado >2h = bug, no lentitud.

## 8. Definición de "fase cerrada" (actualizada)

- [x] v5-1b completo + veredicto terminal (ctxnec 0.752)
- [x] SFT-v3 + regresión ctxnec (0.7208 — binding intacto)
- [x] SFT-v4 multi-turno + regresión (0.7233)
- [ ] v6-2b curva comparable + veredicto capacidad-vs-receta
- [ ] sciev 4º punto (trayectoria 4 pts)
- [ ] Probe chat MT (sft-v4; luego v6-2b-terminal+mt si aplica)
- [ ] P8 redactado / P9 tras ckpt terminal v6
- [ ] Veredicto final tesis
