# EcoReasoner — Reporte de avance dLLM (2026-10-25)

Estado: **v6-2b-nv ENTRENANDO en NVIDIA — g43500 vivo** (job 32329845,
3×A40 r32r05n01, ~11K pasos/día). El brazo de capacidad ya corre limpio
tras el crash-loop RCCL/SHM de MI210 (10-16→10-22, ~45 olas FAILED,
documentado en `V6-2B-RCCL-CRASHLOOP-2026-10-22.md`). **Primer punto de
batería v6-2b (g25001) ya medido** — tabla §1. Cadena SFT cerrada
(v3 + v4 el mismo 10-22); probe chat multi-turno dio el veredicto clave:
**el cuello de chat es capacidad del backbone, no datos MT** → motiva
directamente v6-2b. Continuación del plan-seguimiento-2026-10-22.

---

## 1. v6-2b — estado del brazo de capacidad

Config: L=24 (vs L=12 en v5-1b), 2.06B total / ~365M activos, receta
idéntica (9,216 tok/step), mismo corpus 11.96B tok — el experimento
aisla capacidad (2× activos) con todo lo demás congelado.

**Progreso**: g12501 (residual MI210) → **g43500** (10-25 22:50).
+31K pasos en ~60h ≈ **~11K pasos/día** en la isla NVIDIA actual
(nodo compartido; AUTO_RESUBMIT rota islas cada ola). A este ritmo:
g100K ~10-30, g300K ~11-20 — dentro de la ventana prevista para el
veredicto O4 (señal legible ~g100-300K, sin esperar a 1.3M).

**Primer punto de batería — g25001 (EMA)**:

| métrica | v6-2b g25001 | v5-1b ref (g207501, 1er punto comparable) |
|---|---:|---:|
| ctxnec pairwise L3 | 0.5317 | 0.690 |
| ctxnec gen_exact | 0.063 | 0.240 |
| cons_inf pairwise | 0.5786 | 0.657 |
| cons_inf gen_exact | 0.163 | 0.307 |
| dense L3 | 0.5014 | 0.546 |
| remask gen_exact ctx | 0.061 | ~0.28 (banda tramo tardío) |

Lectura: a g25K el modelo aún es inmaduro (v5-1b no tiene punto de
batería tan temprano — la serie arranca ~g135K solo dense, g207K
completa). La comparación útil empieza cuando v6-2b entre en el tramo
g100-300K contra la curva v5 ya consolidada. Subtipos ctxnec a g25K:
role_swap 0.635 (líder, como en v5), entity_rebind 0.527,
value_rebind 0.513, direction_rebind_word 0.340.

**Próxima batería**: automática ~g50001 (gap 25K desde g25001) vía
`watch_v6_battery.sh` (vivo, polling horario).

## 2. Semana en ops — MI210 → NVIDIA

- **Crash-loop MI210 falsado como causa**: ~45 olas FAILED en ≥7 nodos,
  6 días para +500 pasos. `py-spy` mostró el hang en la PRIMERA
  colectiva DDP (`_verify_param_shape_across_processes`) + `queue
  evicted` en dmesg → inestabilidad RCCL/SHM, no el workload. En NVIDIA:
  +2000 pasos en la primera noche.
- **2º fallo — OOM tier 40-60G** (10-22 23:00): EMA-GPU dejaba ~43.9GB
  en islas 44.4GB. Fix: `bs2 acc2 + zero1 + grad_ckpt + EMA-CPU@8`.
- **Canary 1-GPU (32262493)** validó resume cross-stack ROCm→CUDA
  (model+optim+EMA) y el fix de memoria → O1/O2 cerrados 10-23.
- **Pool MI210 pendiente de destino** (O9): evidencia sugiere
  NRANKS=1 (24K tok/s sin colectivas > DDP 3-rank 13.8K) o abandonar
  DDP ahí; reportar a crc@ sigue como opción.

## 3. Cadena SFT — CERRADA (10-22)

| run | init | datos | veredicto |
|---|---|---|---|
| sft-v3 (g37000) | g1300000-EMA | UNAM-QA 148,838 pares / 14.43M tok | ctxnec **0.7208** — binding intacto (vs 0.752 terminal, en banda) |
| sft-v4 (g16000) | terminal sft-v3 | mt 66,338 docs / **39.7% multi-turno** | ctxnec **0.7233** — datos MT tampoco erosionan |

- Probe chat MT sobre sft-v4 (O5, 10-23): `eos=0% next_turn=0%
  max_new=100%` — degeneración semántica desde turno 0. **El backbone
  1.13B no sostiene generación coherente aunque el formato se absorba**
  → cuello = capacidad. La prueba real de chat queda para
  v6-2b-terminal + sft_ids_v4_mt.
- Bootstrap sft-v3→v4 quedó automatizado en el watchdog (semilla
  `sft_v4_init.txt` al caer `SFT_V3_DONE.flag`).

## 4. Objetivos (lista maestra 10-22)

| # | objetivo | estado |
|---|---|---|
| O1 | v6-2b en NVIDIA sin crash-loop | **HECHO** — 4.5h limpias 10-23, ahora ~60h acumuladas |
| O2 | Verificar 1ª ola NV | **HECHO** — canary + mainline |
| O3 | Curva v6-2b baterías c/25K | **EN CURSO** — 1er punto g25001; próxima ~g50001 |
| O4 | Veredicto capacidad-vs-receta | pendiente — señal ~g100-300K (~1-3 sem) |
| O5 | Probe chat MT sft-v4 | **HECHO** — capacidad es el cuello |
| O6 | sciev 4º punto g1300000 | **bloqueado externo** — "Respuesta sciev" sigue vacía en la nota |
| O7 | P8 redacción | material listo (crash-loop RCCL + migración = evidencia ops nueva) |
| O8 | Integración UNAM-EN | abierto desde 10-04 |
| O9 | Destino pool MI210 | pendiente |
| O10 | Veredicto final tesis | tras O4+O6 |
| O11 | Push a origin | **casi cerrado** — solo 3 commits locales sin push |

## 5. Salud de la automatización

- `watch_v6_battery.sh` y `watch_v5_battery.sh`: vivos (PIDs Oct24),
  polling correcto — el log de v6 reporta `latest=g43001 evaled=g25001`
  consistente con el cluster.
- **Ojo — watchdog_ecoreasoner.sh**: última entrada de log 10-24
  ~18:20 ("ssh fallo, skip"); hay 2 instancias del script corriendo
  desde Oct24 sin escribir desde entonces. Los watchers cubren la
  función crítica y AUTO_RESUBMIT vive dentro de cada job, pero el
  mainline_guard lleva ~28h mudo — conviene verificar que no esté
  bloqueado en un ssh sin timeout (mi ssh a kuhpc funciona hoy).
- Cron `*/15` + `@reboot` intacto en `reumanlab`.

## 6. Cola de decisiones para Angel (orden sugerido)

1. **O9 pool MI210**: reportar nodos a crc@ / rescatar con NRANKS=1 /
   abandonar DDP ahí.
2. **O8 UNAM-EN**: integrar 6,900 pasajes al corpus (dosis) o dejarlos
   solo-QA — abierto desde 10-04.
3. **O7 P8**: empezar redacción — ya hay material nuevo (crash-loop +
   migración cross-stack como evidencia de ingeniería).
4. **O6 sciev**: alternativa si la nota sigue sin respuesta — lanzar
   nosotros mismos el set comparable cuando el harness sciev esté
   inactivo (condición original).
5. O11: `git push` (3 commits).

## 7. Fuente de datos

- Cluster (verificado 10-25 ~23:00): `runs/v6-2b/state.json` (g43001),
  `train.log` (g43500), squeue job 32329845 R 4h+,
  `runs/v6-2b/battery_curve/g25001/*/logicdiff_summary.json`,
  `sciev-code/COORDINACION-ECOREASONER-2026-10-11.md` (respuesta vacía).
- Local: `data/watch_v6_battery.log`, `data/watchdog.log`, `ps` watchers.
