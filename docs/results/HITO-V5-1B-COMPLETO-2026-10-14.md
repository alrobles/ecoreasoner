# HITO — v5-1b mainline COMPLETO (g1,300,000)

**Fecha**: 2026-10-14 21:21 · Estado: `training_complete.flag` +
`checkpoint-g1300000` escritos, `srun exit 0`.

## Lo que se completó

| dimensión | valor |
|---|---|
| Pasos optimizador | **1,300,000** (objetivo cumplido) |
| Tokens consumidos | **~11.96B** (9,216 tok/paso) |
| Documentos únicos | ~17.38M |
| Wall-clock | ~26 días (lanzado ~09-18) |
| Arquitectura | MdLM-MoE 16E top-1 · 768h×12L · **1.13B total / 285M activos** |
| Checkpoints | EMA 0.999 + raw, ZeRO-1 consolidado, atomic |
| Infraestructura | flota heterogénea NVIDIA (6 tipos de isla), DDP intra-isla, AUTO_RESUBMIT sobre sixhour |
| Evaluación | 50 puntos de batería producidos automáticamente por watcher |

## Interrupciones sobrevividas

| fecha | evento | tiempo perdido | mecanismo |
|---|---|---:|---|
| 10-07→08 | apagón BeeGFS | ~24h | cadena relanzada manual + watchers revividos |
| 10-14 15:37 | muerte silenciosa de cadena | ~2h | `sbatch` de `finalize()` tragado por `2>/dev/null` — ahora loguea + reintenta ×3 |
| continuo | TIMEOUT sixhour ×~190 olas | 0 | SIGUSR1→save→exit 42→resubmit |
| 10-15 | batería terminal cayó en nodo MI210 | 0 (detectado) | `--constraint` NVIDIA añadido a los 3 sbatch de eval |

## Estado de la curva (EMA, pre-terminal)

ctxnec pairwise banda 0.739–0.761 (ruido eval ±1.5pt), tendencia suave
~0.743→0.75. `cons_inf` 0.66–0.67 estable. Generativas con oscilación
amplia pero **máximos de serie concentrados en g1.0M+** (gx_ctx 0.314 en
g1201K, remask 0.353, gx_inf 0.337 en g1284K). Lectura: deriva lenta real,
no meseta plana ni ruptura — **la fila terminal decide el veredicto final**.

## Lo que implica

**Inmediato (corriendo)**
- Batería terminal sobre `g1300000` (jobs 31810931-33): dense + cons_inf +
  ctxnec/remask — la última fila de la tabla y el dato que cierra la
  pregunta de exposición.

**Decisiones que quedan sobre la mesa**
1. **sciev #4** — punto terminal de la trayectoria backbone→decisión,
   coordinado con la sesión paralela (nota en
   `sciev-code/COORDINACION-ECOREASONER-2026-10-11.md`); condición: su
   harness estable y comparable a los puntos 1-3.
2. **SFT-v3** — datos listos (`sft_ids_v3_unam.npz`, 148,838 pares /
   14.43M tok), ckpt listo, script listo. Pendiente: decisión del usuario.
   Función: tercer punto de la curva SFT-sonda (¿gold_recall sigue
   subiendo con exposición?), no producto chat.
3. **MI210** — pool AMD (~81 GPUs, 30.9K tok/s/3-isla) ahora libre del
   mainline. Opciones: segundo backbone más grande (el cuello generativo
   es capacidad, no datos), o barrido de ablaciones.

**Para P8** — el arco de sistemas está completo y documentado: flota
heterogénea 26 días, 5 fallos estructurales encontrados por monitoreo
(2×mtolmo, qa_gen, sbatch silencioso, watcher gap terminal + eval-en-AMD),
batería auto-producida. La sección "ops como infraestructura científica"
tiene su caso completo.

**Para P9** — la pregunta madre (¿aprende el dLLM discriminación
inferencial?) tiene su respuesta empírica casi cerrada: discriminación
ctxnec ~0.75 sostenida y deriva lenta al alza en todo el tramo final,
generación abierta rota a esta escala (chat=no documentado), con el veredicto
pendiente solo del punto terminal.
