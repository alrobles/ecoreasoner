# EcoReasoner — Estado de monitorización para Devin (2026-08-27)

> Instrucciones de operación para Devin (agente autónomo autorizado en ku-hpc).
> **ROL: SOLO MONITORIZAR Y REPORTAR. NO relanzar, NO cancelar, NO escalar jobs.**
> La rotación de jobs (SIGUSR1 auto-reenchain de entrenamiento, cron de teachers) es
> automática: si algo muere, el sistema lo relanza solo. Si NO se relanza solo tras
> 2 ciclos (≈12h), **reportarlo**, no tocarlo.

## Acceso

- Cluster: `ssh kuhpc` (usuario a474r867)
- Base: `/beegfs/a474r867/ecoreasoner/`
- Data: `/beegfs/a474r867/ecoreasoner/data/` (nomenclatura en `docs/dataset-registry.md`)
- Logs: `/beegfs/a474r867/ecoreasoner/logs/`
- Repo: `https://github.com/alrobles/ecoreasoner` (branch main, limpio, todo pusheado)

## Componentes vivos y qué mirar

### 1. Fase A — retrain corpus v3 (mdlm-retrain-v3)
- **Estado**: COMPLETA (flag `outputs/retrain_v3/training_complete.flag` existe, checkpoint-g8000).
- Verificar: `ls /beegfs/a474r867/ecoreasoner/outputs/retrain_v3/training_complete.flag`
- Si el flag NO existe pero el job murió: reportar (puede ser re-relanzado por SIGUSR1).

### 2. Ola E2+E4 v2 (job 27452507) — loss aux balance + LR cosine
- **Estado**: COMPLETA (6000 steps, `outputs/moe_expE2_v2/e2_final.pt`, loss final 0.593, effN 2.49).
- Verificar: `tail -3 /beegfs/a474r867/ecoreasoner/logs/moe_e2v2_27452507.out` y que exista `e2_final.pt`.
- **Métricas a reportar** (del log, formato `[step N] loss X aux Y lr Z entH W effN V`):
  - loss final, aux final, **effN final** (5-8 = balance sano; <3 = router colapsado).
  - Comparar effN final vs el v1 (27441904, sin aux): el v1 tenía effN ~7.6 global pero capa 11 ~1.
- Siguiente: análisis de activación del `e2_final.pt` v2:
  `python3 scripts/analyze_activation_expE2.py --ckpt outputs/moe_expE2_v2/e2_final.pt --data /beegfs/a474r867/ecoreasoner/data/train_ids_v3.npy`

### 3. Destilación — trayectorias v4
- Datasets: `data/distill_v4_round{1,2,3}.jsonl` (round3 = 49 trayectorias).
- Job: `squeue -u a474r867 | grep distill` → si aparece, está activo; reportar nº de trayectorias.
- El teacher v4 (ollama-v4serve) rota por cron cada ~6h (nuevo job+nodo+puerto). El watchdog
  `watchdog_v4teacher.py` (corre en reumanlab, ~/.hermes/scripts) detecta la rotación y reporta.
  NO tocar el teacher: si no responde, esperar a que el cron lo relance.

### 4. Teachers online (para destilar)
| Modelo | Job | Nodo:puerto (puede rotar) |
|---|---|---|
| deepseek-v4-flash | ollama-v4serve | r30r08n01 (port varía, ver output) |
| qwen3.6:35b | ollama-qwen36 ×2 | r08r28n01, r08r30n01 |
| glm-4.7-flash:q4_K_M | ollama-q6000 | r22r10n01 |

Verificar que responden: `curl -s http://<nodo>:<puerto>/v1/models` (desde el cluster, no login).

### 5. Ingesta PMC (pmc-ingest array 27434086)
- **Estado**: array 0-54 lanzado, 55 tareas RUNNING (3:24h de 5:50). Ya hay **55 parquet escritos**,
  **1,100,000 filas** (años: 2024=413K, 2025=586K, 2026=100K). Los jobs pueden seguir RUNNING
  en el flush final — no es problema.
- Verificar:
  - `ls /beegfs/a474r867/ecoreasoner/data/pmc_parquet/shard_*.parquet | wc -l` → deben ser 55
  - Filas: python3 con pyarrow sumando `num_rows` de cada parquet (deben ser ~1.1M).
  - `squeue -u a474r867 -n pmc-ingest | wc -l` → van bajando; al terminar, 0.
- Cuando acabe: el pipeline siguiente es merge del parquet → corpus (NO hacerlo sin instrucción).

### 6. Watchdog teacher (en reumanlab, no en cluster)
- Proceso local: `pgrep -af watchdog_v4teacher` (si está muerto, **reportarlo**, no relanzarlo sin OK).
- Log local: `/tmp/watchdog_v4.log`
- Este watchdog vigila el teacher v4 y la destilación; si el teacher rota, reporta el nuevo endpoint.

## Umbrales de ALERTA (reportar si...)
- Un job de entrenamiento (mdlm-retrain-v3, moe-E2v2) muere y NO reaparece en 12h.
- El teacher v4 no responde durante >30 min tras un cambio de job (el cron relanza ~6h, puede tardar).
- pmc-ingest termina con **menos de 55 parquet** o **< 1.0M filas** acumuladas.
- effN del router en logs del E2 v2 < 2 al final (colapso severo).
- Cualquier ERROR/traceback en logs/latest.

## Lo que NO hacer
- NO `scancel`, NO `sbatch` (salvo instrucción explícita).
- NO modificar scripts ni datos.
- NO abrir túneles ni puertos.
- NO tocar el repo (solo leer).

## Formato de reporte semanal/diario
```
EcoReasoner status <fecha>:
- Fase A: COMPLETA @g8000 / pendiente
- E2 v2: loss .593, effN 2.49, e2_final.pt OK
- Destilación: r3=49 tray, job=ACTIVO/PARADO
- Teachers: v4 OK(nodo:p), qwen36 x2 OK, glm OK
- PMC: 55/55 parquet, 1.1M filas, 0 jobs RUNNING (o N RUNNING)
- Watchdog: vivo/muerto
- Alertas: [ninguna | ...]
```