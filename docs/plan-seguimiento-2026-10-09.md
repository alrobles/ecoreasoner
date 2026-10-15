# Plan de seguimiento — EcoReasoner dLLM (2026-10-09)

Horizonte: ~2 semanas. Objetivo: cerrar la fase v5-1b (1.3M) con la
curva completa, el 4º punto sciev y la cadena UNAM→QA→SFT-v3 disparada.
Base: PROGRESS-REPORT-2026-10-09 + estrategia-escala-2026-10-06.

Convención: **[AUTO]** = corre solo (watcher/slurm), **[MANUAL]** =
requiere acción, **[DECISIÓN]** = requiere a Angel.

---

## 0. Blindaje de automatización (hecho 10-09)

- [x] `watch_v5_battery.sh` relanzado 23:41 — batería g992501 en vuelo.
- [x] `watchdog_ecoreasoner.sh` nuevo + cron `*/15` + `@reboot`:
  relanza los 3 watchers si mueren (respeta flags DONE). El hueco
  g841K-g975K no se repite. Log: `data/watchdog.log`.

## 1. Línea de tiempo esperada

| cuándo | evento | tipo |
|---|---|---|
| ~cada 25K steps (~8h) | batería automática (dense+cons_inf+ctxnec+remask) | AUTO |
| 10-10 ~08:00 | batería g992501 completa → nuevo punto de curva | AUTO |
| ~10-13/14 | **g1.3M alcanzado** → batería final automática sobre ckpt terminal | AUTO* |
| ~10-12/14 | UNAM-EN pendientes=0 → `UNAM_EN_MT_DONE.flag` | AUTO |
| +minutos tras flag | merge `unam_en.jsonl` → flota `qagen_mt` (0-15) → finalize → `UNAM_QA_SFT_READY.flag` + `sft_ids_v3_unam.npz` | AUTO |
| post-1.3M | sciev 4º punto sobre ckpt terminal (EMA+raw) | MANUAL |
| post-1.3M | lanzar SFT-v3 (`sft_mdlm_v5.slurm` INIT=ckpt terminal, data=sft_ids_v3_unam.npz) | DECISIÓN+MANUAL |
| libre | spike MI210 (`smoke_mi210.slurm`) — pool ~81 GPUs sin usar | MANUAL |
| libre | auditoría `entity` n=1 en subtipos (barata) | MANUAL |

*El watcher evalúa el ckpt más reciente con gap ≥25K — cuando el run se
detiene en 1.3M, la batería del terminal dispara sola **si el watcher
sigue vivo** (ahora cubierto por el watchdog en cron).

## 2. Qué vigilar y cómo (1 comando cada una)

```bash
# curva + baterías (local)
tail -5 /home/reumanlab/ecoreasoner/data/watch_v5_battery.log

# traducción UNAM (local)
tail -3 /home/reumanlab/ecoreasoner/data/watch_mtolmo.log

# cadena QA→SFT (local)
tail -5 /home/reumanlab/ecoreasoner/data/watch_unam_qa.log
ls /home/reumanlab/ecoreasoner/data/*.flag

# mainline vivo (cluster)
ssh kuhpc 'cat /beegfs/a474r867/ecoreasoner/runs/v5-1b/state.json'
ssh kuhpc 'squeue -u a474r867 -n v5-1b -h'
```

Cadencia sugerida: 1 chequeo/día de los logs locales (10 s cada uno).
Si `state.json` no avanza en >6h → la cadena AUTO_RESUBMIT murió:
relanzar ola con `sbatch scripts/v5_1b_ddp.slurm` (INIT=último ckpt).

## 3. Contingencias

| señal | causa probable | acción |
|---|---|---|
| `watch_*.log` mudo >2h | watcher muerto | el watchdog lo revive en ≤15 min; si persiste, relanzar a mano |
| `state.json` viejo >6h | sin ola viva (sixhour lleno) | re-sbatch manual; considerar pedir prioridad o esperar ventana |
| pendientes mtolmo estancados | tasks mueren sin completar | watcher relanza array; revisar `logs/mt_olmo_*.err` |
| batería no dispara post-1.3M | gap <25K al último eval | `ssh kuhpc 'cd /beegfs/.../ecoreasoner && bash scripts/battery_point.sh 1300000'` manual |
| nuevo apagón beegfs | infra | las flags son idempotentes; al volver, todo reanuda solo |

## 4. Decisiones que quedan para Angel

1. **SFT-v3**: ¿lanzar sobre ckpt terminal apenas caiga
   `UNAM_QA_SFT_READY.flag`, o esperar? (la cadena llega hasta el npz;
   el sbatch de SFT es manual por diseño).
2. **Integración UNAM-EN al corpus**: ¿dosis post-1.3M (continuar
   pretrain con unam_en mezclado) o solo vía QA→SFT? Abierto desde 10-04.
3. **Competencia sixhour**: la flota `v6_*` (xsdm) bajó el ritmo del
   mainline a ~55-75K/día. ¿Se acepta ETA ~10-13/14 o se modula?
4. **MI210**: tras el smoke, ¿segundo backbone big-batch en pools
   lentos (q6000/v100) o isla MI210 extra para v5?
   **DECIDIDO 10-15**: segundo backbone en pool MI210 — **v6-2b**, brazo
   de capacidad: misma receta/corpus/batch que v5-1b, única variable
   `--layers 12→24` (~570M act / ~2.1B totales, 2×). Hipótesis bajo
   prueba: el cuello generativo (gen_exact ~0.30, chat degenerado) es
   de capacidad — si v6-2b mueve gen_exact/remask sobre la curva
   comparable, confirmado. Script `scripts/v6_2b_mi210.slurm` +
   `eval-moe-v6.yaml` + `watch_v6_battery.sh` (watcher propio).
5. **Papers**: P8 (systems+harness+falsaciones) ya tiene todo el
   material; P9 espera el ckpt terminal. ¿Se empieza a escribir P8 ya?

## 5. Definición de "fase cerrada"

- [x] Curva g207K→g1.3M con batería terminal consolidada (g1300000: ctxnec 0.752)
- [ ] sciev 4º punto (trayectoria backbone→decisión completa: 4 pts) — coordinado vía nota en sciev-code + addendum set comparable
- [ ] UNAM_QA_SFT_READY.flag + SFT-v3 evaluado (sonda paramétrica final) — flag ya escrito; falta lanzar/evaluar SFT-v3
- [ ] **Brazo capacidad v6-2b en MI210** — curva comparable contra v5-1b; veredicto "capacidad vs receta" sobre gen_exact
- [ ] Veredicto final tesis: exposición → deriva lenta hasta el terminal
  (binding 0.752, generación ~0.30, chat=no, 2 falsaciones
  estructurales + SFT como sonda)

## 6. Coordinación sciev (sesiones Devin paralelas — añadido 10-11)

sciev se desarrolla en paralelo desde **otra máquina** (home
`alrobles`, sesiones `zenith-century`, `pear-abrosaurus`, `childlike-toaster`, …).
Canónico confirmado en `/beegfs/a474r867/sciev-code/AGENTS.md`:

- **Raíz de datos/runs**: `$RJ=/beegfs/a474r867/sciev`
  (`runs/sci/`, `heads/`, `data/`, `evals/`, `serve/`, `logs/`).
- **Checkout de código**: `/beegfs/a474r867/sciev-code` — clone de
  sciev-devel, rama `alignment/scientific-pipeline`; `$RJ/sciev` es
  copia rsync, re-sync tras cada bump. Migración a scratch REVERTIDA
  — beegfs es canónico (incidente 10-08 cerrado).
- **Desarrollo activo hoy**: `gpqa_retrieved` (brazo de retrieval
  real para GPQA grounded, fase B3 — commits/jobs en curso) y
  `antileak_frac` (rebalanceo de control — `sciev_antileak_alp*_b`
  COMPLETED 10-11 01:26).
- **Servicio**: `sciev.ecoseek.org` = Cloudflare → sciev-gate
  (reumanlab:8682) → uvicorn en q6000; watchdog de túnel en ESTA
  máquina (`~/bin/sciev-tunnel-watchdog.sh`, cron */5). Serve job =
  `sbatch ~/sciev_submit/sciev_serve.slurm` — suyo, no tocar.

**Regla anti-colisión para esta sesión**: NO lanzar jobs `sciev_*`
ni escribir bajo `/beegfs/a474r867/sciev*` mientras el harness esté
en desarrollo activo — el 4º punto sciev (sobre el ckpt terminal)
se coordina con la otra sesión o se lanza solo tras confirmar que
`sciev-code` está estable (HEAD quieto, sin jobs `sciev_*` en cola,
y la versión del eval comparable a la de los 3 puntos anteriores).
El resto del pipeline ecoreasoner (mainline, baterías, UNAM→QA)
vive íntegramente bajo `/beegfs/a474r867/ecoreasoner` + watchers
locales — sin solape con sciev.
