# V6-2B — crash-loop RCCL/SHM en MI210 y migración a NVIDIA (2026-10-22)

Estado auditado desde `reumanlab` (post cambio de máquina; antes
`reumanlab-terminal`). Resumen ejecutivo y evidencia.

## Lo que pasó

- v6-2b-mi210 (brazo de capacidad, L=24, lanzado 10-15) entró en
  **crash-loop desde la 2ª ola (10-16)**: ~45+ olas FAILED consecutivas
  en ≥7 nodos MI210 distintos (r06r26, r07r06, r08r10/16/18/20/26/28).
- Progreso real en 7 días: **g12501** (+188 intento incompleto en
  `checkpoint-g12689`, un .tmp de 0B — borrado 10-22).
- Solo ~4 olas completaron su ventana 5:50h en toda la semana; el resto
  murió a los 11min–4h. Throughput efectivo ~500 pasos/4 días →
  inviable para g1.3M.

## Diagnóstico (evidencia en nodo, 10-22)

- Job zombie 32244245 (r06r26n01): 2h+ sin un solo `step` log.
  - `py-spy` rank2: colgado en **`DDP.__init__ →
    _verify_param_shape_across_processes`** — la PRIMERA colectiva.
  - Rank1: `kfd_wait_on_events` (espera evento GPU), rchar congelado.
  - `dmesg`: `amdgpu: Freeing queue vital buffer, queue evicted`
    repitiéndose ~c/2min; kworkers `mm_percpu_wq`/`events_freezable`/
    `cgroup_destroy` a 40-99% CPU por horas.
- Otras olas: `Watchdog caught collective operation timeout:
  WorkNCCL(ALLREDUCE, 7.08M elems) 600082ms` → SIGABRT (30 logs .err
  con la firma, desde 10-16).
- **Conclusión**: deadlock aleatorio de colectivas RCCL sobre
  transporte SHM (`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` — única vía
  válida en esta fabric, verificado en spike). No es nodo, ni ckpt
  (g12501 íntegro), ni batch malo: es inestabilidad RCCL/MI210 con
  este workload. Descartado: P2P/XGMI ya falló antes.

## Acción tomada

- `scancel 32244245` (zombie, 0 pasos en 2h).
- **v6-2b portado a NVIDIA**: `scripts/v6_2b_nv.slurm` (clon de
  `v5_1b_ddp.slurm` con `--layers 24`), deployed a beegfs.
  - Constraint `a100|l40|a40|q8000|pro6000` (q6000/24GB excluido:
    costos fijos ~41GB > 24GB).
  - Tiers VRAM: ≥60G → bs4 acc1 +EMA-GPU; 40–60G → bs2 acc2 +zero1
    +grad_ckpt +EMA-GPU (~42GB); <40G → fallback bs1 acc4 (defensivo).
  - Mismo `runs/v6-2b`, resume g12501 (formato ckpt cross-stack ya
    validado NVIDIA↔ROCm). Batch receta 9216 tok/step preservado.
  - 1ª ola: job 32248027.
- `watchdog_ecoreasoner.sh`: mainline_guard v6-2b apunta ahora a
  `v6_2b_nv.slurm` (job name `v6-2b-nv`, ntasks/gres estilo v5).

## SFT-v3 — eval chain reparada

- Regresión ctxnec fallaba en bucle: watcher lanzaba `c-eval` SIN
  constraint → caía en nodos sin NVIDIA (`no NVIDIA driver`), el bug
  ya documentado 10-15 en battery_point.sh. 5 fallos seguidos.
- Fix: constraint `q6000|a100|l40|a40|q8000|pro6000` + `--gres=gpu:1`
  en `watch_sft_v3.sh` (+ mismo fix heredado en `watch_sft_v4.sh`).
- Guard anti-blip beegfs nuevo: si `$OUT` existe pero `ls` vino vacío
  → ciclo saltado (el 10-20 un blip provocó resubmits OFFSET=0
  fantasma; no corrieron, pero el bug queda tapado).
- Regresión relanzada a mano: job 32247901 → al escribir
  `runs/contrast/pairs_l3_ctxnec_sft-v3/logicdiff_summary.json` el
  watcher escribe `SFT_V3_DONE.flag`.
- **Bootstrap sft-v4 automatizado** en watchdog: al caer SFT_V3_DONE
  siembra `data/sft_v4_init.txt` con el ckpt terminal de sft-v3 y
  lanza `watch_sft_v4.sh` con `TARGET_GN=16000` (~2 épocas de
  `sft_ids_v4_mt.npz`, 66K docs). Aprobado por Angel 10-22.

## Lecciones / pendientes

- MI210/ROCm: el pool queda **sospechoso para workloads DDP** — si se
  reintenta, probar NRANKS=1 (sin colectivas) o NCCL_DEBUG forense.
  Dato del spike: 1×MI210 bs12 = 24K tok/s > 3-rank DDP 13.8K tok/s
  (las colectivas ya comían la mitad del throughput).
- v6-2b curva comparable: señal esperada (gen_exact vs curva v5-1b)
  legible a partir de ~g100-300K; no hace falta el 1.3M completo.
- `watch_v6_battery.sh` sigue vivo: disparará baterías c/25K pasos
  sobre los ckpts nuevos (eval-moe-v6.yaml, constraint NVIDIA en
  battery_point.sh ya existía).
- sciev: sin respuesta en la nota de coordinación; punto #4 sigue en
  espera (harness en dev activo — no lanzar).
