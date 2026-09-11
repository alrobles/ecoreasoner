# Auditoría V3.3 contrastivo — CONTAMINACIÓN train/eval

> Fecha: 2026-09-10 23:55.
> Estado: V3.3 CORRIENDO (job 29184152, relanzado tras fix de seed 6c49e2a;
> primer run 29183112 FAILED 18:57). V3.2 role-aware TAMBIÉN lanzado
> (29184149) — ver §5.
> Auditor: Hermes (revisión de código + logs + datos en disco).

## 1. HALLAZGO CRÍTICO — CONTAMINACIÓN TRAIN/EVAL TOTAL

El trainer `scripts/train_mdlm_moe_v3_contrastive.py`:

- Entrena sobre `--pairs .../pairs_hard_v3/pairs_L3.jsonl` (466 pares).
- Define el "holdout" de monitoreo como `eval_pairs = pairs[:64]` y `train_pairs
  = pairs` (líneas 289-290). Los 64 de eval ESTÁN DENTRO de train. No hay
  separación real.
- La battery final (`suite_smoke_v2 --pairs-dir runs/pairs_hard_v3`) evalúa
  L3 contra **el mismo archivo** pairs_L3.jsonl con que se entrenó.

Consecuencia: **cualquier L3 alto de V3.3 demuestra memorización, no
generalización inferencial**. El criterio del doc (L3 ≥ 0.55 → GO) NO se puede
aplicar al run actual tal como está.

Evidencia en el log real (runs/f0-span-v3-contrastive/train.log):

```
step 2980 loss=1.4853 rank=0.0000 rank_acc=1.000 rank_ce=8.4357 mlm=7.4263
step 2990 loss=1.4760 rank=0.0000 rank_acc=1.000 rank_ce=7.6776
COMPLETO: 3000 steps
```

- `rank_acc=1.000` + `rank=0.0000` → el modelo distingue PERFECTO ok/bad en los
  466 pares. Con 154.8M params y 466 ejemplos, memorizar la pareja es trivial.
- `rank_ce=7.7-8.4` (CE media del sufijo) mientras que el piloto v2 base tenía
  CE ~6.9-7.3 en MLM: el fine-tune NO mejoró el modelado de contenido, solo
  memorizó la distinción.
- `mlm=7.4` (auxiliar) casi igual al base: el objetivo ranking domina y el
  modelo ignora la regularización MLM.

## 2. Bugs concretos en train_mdlm_moe_v3_contrastive.py

1. **Contaminación eval ⊆ train** (líneas 289-290): `eval_pairs = pairs[:64]`
   está dentro de `train_pairs = pairs`. Fix: split por contexto (ctx únicos)
   con seed, p.ej. 80/20 + `n_max` para equilibrar.
2. **mask_id inconsistente**: el script usa `mask_id = mcfg["vocab"]` (126080)
   pero el trainer v2 deriva MASK de `tok.vocab_size` (línea 992 de v2:
   `ARGS.vocab = tok.vocab_size`; Embedding con `vocab+1` para el slot MASK).
   El tokenizer real (LLaDA) puede no ser 126080 exacto → enmascara con un token
   de vocabulario VÁLIDO en lugar del slot MASK. Riesgo de OOB si vocab difiere.
   Fix: usar `tok.vocab_size` del tokenizer cargado, no el del YAML.
3. **La battery final NO usa un conjunto de eval independiente**: ver §1.
   Necesita `--pairs-dir` apuntando a pares generados con OTRA seed
   (pairs_hard_v3_eval), o al menos evaluar contra el holdout 20%.
4. **training_complete.flag dice "steps: N" en vez de "COMPLETE"** — los
   watchdogs de la casa esperan `COMPLETE`. Menor, pero rompe automatización.

## 3. Qué falta para que el GO/NO-GO sea válido

1. **Eval de generalización con pares NO vistos**: generar pairs_hard_v3_eval
   (otra seed, mismas proporciones/subtypes) y evaluar contra ESE dir, no el de
   train.
2. **Reportar train_acc vs eval_acc**: si train≈1.0 y eval≈0.5 (o sin señal),
   el veredicto automático debe ser MEMORIZACIÓN, no GO.
3. **Guard de "memorización temprana"**: si rank_acc llega a >0.95 antes de
   step ~500 y eval no sube de 0.55, parar y marcar el run como inválido
   (ahorra 2h de GPU en un resultado no interpretable).

## 4. Código REFERENCIADO (rutas)

- Trainer según: `scripts/train_mdlm_moe_v3_contrastive.py` (líneas 260-379
  main, 99-111 load_pairs, 114-153 collate_ranking, 224-257 ranking_loss).
- Slurm: `scripts/moe_v4_micro_v3_contrastive.slurm` (battery final con
  `--pairs-dir runs/pairs_hard_v3`).
- Config: `harness/configs/f0-span-v3-contrastive.yaml` (base ckpt del piloto
  v2, pairs_L3, alpha 0.2, 3000 steps).
- Pares: `runs/pairs_hard_v3/pairs_L3.jsonl` (466 pares, 466 ctx únicos —
  verificado 0 duplicados en disco).

## 5. ESTADO DE LANZAMIENTO (lo que encontré en vivo)

- V3.3: primer intento 29183112 FAILED (18:57) — probablemente por la seed
  faltante para suite_smoke_v2; relanzado como 29184152 (RUNNING, log hasta la
  carga de pares). El relanzado USA EL MISMO código contaminado.
- **V3.2 role-aware (29184149) ya lanzado** — pero el plan (doc, ruta única §4)
  lo condiciona a "si V3.3 es direccional/falla" (0.52 ≤ L3 < 0.55). Lanzarlo
  AHORA (con V3.3 en vuelo y SIN veredicto) adelanta un experimento que quizá
  no toque. Decisión pendiente: dejarlo correr (GPU de sobra, resultado extra)
  o cancelarlo y esperar el veredicto de V3.3.