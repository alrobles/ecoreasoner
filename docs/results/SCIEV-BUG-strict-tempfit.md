# BUG sciev: `--allow-truncation` no alcanza el temp-fit de decisiones

**Issue**: https://github.com/alrobles/sciev-devel/issues/1
**Repo**: alrobles/sciev-devel · `sciev/eval.py`
**Encontrado**: 2026-09-27, corriendo `sci_eco_v5.slurm` (c_* sobre
backbone ecoreasoner MdLMMoE, seq_len 768)

## Síntoma

`sciev.eval --allow-truncation` crashea en el temp-fit, antes de llegar
al loop de evaluación:

```
ValueError: decision exceeds token budget and would require truncation
  File "sciev/eval.py", line 723, in main
    r2_temp = fit_r2_temperature_decisions(...)
  File "sciev/eval.py", line 200, in fit_r2_temperature_decisions
    logits, prepared = decision_logits(..., strict=strict)
  File "sciev/decisions.py", line 185, in prepare_decision
```

## Repro

```bash
python -m sciev.eval \
    --ckpt <ecoreasoner model.pt> --config eval-moe-v5.yaml \
    --r2-mode spanpool --r2-layers="-1,-4,-7,-10" \
    --canonical-order --allow-truncation \
    --r2-temp-fit-decisions data/sci_battery/systemone-v2/sci_choice_dev.jsonl \
    --decisions-eval data/sci_battery/systemone-v2/sci_choice_eval.jsonl
```

Cualquier backbone con `seq_len` corto (768) sobre archivos con
encoding actual → dev rows que exceden el presupuesto → excepción.

## Causa raíz

`--allow-truncation`/`--strict-inputs` (`args.strict_inputs`) se pasa al
loop de eval (`eval_decisions_ids(..., strict=args.strict_inputs)`,
~l.793) pero **no** al call del temp-fit (~l.722):

```python
r2_temp = fit_r2_temperature_decisions(
    model, head, dev_rows, args.device, mode=args.r2_mode,
    layers=layers, canonical=args.canonical_order)   # falta strict=
```

Con `strict=None` la función lo auto-deriva del encoding del archivo
(`== ENCODING_VERSION` → strict) — el flag del usuario nunca llega.

## Fix (ya aplicado en el checkout del cluster)

```diff
             r2_temp = fit_r2_temperature_decisions(
                 model, head, dev_rows, args.device, mode=args.r2_mode,
-                layers=layers, canonical=args.canonical_order)
+                layers=layers, canonical=args.canonical_order,
+                strict=args.strict_inputs)
```

`strict=None` default conserva el comportamiento anterior para runs que
no pasan el flag (LLaDA seq-4096). Verificado end-to-end: las 7 evals
de `sci_eco_v5` completan tras el fix.

## Mismo patrón, no verificado

`fit_r2_temperature` (variante pairs, ~l.713) tampoco recibe `strict` —
si su consumidor también puede truncar, mismo tratamiento.

## No-bug relacionado

La validación "option encodings must be distinct" de `data.py` rechaza
la batería v1 (`sci_battery/sci_*_train.jsonl`). Es intencional; solo
implica que los runs eb_* (v1) no son reproducibles tal cual — usar
`systemone-v2` para todo lo nuevo.
