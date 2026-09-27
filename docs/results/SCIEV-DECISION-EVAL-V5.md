# Evaluación de decisión tipada — v5-1b como backbone (sciev/c_*)

**Fecha**: 2026-09-27 · **Backbone**: v5-1b EMA @g316001 (MdLMMoE 1.13B,
12L/768/12H/16E top-1, snapshot `runs/v5-1b/battery_curve/ckpts/g316001/`)
**Instrumento**: receta c_* de sciev — heads por tipo (AttnPool choice,
MLP noul/score + ordinal), canonical ordering, backbone congelado,
batería `sci_battery/systemone-v2`. Job: 30542606
(`sciev/scripts/sci_eco_v5.slurm`).

Punto de la trayectoria backbone→decisión: `eb_*` = piso (bw1_sr@g20,
prácticamente sin entrenar); `c_*` = techo de referencia (LLaDA-8B).
Nota de comparabilidad: eb_* y c_* corrieron sobre la batería v1
(n distinto); la lectura eb↔ev5 es aproximada, c_* es el techo.

## Resultados (flip_rate = 0.0 en todo — canonical exacto)

| eval | ev5 (v5-1b) | eb piso | c_* LLaDA-8B |
|---|---:|---:|---:|
| choice elite (n=448) | **0.473** | 0.280 | 0.870 |
| noul elite (n=1200) | 0.667 | 0.667 | 0.783 |
| score elite (n=1218) | 0.329 | 0.334 | 0.608 |
| GPQA main (n=441) | 0.256 | 0.266 | ~0.315 |
| GPQA diamond (n=193) | 0.275 | 0.283 | ~0.33 |
| SciFact noul (n=332) | 0.593 | 0.594 | 0.853 |
| SciFact score (n=304) | 0.286 | 0.30 | 0.624 |

Auto@5%err ≈ 0 en todo (aún no desplegable); ECE bajo por confianza
sistemáticamente baja (modelo inseguro, no calibrado-bueno).

## Lectura

1. **La ganancia existe y se concentra en `choice`**: 0.28 piso → 0.47
   v5-1b → 0.87 LLaDA. El backbone ya extrae ~55% del headroom de
   decisión choice in-distribution. Consistente con los hallazgos
   internos (gen_exact_ok subiendo lento — el modelo SÍ aprende a
   producir contenido forzado, despacio).
2. **`noul` saturado por prior trivial**: 0.667 = idéntico al piso → el
   tipo se resuelve con un sesgo mayoritario que no usa el backbone.
   LLaDA lo rompe (0.78); v5-1b aún no aporta.
3. **`score`/`GPQA`/`SciFact` ~ piso**: conocimiento paramétrico y
   evaluación ordinal no emergen todavía — el "gap del backbone" del
   doc reverse-jev se confirma también para nuestro desde-scratch.
4. **Coherencia con el cuadro interno**: las decisiones tipadas exigen
   exactamente los dos cuellos medidos adentro — binding de contexto
   (C4: ~9pts de 0.71) y conocimiento paramétrico (plateau de
   plausibilidad 0.65). La evaluación externa converge con la interna:
   capability parcial, creciente, no saturada.

## Caveats

- **Truncación**: seq_len 768 del backbone vs decisiones largas —
  `--allow-truncation` (ctx→≤384 tok). LLaDA evalúa sin truncar
  (seq 4096): asimetría inherente que favorece al techo. Reportado
  como `truncation.context_rows` en los JSON de eval.
- **Batería v2** (systemone-v2): estricta — el set v1 que usaron eb_*
  y los primeros c_* tiene duplicados de opciones que hoy no pasan
  validación.
- **Bugfix aplicado en sciev** (repo paralelo, no-git en cluster):
  `sciev/eval.py` — el temp-fit de decisiones no propagaba
  `strict`; añadido `strict=args.strict_inputs` al call
  (fit_r2_temperature_decisions). Sin el fix, --allow-truncation no
  cubre el temp-fit y la eval muere en dev. Pendiente portarlo al
  checkout principal de sciev-devel.
- **EMA wrapper**: `ema_as_model.pt` = `{"model": ema_sd, "model_config"}`
  creado junto al snapshot (load_backbone espera la llave "model").

## Próximos puntos de la trayectoria

- Re-eval con ckpts v5-1b posteriores (la curva sigue corriendo —
  próximo snapshot automático ~g341K): decision-acc vs exposición es
  la curva que el doc reverse-jev propone como probe de madurez.
- Brazo sft-v1 (`runs/sft-v1/checkpoint-g32000`, model.pt directo):
  ¿SFT mueve decisiones tipadas? `TAG=es5 CKPT=... sbatch sci_eco_v5.slurm`.
- Si interesa headroom inmediato: DAPT-LoRA sobre LLaDA ya existe en
  sciev (`runs/dapt/lora-final`) para aislar "dominio del corpus" vs
  "madurez del backbone".
