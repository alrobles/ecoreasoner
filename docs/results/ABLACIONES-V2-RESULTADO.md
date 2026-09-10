# ABLACIONES PILOTO v2 — f0-span-v2 — RESULTADO CONSOLIDADO (2026-09-10)

Fecha: 2026-09-10 07:25 CDT · 4 runs × 10K steps en pro6000 · todos con
battery L0-L3 automática y battery_verdict.json
· Jobs: no-whole-stage 29000908 · weight-tying 29000909 · rope 29000910 · 50m 29000911
· Corpus: train_ids_skeleton_v2.npz (CORREGIDO, eos_id=0) · pairs_hard_v2

## Tabla de veredictos

| Ablación | Modelo | L0 | L1 | L2 | L3 | Veredicto |
|---|---|---|---|---|---|---|
| (piloto, whole_stage=1) | 155M dense | 0.5039 ns | 0.4724 ns | **0.6118 \*\*\*** | 0.5065 ns | STAGE_GRAMMAR |
| **no-whole-stage** (WHOLE_STAGE=0) | 155M | 0.5490 \* | 0.5079 ns | **0.5961 \*\*\*** | 0.4848 ns | STAGE_GRAMMAR |
| **weight-tying** | 155M tied | 0.5020 ns | 0.4764 ns | **0.6294 \*\*\*** | 0.5022 ns | STAGE_GRAMMAR |
| **rope** | 155M RoPE | 0.5392 \* | 0.5020 ns | **0.5961 \*\*\*** | 0.4696 ns | STAGE_GRAMMAR |
| **50m** (384/6/6+tying) | ~60M | 0.5020 ns | 0.4626 ns | **0.5569 \*\*** | 0.4826 ns | STAGE_GRAMMAR |

(piloto: job 28998750, ya reportado como STAGE_GRAMMAR el 09-09)

## Lectura

Las 4 ablaciones confirman el patrón del piloto:

1. **L2 siempre significativo** (0.557-0.629, ** / \*\*\*): TODAS las variantes
   aprenden el orden/rol de etapa del esqueleto IMRaD. La señal estructural es
   ROBUSTA a: quitar whole-stage, weight-tying, RoPE, y reducir el modelo a 50M.
2. **L3 nunca significativo** (0.47-0.51 ns): NINGUNA aprende inferencia de
   contenido. La línea dLLM puro NO escapa a la gramática-estructural-sin-
   inferencia con esta receta.
3. L0/L1 alrededor del azar (0.46-0.55): el modelo no discrimina por dominio/
   tópico — la coherencia temática no emerge.

## Conclusiones

- **el whole-stage NO es el ingrediente responsable de L2**: no-whole-stage
  conserva 0.596 (≈ piloto 0.612). La señal de orden de etapa viene del
  span-mask sobre esqueleto en sí.
- **weight-tying da el L2 más alto** (0.6294): mejora marginal, no cambia el
  veredicto cualitativo.
- **El tamaño 50M no es limitante para L2** (0.557 con 2.6× menos parámetros):
  la ausencia de L3 no se explica por capacidad pequeña.
- **Línea dLLM-puro con máscara aletoria/span + esqueleto queda CONFIRMADA como
  STAGE_GRAMMAR** — limitada a estructura sin inferencia. La vía productiva
  sigue siendo el controller/verificator (Opción D, ya validada 98.6%).

## Decisión pendiente para el usuario/Lab

- (A) Archivar la línea dLLM-puro como falsada en inferencia (L3) con evidencia
  ABLACIÓN completa (este doc = material publicable negativo: "multip el
  masking estructural enseña estructura, no contenido").
- (B) Explorar objetivo contrastivo L3 explícito (ultimo item de §13.5) si se
  quiere insistir en la vía dLLM (costo extra en pro6000).
- (C) Cerrar la vía dLLM para el producto y focalizar en controller/verificator
  (ya 98.2-98.6% en replicación) + entrenar el dLLM solo como fluidez/knowledge
  auxiliar.

## Artefactos

- docs/results/ablaciones_v2/battery_verdict_*.json (4 veredictos)
- Runs HPC: /beegfs/a474r867/ecoreasoner/runs/f0-span-v2-{no-whole-stage,weight-tying,rope,50m}/
- Este doc