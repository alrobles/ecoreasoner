# F2-SPANES — RESULTADO (2026-09-08) — FALSIFY

## Veredicto oficial (verdict_f2.py, 69 puntos, decisiones por umbrales pre-registrados)

- **VERDICT: FALSIFY** — pairwise_acc final 0.5273 (< falsify_threshold 0.55)
- last_step: 49,301 | start_ref: 0.535 (receta micro ganadora del sweep)
- slope (últimos 3): **-6e-06** (negativo, sin pendiente positiva al final)
- delta (primeros 3 → últimos 3): +0.0546 (mejora real desde ~0.47, pero sin cruzar GO)
- hit_threshold: 0.60 (no alcanzado) | EXTEND habría requerido >0.535 con slope+

## Timeline del run

| Job | Estado | Detalle |
|---|---|---|
| 28962112 | TIMEOUT 5:50 @ 88% | Murió por deadlock SIGUSR1↔save (auditoría 1.1, confirmado en prod) en step 44,051/50,000; sin flag; cadena muerta sin resubmit |
| 28982555 | **COMPLETED** 00:47:38 | Relanzado con fix anti-reentrada activo; resume() desde step 44,151 → 50,000/50,000; flag COMPLETE 21:34; report.json escrito (pairwise_acc 0.5273) |

## La curva (decisiones; n=256 pares, seed 7331, eval determinista, --no-gen)

- 69 checkpoints evaluados por el watch (dedupe por step, filas error ignoradas).
- Arrancó 0.469 → subió a picos 0.5664 (28,351) / 0.5586 (30,951) → **decayó al final** a 0.52-0.54 (últimos 10 puntos: 0.5195-0.5508, cerrando 0.5273).
- mean_delta se mantuvo positivo todo el run (+0.04 a +0.08) — el modelo SÍ discrimina mejor que azar a nivel de loss, pero no lo suficiente para el criterio.

## Lectura crítica (la parte que importa)

1. **El pico 0.5664 NO es señal — es el problema de comparaciones múltiples** que la
   auditoría Devin señaló (§2a): n=256 → σ≈3.1 p.p., el máximo observado sobre decenas
   de miradas se infla. El punto FINAL (0.5273, a ~0.9σ del azar) es la decisión.
2. **la mejora temprana (0.47→0.55 en ~28K steps) y la caída posterior** sugieren que
   el modelo aprendió un atajo local (coherencia posicional/estilística del esqueleto,
   análogo al copy-vecino de bw3 un nivel arriba) que satura ~0.53-0.55 y no escala a
   discriminación inferencial real. La loss del entrenamiento siguió bajando (6.9-7.2,
   estable) mientras la discriminación se estancaba — pérdida NO es proxy (confirmado
   de nuevo, ahora en la dirección contraria a bw3).
3. **El GO 0.55 no se sostuvo**: los valores ≥0.55 en 28-36K fueron transitorios; el
   modelo no consolidó. La zona 0.52-0.54 es su techo con 50K steps de esta receta.
4. **La receta micro ganadora (span64@15, esqueleto, 12,288 tok/update) a escala
   ×5 pasos NO produjo HIT** — la promesa del puente ("si despega de 0.535 a 0.6+")
   queda falsada. La pregunta de diseño queda: la discriminación de esqueletos a este
   tamaño (155M) no surge de más steps con el mismo objetivo.

## Postmortem L0-L3 hard negatives (ver `F2-L0L3-BATTERY-RESULTADO.md`)

La batería L0-L3 confirma la falsación: el modelo solo aprendió
**coherencia temática / overlap léxico** (L0 0.5738**, L1 0.5607**) y
falla tanto en orden de etapa (L2 0.465 ns) como en inferencia de
contenido (L3 0.5089 ns). El pico 0.566 fue atajo + múltiples miradas.

## Decisión (ROADMAP pre-registrada)

- **Archivar la línea f2 como falsada** con este documento.
- **Siguiente: arquitectura controller/verificator** (deepseek/glm genera + dLLM como
  conocimiento/repair — opción D del veredicto A/B) — NO relanzar f2 a 100K.
- El dLLM propio queda como línea de fluidez a 155M (ruta Devin) SOLO si se re-formula
  el gate de falsación (métricas word/rep4/uniq engañables — ver crítica a la ruta).

## Legado técnico (validado en producción)

- Deadlock SIGUSR1↔save: **confirmado y corregido** (flag SAVING anti-reentrada,
  validado con test EAGAIN). Los runs de decenas de olas son ahora posibles.
- sort -V para elección de ckpt (antes elegía g99xxx vs g100000) + --index absoluto:
  desplegados; el eval final de f2 (g50000... g49301 por retention) usó el fix.
- strict=True en suite_smoke; tmp con PID en eval_curve; warmup REAL (era arg muerto).
- verdict_f2.py: arreglado SyntaxError f-string (Python <3.12) — ahora corre.

## Artefactos

- `runs/f2-spanes/eval_curve.jsonl` (curva completa, 69 pts)
- `runs/f2-spanes/report.json` (eval final del job: pairwise_acc 0.5273, mean_delta 0.05965)
- `runs/f2-spanes/verdict_f2.json` (veredicto oficial)
- `runs/f2-spanes/checkpoint-g50000` + g49301 (retention-2, últimos)
- Este doc: `docs/results/F2-SPANES-RESULTADO.md`


---

## Anexo — SANITY L0-L3 (post-FALSIFIC, batería Devin PR #1, 2026-09-08)

**Pregunta**: ¿hay señal inferencial escondida o falsación fuerte? Negativos
graduados sobre el ckpt final g50000 (L0=427, L1=428, L2=428, L3=338 pares,
seed 7331, --no-gen):

| Nivel | bad = | acc | IC95 Wilson | p (1-cola) | sig |
|---|---|---|---|---|---|
| L0 | otro doc, cualquier dominio (coherencia) | 0.5738 | [0.526, 0.620] | 0.0013 | ** |
| L1 | otro doc, MISMO dominio (tópico controlado) | 0.5607 | [0.513, 0.607] | 0.0068 | ** |
| L2 | misma doc, etapa j≠k (orden/rol) | 0.4650 | [0.418, 0.512] | 0.933 | ns |
| L3 | misma etapa, payload mutado (inferencia) | 0.5089 | [0.456, 0.562] | 0.393 | ns |

**Lectura (pre-registrada)**: solo L0/L1 significativos → **FALSIFICACIÓN FUERTE**:
el atajo temático (overlap ctx↔ok) explica toda la señal previa; sin inferencia de
orden (L2, incluso bajo azar: el bad más solapado "gana") ni de contenido (L3 a
azar). El pico 0.566 del F2 era el atajo + comparaciones múltiples (auditoría 2a).
Vía libre al **controller/verificator**. Diagnóstico previo (pairs.jsonl original):
jac(ctx,ok)=0.1825 vs jac(ctx,bad)=0.0713 (margen +0.117) → el atajo existía de fábrica.

Fix del PR aplicados (main): trust_remote_code en build_pairs_hard (sin él → 0 pares
silencioso), _select_k a k=2 (corpus real es OBS/EVID/CONC sin PREDICCION), verdict.py
leyendo discrimination anidado. Artefactos: runs/pairs_hard/*, runs/f2-spanes/
battery_L0L3/*, runs/f2-spanes/battery_verdict.json.
