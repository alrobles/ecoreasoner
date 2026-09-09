# F2 — Batería L0-L3 de hard negatives: resultado final

Fecha: 2026-09-08 · Rama: `devin/eval-battery-l0l3` (resultado integrado en `main`).

## Hipótesis y lectura pre-registrada

La batería L0-L3 fue diseñada para descomponer qué aprendió realmente el
checkpoint final de F2 (`pairwise_acc 0.5273`, n=256):

| Nivel | Qué mide | Lectura si es significativo |
|---|---|---|
| L0 | Coherencia temática: contexto de **otro** documento, mismas palabras clave | atajo tópico/lexical |
| L1 | Tópico controlado: contexto del **mismo dominio**, doc diferente | atajo tópico refinado |
| L2 | Orden de etapa: contexto del **mismo documento**, etapas permutadas | aprendizaje de orden estructural sin payload |
| L3 | Inferencia de contenido: contexto del mismo doc, **payload mutado** | señal inferencial real |

## Resultados

| Nivel | pairwise_acc | p (binomial exacta, 1-sided) | Significativo | Interpretación |
|---|---|---|---|---|
| L0 | 0.5738 | 0.0013 | ** | Solo coherencia temática / NSP / overlap |
| L1 | 0.5607 | 0.0068 | ** | Tópico controlado; sigue siendo atajo de dominio |
| L2 | 0.465  | 0.93 | ns | Incluso bajo azar: no aprendió el **rol** de las etapas |
| L3 | 0.5089 | 0.39 | ns | No hay señal inferencial de contenido |

* n ≈ 256 pares por nivel, seed 7331, evaluación determinista, sin generación.
* L2 < 0.5 confirma que el modelo no solo no aprendió orden estructural,
  sino que lo que aprendió (coherencia temática) lo empuja a elegir
  permutaciones incorrectas.

## Conclusión

**F2 queda falsificado de forma fuerte y clara.** El checkpoint final (y el
pico 0.566 observado durante el entrenamiento) se explican por un único
atajo: **coherencia temática / overlap léxico**, es decir, el modelo
identifica qué continuación *suena* al mismo tema sin acceder al rol
estructural ni al contenido inferencial.

- L0 y L1 significativas → el atajo es robusto y transfiere entre documentos.
- L2 no significativa (bajo azar) → el atajo no contiene información de
  orden/rol.
- L3 no significativa → el atajo no contiene información de payload causal.

El pico 0.566 fue, como se sospechaba en la auditoría, una combinación de
ese atajo con **múltiples miradas** (69 evaluaciones a lo largo del run).

## Consecuencia experimental

- La línea F2 (esqueleto puro con span masking a 155M) se **archiva como
  falsificada**.
- No tiene sentido relanzarla a más steps ni escalarla antes de aislar un
  mecanismo real.
- Vía libre al **controller/verificator** y/o a la línea **prosa-dllm**
  (Fase B), pero esta última debe ser juzgada con el mismo criterio
  pre-registrado: ≥0.55 pairwise transitorio **y** L3 significativo en la
  batería L0-L3.
