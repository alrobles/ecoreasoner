# Roadmap dLLM v3 — De STAGE_GRAMMAR a inferencia ecológica

> Estado: 2026-09-10 · Tras piloto v2 + 4 ablaciones (STAGE_GRAMMAR: L2 0.557–0.629***/**, L3 < 0.51 ns).
> Objetivo: decidir si la vía dLLM puro puede aprender contenido inferencial con recursos realistas, o si pivoteamos a controller/verificator.

---

## 1. Diagnóstico del límite v2

El piloto v2 y las cuatro ablaciones muestran un patrón estable:

- **L2 (orden de etapa) siempre significativo**: 0.557–0.629.
- **L3 (inferencia de contenido) nunca significativo**: 0.47–0.51.
- **L0/L1 ~ azar**: no hay atajo temático.
- La variante con **weight-tying** alcanza el mejor L2 (0.629).
- La variante con **RoPE** tiene la menor loss de reconstrucción (~6.9) pero el peor L3 (0.47), lo que confirma que **menor loss ≠ mejor razonamiento**.
- El tamaño 50M mantiene L2 (0.557), por lo que la ausencia de L3 **no es puramente capacidad de parámetros**.

**Hipótesis del límite**: el objetivo MLM span sobre esqueletos enseña a reconstruir la **secuencia de etiquetas** (etapas), pero no fuerza al modelo a modelar las **relaciones causales/numéricas/semánticas** dentro y entre etapas. La señal de entrenamiento es demasiado local y forma-oriented.

---

## 2. Estrategia v3: cambiar la señal, no (solo) la escala

El plan v3 apunta a cuatro ejes ortogonales, en orden de costo:

| Eje | Idea | Costo | Prioridad |
|---|---|---|---|
| **A. Instrumento** | Batería L3 con hard negatives semánticos reales (dirección, mecanismo, variable, conectiva, entidad) | CPU | Inmediata |
| **B. Curriculum de masking** | Empezar con spans pequeños + baja máscara, escalar a spans largos + alta máscara | GPU ~4h | Alta |
| **C. Role/conective-aware masking** | Mascar con mayor probabilidad conectivas, relaciones causales, números y entidades | GPU ~4h | Alta |
| **D. Corpus eco-fino** | Entrenar con documentos de especies/regiones reales en lugar de esqueletos genéricos | CPU + GPU ~5h | Media |
| **E. Fine-tuning contrastivo** | SFT de ranking sobre pares L3 (GIFT/LIFT style) sobre el mejor checkpoint v2/v3 | GPU ~8h | Media-Baja |
| **F. Escala de steps** | 50K steps con la mejor configuración v3 | GPU ~20h | Baja (último recurso) |

No abandonamos dLLM, pero establecemos **puntos de parada duros**:
- Si **A+B+C** no logran L3 ≥ 0.55, el problema es la representación semántica y es poco probable que más steps o corpus lo resuelvan.
- Si **L3 cruza 0.55** en alguna variante v3, entonces justifica escalar a 50K y/o corpus eco-fino.

---

## 3. Hitos concretos

### V3.0 — Evaluar piloto v2 contra hard negatives v3
- Script: `harness/build_pairs_hard_v3.py` (listo, commiteado).
- Pares: `/beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3/`
- Slurm: `scripts/eval_v3_piloto.slurm`
- Entregable: `battery_v3_verdict.json` en el piloto.
- **Decisión**:
  - Si L3 v3 ≥ 0.55 con el modelo v2 actual → el problema era la batería; seguimos con v2 + batería v3.
  - Si L3 v3 sigue ~0.50 → el problema es el modelo; avanzar a V3.1.

### V3.1 — Curriculum de masking
- Modificar `pre_tokenize_v2.py` o crear `pre_tokenize_v3.py` que varíe `span_len` y `b_h` según el step global del entrenamiento.
- Config: `f0-span-v3-curriculum.yaml`
- Slurm: reutilizar `moe_v4_micro_v2.slurm` con variables de entorno para el curriculum.
- Esquema propuesto:
  - Steps 0–2K: `span_len=16`, `b_h=0.30` (aprender relaciones locales)
  - Steps 2K–5K: `span_len=32`, `b_h=0.60` (etapas parciales)
  - Steps 5K–10K: `span_len=64`, `b_h=0.95` (estructura global)
- Evaluar contra `pairs_hard_v3`.

### V3.2 — Role-aware / conective-aware masking
- Añadir pesos de masking por POS/rol: conectivas (`because`, `despite`), verbos de relación (`increases`, `reduces`), números, nombres de especies.
- El modelo debe reconstruir los tokens más informativos para la inferencia.
- Requiere un pre-proceso ligero (listas de palabras clave) sin NER pesado.

### V3.3 — Corpus eco-fino
- Usar `species_eco_fine_tagged.json` + `build_toolcalls_500.py` para construir esqueletos con contenido ecológico real.
- Pre-tokenizar y entrenar 10K steps con la mejor config (weight-tying + curriculum).

### V3.4 — Fine-tuning contrastivo L3
- Tomar el mejor checkpoint de V3.1/V3.2/V3.3.
- Entrenar con ranking de pares L3: minimizar log(1 + exp(neg - pos)).
- Ponderar tokens de conectivas/relaciones (GIFT/LIFT).

### V3.5 — Escala a 50K steps
- Solo si V3.1–V3.3 muestran L3 ≥ 0.52.
- Multi-ola con resubmit automático.

---

## 4. Criterios de decisión v3

| Resultado | L3 v3 | L2 | Veredicto | Acción |
|---|---|---|---|---|
| GO | ≥ 0.55 | ≥ 0.55 | Inferencia real | Escala a 50K / corpus eco-fino |
| DIRECCIONAL | 0.52–0.54 | ≥ 0.55 | Señal débil de inferencia | Iterar V3.2–V3.4 |
| STAGE_GRAMMAR | < 0.52 | ≥ 0.55 | Solo estructura | Considerar V3.4 o pivotar a controller/verificator |
| FALSIFY | < 0.52 | ≤ 0.52 | Nada | Pivotar a controller/verificator |

---

## 5. Soporte en literatura reciente

- **The Confidence Shortcut** (arXiv 2605.29123): random masking es más robusto para reasoning que masking alineado a confianza local.
- **LogicDiff** (arXiv 2603.26771): gran parte del déficit de razonamiento en MDMs es el orden de denoising, no las representaciones.
- **DreamReasoner-8B** (arXiv 2606.19257): curriculum de block-size mejora razonamiento en modelos de difusión.
- **Apollo** (arXiv 2212.09282): masking selectivo de POS de alto orden + pérdida de entailment/contradiction mejora razonamiento lógico.
- **MERIt** (ACL Findings 2022): hard negatives generados modificando relaciones en meta-paths.
- **SYNC** (IJCNLP 2023): hard negatives curriculares con perturbaciones estructurales.
- **GIFT / LIFT / d1 / d2** (2025–2026): SFT/RL para diffusion models con importancia-aware loss.
- **INDUS / OmniScience** (2024–2025): pretraining en corpus científico de dominio mejora razonamiento STEM.

---

## 6. Recursos estimados

| Fase | pro6000 | Trabajo Devin | Notas |
|---|---|---|---|
| V3.0 | 0 (ya generado) / 20 min eval | hecho | esperando launch Hermes |
| V3.1 | ~4h × 2 runs | 1 día | curriculum masking |
| V3.2 | ~4h × 2 runs | 1 día | role-aware masking |
| V3.3 | ~5h × 1 run | 1–2 días | corpus eco-fino |
| V3.4 | ~8h × 2 runs | 2 días | SFT contrastivo (nuevo trainer) |
| V3.5 | ~20h | mínimo | escala de steps |

---

## 7. Próxima acción

Hermes debe lanzar:

```bash
sbatch /beegfs/a474r867/ecoreasoner/scripts/eval_v3_piloto.slurm
```

Esto generará `battery_v3_verdict.json` para el piloto v2. Con eso decidimos si el límite estaba en la batería v2 o en el modelo.
