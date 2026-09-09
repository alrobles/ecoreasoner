# PLAN COMBINADO — Empujar el prototipo dLLM con prosa científica

Fecha: 2026-09-08 · Autor: Devin (SWE-1.7) · Estado: F2 FALSIFY, infra madura.

> Objetivo: decidir si un dLLM <500M entrenado desde cero con **prosa y
> estructura científica** puede ser el motor de EcoReasoner, y si no, cuándo
> pivotar a BD3-LM o controller/verificator.

Este plan fusiona dos líneas:
- (a) la ruta anterior (Fase A-E de auditoría/falsación)
- (b) la explotación de la prosa científica de alta calidad (PubMed/PMC,
  arXiv, ecoseek-litdump) que tenemos y que OWT no tiene.

---

> **Contraparte de software**: ver `docs/designs/arquitectura-prosa-dllm-2026-09-08.md`
> para módulos, flujo de datos, stubs y orden de implementación.

## 0. Preparación: clonar los repos de referencia

Ejecutar en el cluster **antes de empezar** (ya proveído en
`scripts/vendor_clone.sh`):

```bash
bash scripts/vendor_clone.sh
```

Esto descarga bajo `third_party/` (no trackeado en git):
- `kuleshov-group/mdlm` — receta MDLM, SUBS loss, sampling, entrenamiento OWT.
- `kuleshov-group/bd3lms` — BD3-LM, generación de bloques semi-AR, KV-cache.
- `ML-GSAI/RADD` — losses y samplers alternativos.
- `ML-GSAI/LLaDA` — evaluación y generación del techo 8B.

**Por qué ahora**: necesitamos leer el código de MDLM/BD3-LM para decidir si
portar la arquitectura o simplemente copiar entrenamientos, y necesitamos
poder correr LLaDA-8B zero-shot en la misma suite (Fase E).

---

## 1. Fase A — Cerrar el diagnóstico de F2 (días, pocos GPU)

Ya implementado en PR #1 (`harness/build_pairs_hard.py`, `scripts/verdict.py`,
`scripts/eval_battery.slurm`).

- [ ] Construir pares L0-L3 en beegfs: `runs/pairs_hard/`.
- [ ] Correr `--diagnose` sobre `runs/pairs.jsonl` (0 GPU, inmediato).
- [ ] Correr `eval_battery.slurm` sobre el ckpt final de f2.
- [ ] Leer `battery_verdict.json`.

**Decisiones resultantes**:
- L3 > 0.55 con p<0.05 → hay señal inferencial; la métrica L0 era demasiado
  fácil; se puede salvar la línea con negativos duros.
- L2 > azar, L3 ~ azar → aprendió orden de etapas, no contenido.
- Solo L0/L1 → atajo temático; línea archivada limpio.
- Nada → falsación fuerte; vía libre a BD3-LM y/o controller/verificator.

---

## 2. Fase B — Receta v2 sobre prosa Y estructura (1-2 semanas)

No se escala nada todavía. Se lanzan micro-runs de 10K-50K steps con recetas
mejoradas, cada uno <4h de GPU, con el harness.

### B.1 Prosa científica v8 (la mina de oro)

**Corpus**:
- `train_corpus_v7_clean.jsonl` (~4.3B tok) + full-texts PMC + arXiv-física +
  `ecoseek-litdump`.
- Dedup por DOI/PMID/título.
- Quitar boilerplate de Methods (patrones repetidos tipo "samples were
  collected", "the study was approved").
- **EOS packing por documento** (no stream plano sin fronteras).
- **Filtro de calidad**: perplexity filtering con un modelo referencia pequeño
  (70M causal, 1-2B tok) → quedarse con perplexity media (ni ruido ni
  plantilla).

**Objetivo (receta MDLM pública, transferida)**:
- `t ~ U(0,1)` por ejemplo (no 15% fijo).
- Explorar SUBS loss / simplex parameterization del repo MDLM.
- Learning-rate: warmup + cosine.
- Modelo: 155M dense (mismo que f2).

**Decoder**:
- `low_confidence` unmasking (default LLaDA, implementar en suite_smoke).
- sweep steps/temperature.
- best-of-N con denoise-loss como reranker.

**Métrica**:
- Fluency: word_ratio, rep4, uniq (baselines de texto real).
- Comparar contra `kuleshov-group/mdlm-owt` (130M, prosa general) en el mismo
  subset de prosa científica.
- Perplexity generativa en un holdout de prosa científica.

**Meta pre-registrada**: alcanzar prosa científica coherente (word_ratio ≥ 0.35,
rep4 ≤ 0.25, uniq ≥ 0.30) con contexto a 155M.

### B.2 Esqueleto receta v2 (en paralelo, solo si Fase A dejó señal)

- `t ~ U(0,1)` por ejemplo.
- Enmascarar etapas COMPLETAS como objetivo auxiliar.
- Ampliar esqueletos a 400-600M tok (ge2, arXiv-física, esqueletos sintéticos).
- Hard negatives (L0-L3) como métrica de validación.
- n≥620 pares, binomial exacto (`verdict.py`).

### B.3 Sketch-to-text / estructura condicionada (la fusión de prosa+estructura)

Aprovechar que cada paper tiene **prosa + esqueleto**.

- Construir pares: input = esqueleto, output = párrafo original.
- Entrenar denoising de prosa **condicionada a esqueleto** como prefijo no
  enmascarado.
- O en dos etapas: (1) generar esqueleto, (2) generar párrafo (estilo
  plan-then-write).

**Por qué es el producto**: esto da un motor que propone estructura de
argumento y luego la convierte en prosa científica. Es el encaje con el
EcoReasoner híbrido original (AR+dLLM): el dLLM propone esqueleto, la prosa
viene del mismo modelo o de un AR.

---

## 3. Fase C — Real inference desde la prosa (1-2 semanas)

Usar el **prosa** mismo para construir tareas de inferencia más difíciles,
no depender solo del extractor de esqueletos:

1. **Contexto de cita → resumen citado**:
   - Extraer "...as shown by [X]" de PMC fulltexts.
   - `ok` = abstract/conclusión del paper citado.
   - `bad` = abstract de otro paper citado en contexto similar.
   - Requiere mapeo PMID/PMCID del corpus. Si no existe, emparejar por vecinos
     de citación.

2. **Claim-evidencia dentro del mismo paper**:
   - Heurísticas: oraciones con "we found", "these results suggest".
   - Pares: claim → evidencia del mismo paper.
   - Bad: evidencia de otro paper del mismo dominio.

3. **Dirección del efecto real (L3 desde prosa)**:
   - Buscar pares de papers con resultados contrarios sobre la misma
     especie/gen/variable.
   - `ok` y `bad` tienen el mismo tópico; solo difiere la dirección del
     efecto.

4. **Ordenar secciones IMRaD**:
   - Barajar secciones de un paper; entrenar a ordenarlas.

El objetivo de esta fase es demostrar que el dLLM puede aprender **inferencia
no trivial** cuando los negativos son realmente duros y los datos vienen de
prosa real.

---

## 4. Fase D — Migrar a BD3-LM (2-4 semanas, decisión de arquitectura)

**Condición de disparo**:
- Fase B.1 (prosa) no alcanza fluidez a 155M con ~5-10B tokens de prosa
  científica, **Y**
- Fase B.2/C no producen señal en inferencia estructural/intra-paper.

**Acción**:
- Portar/adaptar `kuleshov-group/bd3lms` a nuestro corpus de prosa + esqueletos.
- BD3-LM soporta generación de longitud variable, KV-cache, y bloques
  semi-AR — resuelve los problemas de nuestro generador actual.
- Entrenar desde cero 155M con receta BD3-LM y nuestros datos.

**Nota**: BD3-LM es el cambio de arquitectura más natural, no un salto a AR.
Si BD3-LM tampoco da señal con buenos datos, la tesis dLLM pasa a ser
falsada con evidencia más fuerte y se abre el camino al controller/
verificator.

---

## 5. Fase E — Techos de medición

- [ ] **MDLM-OWT 130M**: correr `suite_smoke` (sin entrenar) sobre un subset de
      prosa científica de test → techo de un dLLM pequeño entrenado en OWT.
- [ ] **LLaDA-8B-Base**: correr la misma suite con `ML-GSAI/LLaDA` (transformers
      + trust_remote_code) → techo absoluto de dLLM 8B.
- [ ] **Twin AR causal** (opcional, del protocolo original RQ1): entrenar una AR
      155M idéntica en los mismos datos para medir la brecha
      específica de difusión.

Estas tres mediciones hacen que cualquier resultado nuestro sea
**interpretable**.

---

## 6. Fase F — Escalar a 350-500M (solo si hay señal)

Condición: Fase B, C o D produce un modelo 155M con (a) fluidez de prosa
aceptable, o (b) discriminación inferencial > 0.55 con p<0.05 en negativos
realmente duros.

- Aritmética: ~7-10B tokens para Chinchilla 350-500M. Con nuestro pool
  (pro6000 12.5K tok/s) son ~1-2 semanas de olas.
- No escalar sin señal en 155M: sería repetir bw3/f2 con más parámetros.

---

## 7. Decisiones go/no-go pre-registradas

| Fase | Gate | Si pasa | Si falla |
|---|---|---|---|
| A | L0-L3 battery | Decide si hay señal escondida | Archivar línea con mapa |
| B.1 | prosa v8 fluidez ≥ baselines | Tenemos motor de superficie | Paso a BD3-LM o datos v9 |
| B.3 | sketch-to-text con BLEU/LLM-judge | Tenemos producto (esqueleto→prosa) | Más datos o dos etapas |
| C | inferencia intra-paper ≥ 0.55 | Línea estructura/inferencia viva | BD3-LM |
| D | BD3-LM 155M mejora a B.1/B.2 | Arquitectura BD3 es la vía | Falsar dLLM puro, ir a controller/verificator |
| E | LLaDA/MDLM techo medido | Comparativa justa lista | — |
| F | señal en 155M + corpus ≥ 7B | Escalar a 350-500M | No escalar |

---

## 8. Próximos comandos para Hermes (orden de ejecución)

```bash
# 1) clonar repos de referencia
bash scripts/vendor_clone.sh

# 2) Fase A: batería L0-L3 (PR #1)
python3 harness/build_pairs_hard.py --in data/train_skeleton.jsonl \
    --tokenizer <path_llada> --n 512 --out runs/pairs_hard --seed 7331
python3 harness/build_pairs_hard.py --diagnose runs/pairs.jsonl
sbatch --export="RUN_DIR=/beegfs/a474r867/ecoreasoner/runs/f2-spanes" \
    scripts/eval_battery.slurm

# 3) Fase B.1: construir prosa v8 (data eng, pocos días)
#    [script a implementar: scripts/build_prosa_v8.py + .slurm]

# 4) Fase E: techo LLaDA-8B
#    [script a implementar: scripts/ceiling_llada8b.slurm]
```

---

## Referencias principales

- MDLM: arxiv.org/abs/2406.07524 / https://github.com/kuleshov-group/mdlm
- mdlm-owt: https://huggingface.co/kuleshov-group/mdlm-owt (130M, 33B tokens)
- BD3-LM: arxiv 2503.09573 / https://github.com/kuleshov-group/bd3lms
- RADD: arxiv 2406.03736 / https://github.com/ML-GSAI/RADD
- LLaDA: arxiv 2502.09992 / https://github.com/ML-GSAI/LLaDA
- LLaDA-8B: https://huggingface.co/GSAI-ML/LLaDA-8B-Base

*Documento de diseño rastreable. Fusión del documento previo
`devin-investigacion-dllm-prototipo-2026-09-08.md` con la explotación de
prosa científica.*
