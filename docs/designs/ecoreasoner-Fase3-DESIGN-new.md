# EcoReasoner Fase 3 — Diseño (refundación `-new`): excavar un dLLM científico

Fecha: 2026-09-04
Autor: A.L. Robles Fernández (Hermes)
Estado: RESET v0.1 — reemplaza el enfoque dependiente de pivote SFT; pendiente validación de Angel
Supersede: ecoreasoner-Fase3-DESIGN.md (v0.2) — conservar como histórico

---

## 0. Resumen ejecutivo (la refundación)

**Objetivo real (declarado por Angel):** generar agentes con capacidad de razonamiento
científico. El branding "ecológico" es envoltura; la tesis que se excava es: ¿puede un
dLLM **desde cero** aprender a razonar científicamente (no solo a generar prosa)?

**Lo que queda fuera de la mesa:**
- **NO pivote a LLaDA-MoE-7B-A1B.** LLaDA existe solo como *regla de medición*: está ahí
  para enseñar que un MoE de difusión puede hacer tool-use; se usa como **techo de
  referencia** en una comparativa justa (misma tarea, mismo dominio, nuestro dLLM vs
  LLaDA). No es pieza del producto ni atajo de arranque. Un atajo no nos hace
  competitivos frente a nadie: no construye capacidad propia.
- **NO SFT como vía productiva.** Descartado formalmente (era mi punto 4 del banquillo).
  El SFT sobre un MoE preentrenado "ya habla" pero no excava capacidad propia; lo
  descartamos por coherencia con el objetivo: excavar desde cero hasta donde se llegue.
- **NO horizonte corto.** "Con éxito en 3-4 semanas" muere. Este es un proyecto de
  exploración del límite (meses). No hay prisa.

**Reglas que se conservan del diseño v0.2** (siguen correctas y son aprendizaje real de
bw3/bw4_span):
- Harness como "CI del training"; cada hipótesis validable en <4h GPU antes de escalar.
- Test-and-drop: el perdedor se descarta sin escalar; micro-sweep con criterios
  pre-registrados.
- Desacoplar objetivo / datos / escala; loss NO es proxy de lenguaje (siempre + smoke).
- Escala heterogénea (pool NVIDIA completo) como hito de systems.

**Cambio central que toca objetivos, datos y eval:** pasar de "fluidez de prosa" a
**razonamiento científico**. Eso reescribe qué se mide, con qué se entrena y qué se
considera éxito.

---

## 1. Lecciones de bw0→bw4_span (lo que NO se repite, y por qué)

| Experimento | Resultado | Lección |
|---|---|---|
| bw0/bw1v5 (base, random 15%) | loss 11.9→5.2, sin texto coherente | 37M tok vistos ≈ 3% del corpus: subentrenamiento severo |
| bw3 (random 15%, v7, 93K) | loss 2.0-4.2; word 0.20-0.33, rep4 0.20-0.41 (colapso copia-vecino) | mask aleatoria dispersa premia copiar el adyacente; loss NO es proxy de lenguaje |
| bw4_span (span 64, v7, 1 epoch) | rep4 0.003 (repetición erradicada) pero word 0.19-0.34, uniq 0.19-0.39 (sopa de subwords); 0/180 JSON | span rompe el atajo local pero no construye estructura; el objetivo solo no basta |
| bw5_spanhi (mask_p 1.2 → 60% real, staged, sin lanzar) | teoría: máscara alta obliga a reconstruir con contexto escaso | tercer punto de la curva de ablación del objetivo (ver §5) |

Diagnóstico consolidado: (a) masking al 15% tiene un atajo local; (b) escala inviable
para lenguaje desde-cero; (c) los datos (prosa cruda) no aportan **estructura de
argumento** explícita — y esa estructura, no la fluidez de superficie, es lo que el
razonamiento científico necesita.

## 2. Realidad de cómputo (2026-09-04, verificada en cola)

| Recurso | Cantidad | Notas |
|---|---|---|
| pro6000 (Blackwell) | 3 (r30r08, r30r24; r23r09) | únicos torch 2.7.1 cu128; ~14 steps/min × 24576 tok/opt-step |
| Q6000 | ~28 (7×3, 2×4) | SIF 2.4.1 cu121; NCCL ~50ms/10MB; sin sm_120 |
| L40 | ~4 | — |
| A100 | ~4 | — |
| MI210 | 4 nodos | NO confiable — no contar |
| **Pool NVIDIA (teórico)** | **~35-40 GPUs** | ~10-14× el cómputo de 2 pro6000 |

**Advertencia (verificada en cola hoy):** el "pool completo" es teórico. Ahora mismo en
cola a474r867 solo hay pro6000 (bw5 cargado) y ollama-q6000. Los Q6000/L40/A100 no están
libres al mismo tiempo. Además, la eficiencia heterogénea real (NCCL ~0.1-0.2 GB/s +
sync_every) dará más cerca de 2-4B tok/día que 5-7B. **El harness de escala (paper de
systems) se construye EN PARALELO con el harness de micro-runs** — es infra, no
experimento: no hay razón para ponerlo en serie con el sweep.

**Aritmética de micro-run (corrección a v0.2):** el diseño v0.2 prometía "1-2B tok en
1-4h en 1 GPU". Verificado: 1 pro6000 ≈ 2.9K tok/s → en 4h ≈ **42M tok**, no 1-2B.
Un modelo 150-400M (Chinchilla) necesita 2.5-7B tok. Consecuencia: el micro-run es
equivalente a bw0 (subentrenado) si se pide "tokens suficientes". Dos vías honestas:
(a) micro-modelo 50-100M cuyo criterio es "¿aprendió la ESTRUCTURA del objetivo?", no
tokens inferiores; o (b) aceptar que una hipótesis de objetivo cuesta ~1 día, no 4h.
El diseño NO promete lo que no cumple; ajustar §Análisis del criterio en consecuencia.

## 3. La tesis y su falsación pre-registrada (novedad del `-new`)

**Tesis:** un dLLM desde-cero, entrenado para denoising de *estructura de argumento
científico* (no prosa cruda), puede aprender a discriminar la continuación
inferencialmente correcta de un razonamiento frente a alternativas plausibles pero
incorrectas, y completar el esqueleto de un argumento.

**Es una tesis abierta.** Nadie ha demostrado que un dLLM <1B aprenda razonamiento
multi-paso por denoising. Por eso el fracaso se escribe ANTES de gastar el pool:

> **Criterio de parada (pre-registrado):** si tras N tokens vistos el modelo no
> discrimina inferencias por encima del azar (≥55% pairwise, transitorio) NI completa
> esqueletos coherentemente, la tesis quedará **falsada a nuestra escala**. Un negativo
> limpio con ablaciones (random / span / span-alto × datos prosa / esqueleto) es un
> resultado publicable. El diseño NO exige victoria: exige **respuesta** con evidencia.

El valor está en "excavar hasta donde podamos llegar": el límite mismo es el entregable
si se pre-registra y se mide bien.

## 4. La eval: de fluidez a discriminación inferencial

**La eval actual NO sirve para este objetivo.** word_ratio / rep4 / uniq miden fluidez
de prosa, no razonamiento. Para razonamiento científico la unidad mínima es la
inferencia, no el token. Suite smoke nueva (barata y discriminativa):

1. **Discriminación inferencial (pairwise, sin LLM-judge):** dado un argumento truncado,
   ¿elige el modelo la continuación inferencialmente correcta sobre una plausible-pero-
   incorrecta? Métrica = accuracy pairwise. Barata (ranks, no generación libre).
2. **Completación generativa:** completar el siguiente paso del esqueleto de argumento
   (hipótesis → predicción → evidencia) y comparar contra ground-truth / jueces de
   superficie validados.
3. Fluidez (word_ratio/rep4/uniq) se conserva como **diagnóstico**, nunca como criterio
   GO del objetivo científico.

**Comparativa justa:** LLaDA-MoE-7B-A1B corre la MISMA batería (misma tarea, mismo
dominio, mismo protocolo) como techo de referencia. Nuestro objetivo de capacidad es
una fracción de lo que LLaDA puede hacer en prosa, pero sobre lo que se excava (la
tesis de razonamiento inferencial), LLaDA define el tope a medir contra — no el techo
de producto que perseguimos.

## 5. Línea de ablación del objetivo (dónde encaja bw5_spanhi)

| Objetivo | Variable | Pregunta que responde |
|---|---|---|
| random 15% (bw3) | — | ¿el masking disperso sin estructura? → atajo local, fracasa |
| span 64 × mask_p 0.15 (bw4_span) | ratio máscara | ¿span solo construye estructura? → rompe atajo, no estructura |
| span 64 × mask_p 1.2 → ~60% (bw5_spanhi, staged) | ratio máscara ALTO | ¿el problema es la RATIO de máscara o la ESTRUCTURA de los datos? |
| esqueleto de argumento (nuevo, abajo) | datos | ¿es la ESTRUCTURA el ingrediente que falta? |

bw5_spanhi responde la bifurcación crítica: si span-alto TAMBIÉN fracasa → la hipótesis
"datos estructurados" queda como única vía viva y justifica el extractor de esqueletos.
Es punto de cierre de la línea "ratio de máscara", no vía productiva. Ya está staged
(0 esfuerzo en lanzarlo) y cuesta 1.6 días de los 3 Blackwell. **No bloquea nada**: el
diseño de excavación y el harness corren en paralelo y no dependen de su resultado.

## 6. Datos: de prosa a ESTRUCTURA DE ARGUMENTO (corrección de mi propia idea)

La idea de sketches previa (relaciones NER spaCy estilo "X regula Y") era correcta en
intención pero pobre en ambición: 4 verbos de señal no cubren la literatura real y los
esqueletos de relación son demasiado pobres para RAZONAMIENTO, además de caros de
extraer (NER de relaciones).

**Corrección:** los papers YA traen la estructura de argumento casi etiquetada —
IMRaD, abstracts estructurados, y frases-faro de señal ("we hypothesized that", "we
predicted", "we found", "these results suggest"). Extraíble con **heurística de
secciones + señales sintácticas**: cobertura alta, 0 GPU, sin NER.

**Átomo de entrenamiento:** el esqueleto de argumento serializado plano:

    [OBSERVACIÓN]<sep>[HIPÓTESIS]<sep>[PREDICCIÓN]<sep>[EVIDENCIA]<sep>[CONCLUSIÓN]

Denoising sobre esto = aprender la **gramática de la inferencia** (qué pasa de una etapa
a la siguiente), no la superficie textual. Esto ataca directamente el diagnóstico (c):
la estructura deja de ser implícita y pasa a ser el objetivo.

**Generalidad (razonamiento CIENTÍFICO, no solo ecológico):** balanceo de dominios en
la curaduría — física + computación + biomed + eco — para que la estructura domine
sobre el vocabulario de un solo dominio. Ya tenemos los ladrillos: v7_clean + phys
(arXiv).

## 7. Harness (se conserva del v0.2, con la eval reescrita)

1. `harness/configs/run.yaml` — spec declarativa (modelo, data, masking, steps, seed, tag).
2. `harness/run_micro.py` — valida config, arma slurm 1-GPU, ejecuta, recolecta.
3. `harness/report.py` — JSON estándar: suite smoke (discriminación inferencial +
   completación + fluidez-diagnóstico) + loss curve + throughput + sha256(config) + seed.
   Append a `runs/index.jsonl` (comparable entre runs).
4. `harness/compare.py` — tabla A/B/N con deltas y veredicto go/no-go contra baselines,
   incl. LLaDA como techo de referencia.
5. `runs/` en beegfs = fuente de verdad.

Reutilizar: smoke_cont_bw3.py / smoke_cont_span.py (pasar a suite_smoke.py),
train_mdlm_moe.py `--mask_type {random,span}` + `--span_len`, curate_embed
(vectors_full.npy, BGE-small), train_ids_v7_clean.npy.

## 8. Escala heterogénea (hito systems) — en paralelo, no en serie

- train_hybrid.py (grad_accum adaptativo por VRAM, all-reduce manual) ya existe y está
  validado single-GPU; extender al trainer full del dLLM.
- Pool: Q6000 + L40 + A100 + pro6000; NCCL 0.1-0.2 GB/s exige sync_every / all-reduce
  manual de gradientes para llegar a 2-5B tok/día REALISTAS.
- Micro-batch por rank según VRAM; grad_accum adaptativo = gradiente global equivalente.
- Slurm: patrones validados (overlay + torchrun, lock overlay, SIGUSR1 waves,
  checkpoints atómicos + resume tolerante).

## 9. Publicación (2 vías con el mismo cómputo)

- **Paper 1 (systems):** entrenamiento heterogéneo multi-GPU con grad_accum adaptativo
  por VRAM + harness reproductible. El hito publicable de Angel; independiente del
  dominio.
- **Paper 2 (tesis, desde-cero):** ¿aprende un dLLM <1B razonamiento inferencial por
  denoising de estructura de argumento? — con la falsación pre-registrada y ablaciones
  de objetivo (random / span / span-alto × prosa / esqueleto). Acepta el resultado sea
  positivo O negativo limpio.

## 10. Risk/estrategia — secuencia modelo → agente

Razonamiento buscado EN el modelo primero (capacidad latente: el dLLM infiere la
continuación correcta del argumento), y el AGENTE (tools / memoria / loop) encima
después. Primero capacidad en el modelo; luego agente que la explota. El micro-modelo
de 50-100M del sweep sirve para validar esa dirección sin gastar el pool.

## 11. Roadmap (sin prisa, hitos por decisión no por fecha)

| Fase | Qué | Criterio de salida |
|---|---|---|
| F0 harness + eval nueva | suite_smoke (discriminación + completación) + 6 micro-runs | ganador del micro-sweep con GO; index.jsonl con 6 entradas |
| F0b extractor de esqueletos | heurística IMRaD + frases-faro; train_ids_skeleton.npy balanceado por dominio | cobertura ≥70% de papers; validación por dominio |
| F1 entrenamiento objetivo | pool completo, receta del ganador | N tokens; discriminación ≥55% y completación coherente — o falsación registrada |
| F2 agente | loop + tools encima del modelo | razonamiento encadenado medible |
| Paper 1 (systems) | en paralelo con F0/F1 | draft |

Los hitos se miden por decisión, no por calendario (no hay prisa).

## 12. Riesgos (actualizados)

- Escalar antes del micro-sweep = repetir bw3 con más GPUs. NO.
- Loss como proxy: NUNCA sola; la eval de éxito es discriminación inferencial, no loss.
- Micro-run que promete "1-2B tok en 4h": falso (≈42M reales). Decidir estructura
  sobre modelo 50-100M, no tokens inferiores.
- LLaDA como muleta: prohibido como pieza; solo como techo de medición en comparativa
  justa (misma tarea/dominio/protocolo).
- Pool teórico ≠ pool real: asumir 2-5B tok/día y nodos compartidos/ocupados.
- bw5_spanhi: es cierre de curva, no avance; no esperarlo para construir el harness.

## 13. Piloto dLLM v2 post-F2 — control de riesgo antes de escalar (rediseño propuesto)

> Fecha: 2026-09-09 · Estado: **plan pre-registrado, pendiente de aprobación**

### 13.1 Diagnóstico que condiciona el rediseño

- **F0** mostró dirección correcta: `span` sobre esqueletos vencía a `random` y a prosa (`0.5352`), pero **ningún run alcanzó 0.55**.
- **F1** fue NO-GO por configuración de optimización: global batch demasiado grande y 1000 updates; la señal estructural se diluye.
- **F2** (50K steps, receta micro ganadora a escala) fue **FALSIFY** (`0.5273`) con un pico transitorio de `0.5664` que no se sostuvo; la pendiente final fue negativa.
- La **batería L0–L3** sobre el checkpoint final de F2 (`runs/f2-spanes/battery_verdict.json`) demostró que el modelo solo aprendió **coherencia temática**:
  - L0 (`bad` de otro documento): `0.5738`, significativo.
  - L1 (`bad` del mismo dominio): `0.5607`, significativo.
  - L2 (`bad` de otra etapa del mismo documento): `0.4650`, no significativo.
  - L3 (`bad` = etapa real con payload mutado): `0.5089`, no significativo.
- Interpretación: el atajo original (`jac(ctx,ok) > jac(ctx,bad)` por ~0.117) explica toda la señal previa. La receta F2 no enseñó orden de etapas ni inferencia de contenido.

### 13.2 Rediseño propuesto: v2 + evaluación hard-negative

**Tesis a refutar:** si el objetivo de denoising se reformula para reconstruir **etapas enteras** de un esqueleto (no spans aislados), con un schedule de máscara variable y un optimizador decente, un dLLM denso de 155M puede superar el atajo temático y aprender a distinguir la etapa/rol correctos y, en última instancia, el payload inferencial.

**Cambios respecto a F2:**

| Qué | F2 | Piloto v2 |
|---|---|---|
| Packing | No (`train_ids_skeleton.npy` plano) | Sí, con EOS por documento (`pre_tokenize_v2.py` + `.npz`) |
| Masking | Span 64 @ 15% fijo | **Whole-stage masking**: enmascara etapas completas (`[OBSERVACION]..[CONCLUSION]`); fallback a span |
| Schedule | `mask_p=0.15` fijo | `uniform` con `b_l=0.05, b_h=0.95` (máscara variable, muchas actualizaciones con máscara alta) |
| LR / optim | Constante, batch global 65K en F1 | Cosine decay, warmup real, grad clip, AdamW con global batch pequeño (8×2×768 = 12,288 tok/update) |
| Evaluación | 256 pares de `build_pairs.py` (tema soluble) | **Batería L0–L3** de `build_pairs_hard.py` + split train/val para evitar leakage |
| Cadencia | Pico recogido a posteriori | Eval fija cada 1000 steps, **sin mirar picos transitorios**; decisión sobre el checkpoint final |

**Lo que NO cambia aún:** arquitectura denso 154.8M (512/8/8/4), corpus de esqueletos existente, tokenizer LLaDA, vocab 126080, `seq_len=768`. Se aísla el efecto del *entrenamiento* sin confundirlo con escala de modelo o corpus.

### 13.3 Diseño del piloto

**Datos:**
1. Partir `data/skeleton/train_skeleton.jsonl` en `train_skeleton_train.jsonl` (95%) y `train_skeleton_val.jsonl` (5%, seed 7331).
2. Pre-tokenizar ambos con `pre_tokenize_v2.py` a `train_ids_skeleton_v2.npz` y `val_ids_skeleton_v2.npz`.
3. Generar batería L0–L3 desde `val_skeleton.jsonl` con `build_pairs_hard.py --n 512`.
4. El entrenamiento usa solo `train_ids_skeleton_v2.npz`; la batería val se mantiene fija durante todo el run.

**Entrenamiento:**
- Script: `scripts/train_mdlm_moe_v2.py` (validado en local el 2026-09-09).
- Slurm: `scripts/moe_v4_micro_v2.slurm`.
- Config: `harness/configs/f0-span-esqueleto-v2.yaml`, con overrides por `--export`:
  - `TAG=f0-span-esqueleto-v2-piloto`
  - `DATA_CACHE=/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton_v2.npz`
  - `TARGET_STEPS=10000`
  - `MASK_TYPE=span`, `MASK_SCHEDULE=uniform`, `MASK_SCHEDULE_ARGS='{"b_l":0.05,"b_h":0.95}'`
  - `WHOLE_STAGE=1`, `SPAN_LEN=64`
  - `LR=2e-4`, `LR_DECAY=cosine`, `LR_MIN_RATIO=0.1`, `WARMUP=200`, `GRAD_CLIP=1.0`
  - `BATCH_SIZE=8`, `GRAD_ACCUM=2`
  - `WEIGHT_TYING=0`, `USE_ROPE=0`, `EMA_DECAY=0.0`
- Hardware: 1× pro6000 (sixhour), ~2h.

**Evaluación:**
- Cada 1000 steps, correr `harness/suite_smoke_v2.py` cuatro veces (L0–L3) contra los `pairs_L{0..3}.jsonl`.
- Alternativa más limpia: añadir un modo `--pairs-dir` a `suite_smoke_v2.py` para emitir un solo `report_L0L3.json`.
- Registra: `pairwise_acc` y `mean_delta` por nivel; `loss` del trainer; `word_ratio/rep4/uniq` como diagnóstico, nunca como criterio GO.
- No se permite mirar picos: la decisión se toma sobre el **checkpoint g10000** con intervalos de confianza (Wilson binomial) sobre cada nivel.

### 13.4 Criterios de parada pre-registrados

| Resultado en g10000 | L0 | L1 | L2 | L3 | Veredicto |
|---|---|---|---|---|---|
| **FALSIFY** | < 0.55 | < 0.55 | ≤ 0.52 | ≤ 0.52 | La receta v2 no escapa al atajo temático. Archivar y pivotar: (a) objetivo contrastivo/explícito, o (b) controller/verificator como vía práctica. |
| **DIRECCIONAL** | ≥ 0.55 | ≥ 0.55 | > 0.52 | > 0.52 | Hay alguna señal de orden/contenido, pero no suficiente. Extender a 50K steps y reevaluar. |
| **GO** | ≥ 0.55 | ≥ 0.55 | ≥ 0.55 | ≥ 0.55 | Señal inferencial real. Escalar a 50K+ y/o escala heterogénea; comparar contra LLaDA. |

Notas:
- L0/L1 deben seguir altos si el modelo entiende lenguaje; no son la meta.
- L2 mide si aprendió **orden/rol de etapa** aun cuando el tópico es idéntico.
- L3 mide si aprendió **contenido inferencial** (dirección/negación/números).
- Si L0/L1 son bajos, hay un bug de entrenamiento u optimización; se investiga antes de declarar falsificación.

### 13.5 Controles y ablaciones (post-piloto, si pasa el GO de riesgo)

Si el piloto entra en **DIRECCIONAL** o **GO**, las ablaciones de seguimiento serían:
1. `WHOLE_STAGE=0` con `MASK_SCHEDULE=uniform` (para aislar el efecto del whole-stage).
2. `WEIGHT_TYING=1` o `USE_ROPE=1` (mejoras arquitectónicas baratas).
3. Modelo de 50M (hidden 384/6/6) con la misma receta (si la señal es robusta, probar que no es pura capacidad de parámetros).
4. Objetivo contrastivo (ranking de pares L2/L3) si whole-stage no basta para L3.

### 13.6 Costo y riesgo residual

- Costo total del piloto: ~2-3h de pro6000 + minutos de CPU para pre-tokenizar y generar pares.
- Riesgo principal: el atajo temático siga siendo suficiente para L0/L1 y el whole-stage no enseñe L2/L3. Si eso ocurre, el piloto habrá falsado la receta con evidencia barata.
- Riesgo mitigado: no se escala a 100K steps / pool completo hasta que el micro de 10K demuestre señal limpia en L2/L3.

### 13.7 Relación con controller/verificator

El plan B (práctico) queda intacto: si el dLLM propio no aprende, el controller/verificator validado en Fase 3 sigue como arquitectura productiva. Este piloto decide cuánto más invertir en la vía dLLM pura.

---

## Apéndice A — Archivos que se tocan / crean

| Archivo | Acción |
|---|---|
| `scripts/train_mdlm_moe.py` | NO tocar el run vivo; el harness llama con flags |
| `scripts/smoke_cont_bw3.py` / `smoke_cont_span.py` | heredar a `harness/suite_smoke.py` (misma salida JSON + eval de discriminación) |
| `harness/` (nuevo) | run_micro.py, report.py, compare.py, suite_smoke.py, configs/ |
| `scripts/build_skeleton.py` (nuevo) | extractor IMRaD + frases-faro + serialización de esqueleto |
| `data/skeleton/` (nuevo) | jsonl de esqueletos + train_ids_skeleton.npy |
| `runs/` (nuevo, en beegfs) | report.json + index.jsonl + checkpoints |
| Baselines | snapshot de LLaDA-MoE-7B-A1B en la MISMA suite (techo de referencia) |