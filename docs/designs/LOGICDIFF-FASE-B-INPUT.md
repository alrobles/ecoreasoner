# Fase B — El INPUT como variable: corpus, objetivo y mecanismos de razonamiento

> Fecha: 2026-09-12. Tras Fase A (`docs/results/LOGICDIFF-FASE-A-RESULTADO.md`):
> scorer denso revela L3 0.545–0.640 en 8/8 checkpoints. La base falsable ya
> existe — cada experimento de esta fase se juzga con `dense` en
> `pairs_hard_v3_eval` + desglose por subtipo.

## 1. Diagnóstico del input (corpus skeleton actual)

Escaneo de 200K docs de `data/skeleton/train_skeleton_train.jsonl`:

| etapa | presencia |
|---|---|
| OBSERVACION | ~100% |
| EVIDENCIA | ~99% |
| CONCLUSION | ~99% |
| HIPOTESIS | **2.4%** |
| PREDICCION | **0%** |

**El corpus no contiene el paso inferencial intermedio.** Los esqueletos son
OBS→EVID→CONC: el modelo aprende "observación → evidencia → conclusión" sin
nunca ver la derivación (hipótesis → predicción) que conecta evidencia con
conclusión. Los pares L3 mutan exactamente ese contenido. Además:

- Distribución de roles (dataset LogicDiff): CONNECTIVE = 0.5% de tokens —
  los pivotes lógicos son rarísimos en el input.
- Dominio dominado por medgen/genom (~50%); eco solo ~15%.
- Estrategia de extracción: 59% IMRaD + 29% frases-faro — heurística que
  captura secciones, no cadenas deductivas.

**Hipótesis Fase B:** el techo de L3-dense (~0.64, concentrado en
direction_word) refleja un input sin contenido inferencial intermedio ni
contrastes explícitos, más que un límite del mecanismo de difusión.

## 2. Mecanismos de la literatura (búsqueda profunda)

| Mecanismo | Fuente | Qué aporta | Aplicable a nosotros |
|---|---|---|---|
| **LogicDiff** (orden por roles) | arXiv 2603.26771 | +38.7pp GSM8K zero-shot en LLaDA-8B; **ganancia desaparece con few-shot CoT**; cabeza 98.4% | Ya probado: no es nuestro cuello (staged≈dense). Cabeza nuestra 38% vs 98% suya — backbone débil |
| **Gt-Margin / planner de desenmascaramiento** | arXiv 2602.09501 | orden oráculo por margen a ground-truth; planner aprendido | Idem: orden no es el cuello |
| **DAG-aware training** | dLLM-Reason (BDeMo) | enmascarar niveles altos del DAG con más prob durante TRAINING → fuerza depender de niveles bajos | **Directo**: implementar como sesgo de masking por rol en train_mdlm_moe_v2 (ya existe role_mask!) |
| **Block diffusion / semi-AR** (BD3-LM) | arXiv 2503.09573 (ICLR oral) | AR entre bloques + difusión dentro → externaliza cómputo secuencial | Medio: seq por etapas = bloques naturales; esfuerzo moderado |
| **d1: masked SFT + diffu-GRPO** | arXiv 2504.12216 | SFT enmascarado sobre trazas de razonamiento (completion-only loss) + RL | **SFT-masked sí**: equivale a nuestro dense-scorer como objetivo de entrenamiento. RL prematuro con base 155M |
| **TOPD** (destilación on-policy en trayectoria) | arXiv 2607.16872 | teacher supervisa estados parcialmente denoiseados del propio dLLM | Interesante pero complejo; Fase C |
| **RSD — trazas aptas-para-el-estudiante** | arXiv 2509.22230 | destilar CoT largo crudo **degrada** modelos pequeños (−20.5% en 0.6B); trazas filtradas por probabilidad del estudiante mejoran +4.9% | **Advertencia clave**: si generamos HIP/PRED con teacher, filtrar por densidad/longitud apropiada a 155M |
| **Granularidad CoT no-monotónica** | arXiv 2502.18001 | estudiantes débiles aprenden mejor con CoT SIMPLE, no fino | Igual: trazas cortas y directas, no razonamiento largo |
| **FLD×2 / ALT — corpus lógico sintético** | arXiv 2411.12498 | deducción multi-paso procedural + distractores: +30pts razonamiento lógico en 70B | **Directo**: generar cadenas premisa→derivado→conclusión sintéticas con vocabulario ecológico |
| **Reasoning Core** (mezcla simbólica verificable) | arXiv 2603.02208 | mezclar datos simbólicos en pretraining mejora razonamiento sin dañar LM | Idem: mezcla 5–10% |
| **CARE protocol** | arXiv 2607.24763 | rankings de remasking publicados son inconsistentes sin control de cómputo | Valida nuestra lección: evaluación primero |

## 3. Escalera de experimentos (falsables, scorer fijo)

Todos: misma arquitectura 155M, 10K steps (~3.5h/job), eval `dense` sobre
`pairs_hard_v3_eval` (n=475) + subtipos. Control: v3-role dense L3=0.592***,
contrastive L3=0.640***.

### B0 — Auditoría de corpus (0 GPU) — HECHO 09-12

Resultados sobre 100K docs:

| feature | % docs |
|---|---|
| negación (no/not/never/without/lack/absence/fail) | 58% |
| números | 67% |
| conectivas causales/adversativas | 49% |
| palabras de dirección (increase/reduce/...) | 56% |

Por etapa con negación: OBS 24%, EVID 36%, CONC 31% (HIP 12%, PRED 0).

**El vocabulario existe — lo que falta es el CONTRASTE.** El modelo ve
negaciones y números constantemente, pero siempre dentro de argumentos
correctos: nunca ve un argumento plausible-pero-inválido que le enseñe a
distinguir validez de fluidez. Eso explica el patrón de subtipos: las
features están en el input, pero sin señal correcto/incorrecto no se ligan
a la validez inferencial.

Implicación para B2/B3: no basta con añadir vocabulario — hay que añadir
(a) la etapa inferencial intermedia (PREDICCION) y (b) pares/near-miss
donde la diferencia correcto/incorrecto esté marcada o sea inferible por
estructura.

### B1 — Objetivo candidate-focused en entrenamiento (1 job, parche pequeño)

El descubrimiento de Fase A aplicado al training: en una fracción de batches,
enmascarar el 100% de EVIDENCIA+CONCLUSION (o solo CONCLUSION) y computar loss
solo ahí — el modelo practica exactamente la tarea del scorer denso.

- Implementación: `train_mdlm_moe_v2.py` ya tiene whole-stage masking y
  role_mask — añadir modo `candidate_focus` (prob p≈0.3 de forzar máscara
  completa en la última etapa).
- **GO si** L3-dense ≥ 0.66 sobre el mejor base (Δ≥+0.02 vs contrastive).
- Si no mejora → el objetivo no era el cuello; el contenido del input sí.

### B2 — Rellenar la etapa inferencial (corpus + 1 job)

Generar HIPOTESIS+PREDICCION para una fracción del corpus con el teacher del
cluster (ollama-q6000 ya corre en el HPC):

- Prompt: dado OBS+EVID+CONC, escribir 1–2 frases de hipótesis y una predicción
  derivada ("si H es cierta, esperamos observar X"). Trazas CORTAS (RSD:
  modelos débiles necesitan CoT simple).
- Filtrar: longitud ≤60 tokens por etapa, sin números inventados, rechazar si
  repite literalmente la conclusión.
- Mezcla: 30–50K docs aumentados + corpus original.
- **GO si** L3-dense sube Y mejora en subtipos `causal_*`/`mechanism`
  (donde el contenido inferencial importa).

### B3 — Capa sintética deductiva estilo FLD (2–3 días, 1 job)

Generador procedural sobre relaciones ecológicas:

- Templates: `A aumenta → B disminuye`; distractor = permutación plausible;
  incluir TODOS los subtipos débiles: **números, negación, conectivas,
  temporal** (justo donde dense da azar).
- Formato esqueleto: `[OBSERVACION]...[PREDICCION] si <regla>, entonces <pred>`
  `[EVIDENCIA] <datos> [CONCLUSION] <derivada>`.
- 20–50K docs sintéticos, mezcla 5–10%.
- **GO si** subtipos number/negation pasan de ~azar a >0.6 sin degradar el resto.
- Si falla → límite de capacidad/mecanismo, no de input → cerrar línea dLLM
  para razonamiento fino (queda como scorer direccional).

### B4 — Contrastive + scoring denso (1 job)

Repetir V3.3-fixed pero con ranking sobre candidato completamente enmascarado
(el scorer que reveló la señal, convertido en objetivo). **GO si** supera 0.640.

### B5 (condicional) — Semi-AR por etapas

Solo si B1–B3 muestran que el contenido existe pero la generación paralela lo
desperdicia: denoise AR entre etapas, difusión dentro de cada etapa (BD3-LM de
juguete sobre nuestra arquitectura).

## 4. Prioridad recomendada

1. **B0** (gratis, hoy) → confirma/refuta el atajo léxico.
2. **B1** (parche mínimo, señal directa del hallazgo Fase A).
3. **B2 + B3 en paralelo** (atacan el hueco de input por dos vías:
   real-aumentada y sintética-controlada).
4. B4 como seguimiento del ganador; B5 solo si hay evidencia de que el orden
   sí importa en generación (no en scoring).

## 5. Criterio de cierre

- Si tras B1–B3 ningún run supera L3-dense 0.66 ni repara number/negation:
  **el input no era el cuello** → dLLM queda como scorer direccional (0.64 ya
  es útil para rerank en el controller/verificator) y el cómputo pasa al MVP
  híbrido.
- Si B2 o B3 levantan los subtipos débiles: el input era el cuello → escalar
  la estrategia ganadora (más docs aumentados / más mezcla sintética) y
  considerar Fase C (BD3-LM o TOPD).
