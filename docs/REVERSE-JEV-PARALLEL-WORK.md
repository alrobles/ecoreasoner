# Trabajo paralelo reutilizable: reverse-jev (open decision model)

**Fecha**: 2026-09 · **Repo**: `github.com/alrobles/reverse-jev`
(local `~/GitHub/reverse-jev`, cluster `/beegfs/a474r867/reverse-jev`)

Reporte de una línea de trabajo paralela a ecoreasoner que ya resolvió
varios problemas que este roadmap necesita (decisiones tipadas, eval de
razonamiento, DAPT sobre backbone dLLM). Documenta qué existe, qué
funcionó, y qué es directamente reutilizable.

---

## 1. Qué es reverse-jev

Réplica abierta y entrenable de la clase "System One" (Jev de TypeSafe):
un modelo que recibe estado + preguntas tipadas y devuelve decisiones
estructuradas con probabilidades — sin generación de texto.

- **Backbone**: dLLM congelado. Dos soportes ya implementados:
  `GSAI-ML/LLaDA-8B-Instruct` (HF) y MdLMMoE nativo (`--ckpt`).
- **Heads especialistas por tipo** (~2M params cada una):
  - `choice`: AttnPoolHead — query aprendida que hace attention pooling
    sobre los tokens de cada opción, features concatenados de capas
    {8,16,24,32} → MLP
  - `noul`: MLP sobre mean-pool
  - `score`: MLP + loss auxiliar ordinal (CORAL-style: BCE sobre masa
    acumulada P(y≥j)) — respeta el orden 0<1<2 sin cambiar el decode
- **Canonical ordering**: opciones ordenadas determinísticamente por
  contenido antes de tokenizar → invarianza a permutaciones *exacta*
  (flip_rate = 0 por construcción). El promedio de logits sobre
  permutaciones en train NO lo logra; la canonicalización sí, y gratis.

## 2. Resultados (sci battery "elite", LLaDA-8B freeze)

| modelo | choice | noul | score | flip |
|---|---:|---:|---:|---:|
| baseline MLP mixto | 0.775 | 0.707 | 0.522 | 0.76 |
| **c_\*** (per-type + canonical + ordinal) | **0.870** | **0.783** | **0.608** | **0.00** |
| auto@5% (c_*) | 0.81 | 0.28 | 0.43 | — |

Benchmarks públicos (eval-only, temp-fit en dev elite):

| benchmark | tipo | c_* | referencia |
|---|---|---:|---|
| GPQA main | choice | 0.315 | LLaDA-8B nativo ~0.33 (techo) |
| GPQA diamond | choice | 0.328 | — |
| SciFact dev | noul | **0.853** | SOTA-decision-models ~0.89 |
| SciFact dev | score | 0.624 | ~0.70 |
| Enron/SST-2/AGNews/Banking77 | choice | (job en curso) | publicado 0.76–0.99 |

**Hallazgo clave**: la head ya extrae todo lo que el backbone sabe
(0.315 vs techo nativo 0.33). El gap en decisiones de conocimiento
paramétrico es del backbone, no del readout. → *La accuracy de decisión
es propiedad del backbone; la head es ajenas al dominio.*

## 3. Hallazgos de diseño (lecciones transferibles)

1. **Datos volumen ≠ transferencia**: 30× datos de otro dominio
   (qa_raw) → 0.96 in-distribution pero −13pts en elite. Lo que importa
   es cobertura del *tipo de razonamiento* del target, no volumen.
2. **Heads por tipo > head única**: choice quiere AttnPool multicapa;
   noul/score quieren MLP. AttnPool en noul colapsa a clase mayoritaria.
3. **Ordinal loss para score**: +8.6pts — score es el tipo más difícil
   para todos los sistemas medidos (incluido el propietario).
4. **Per-type training en noul necesita cross-data o canonical**:
   solo-noul (0.710) < mixto (0.782); canonical recupera sin orders.
5. **Full-FT del backbone = catastrophic forgetting** (0.775→0.623 en
   experimento temprano). LoRA/DAPT es la vía para inyectar dominio.

## 4. Reutilizable para ecoreasoner

### 4.1 Evaluación del backbone — GPQA como probe de conocimiento
`data/convert_benchmarks.py` convierte GPQA/SciFact/clasificación a
`{ctx, opts, gold}`. Con heads entrenadas sobre ecoreasoner, GPQA
funciona como **probe de madurez del backbone**: curva "decision acc vs
conocimiento acumulado" a través de checkpoints. El panel ya existe:
`scripts/bench_eval.slurm`.

### 4.2 Heads sobre MdLMMoE — ya funciona
`--ckpt` carga MdLMMoE nativo; `hidden_layers()` implementado en ambos
backbones. Receta transferida: canonical-order + per-type heads +
ordinal en score. Para 16 capas usar `--r2-layers=-1,-5,-9,-13`.

### 4.3 `reverse_jev/dapt.py` — mismo objetivo que vuestro trainer
Masked-diffusion `mask ~ U(0,1)`, CE/t sobre corpus de papers
(`corpus_v3/v4/v5` de `data/`), LoRA r=16 → adapter ~300MB. Es el mismo
loop de pretraining de ecoreasoner pero sobre pesos HF existentes —
sirve para comparar "adaptar LLaDA" vs "entrenar desde scratch" con el
mismo corpus y objetivo. `--lora-adapter` ya plumbado en train/eval.

### 4.4 Métricas de despliegue
`reverse_jev/eval.py` implementa: acc, ECE, NLL, Brier, **flip_rate**
(permutación de opciones), **automation@5%err** (fracción auto-
aceptable bajo 5% error — la métrica de un System One real), temp-fit
en dev. Todo reusable para ecobench.

### 4.5 Datos de decisión científica
`data/sci_battery/` (elite: factual/causal/multihop/negation/numerical/
definitional, split por pid) y `sci_battery_v2` (25K pids, ~470K
decisiones) — evals tipadas de razonamiento científico listas.

## 5. Restricciones

- Números de la API Jev: **internos, no publicables** (MCA 2.3(f)).
  Viven en `evals/` y `runs/sci/ABLATION.md` (gitignored). Los públicos
  citables son reportes de terceros (Kumar 2026, jev-harness-lab).
- El ecosistema open ya existe: `kev-*` (jaredpalmer, Qwen+LoRA+pointer
  head), `openjev`, `Open-Jev`, SemIf, Laya — diferenciador nuestro:
  backbone dLLM + dominio científico + canonical ordering exacto.

## 6. Estado actual y siguiente

- `dapt_llada` (en cola): DAPT-LoRA de LLaDA sobre ~100M tokens de
  papers → `runs/dapt/lora-final`
- `sci_llada6` (script listo): receta c_* sobre backbone adaptado →
  eval GPQA/SciFact/elite. Hipótesis: elite/SciFact suben (corpus es
  del dominio); GPQA sube moderadamente.
- Paper draft en `paper/main.tex` (arXiv-track): "Open System-One
  decision models: frozen dLLM backbones + specialist heads".
- Pendiente para ecoreasoner: baseline de heads c_* sobre
  `bw1_sr/checkpoint-g20` + `sft_moe_v2/lora-final` → punto cero de la
  trayectoria backbone-vs-decisión.
