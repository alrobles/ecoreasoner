# EcoReasoner — SSL/Embeddings sobre el input (diseño crítico)

> Autor: Hermes (con A.L. Robles) · 2026-08-28 · Estado: DISEÑO, sin ejecutar
> Contexto: propuesta del usuario "generar embeddings de los papers y aplicar SSL
> para mejorar el input del reasoner". Este documento es el análisis crítico +
> plan concreto. No se ha lanzado ningún job.

## 0. Decisión de fondo (la parte incómoda)

**El objetivo masked-diffusion (D3PM absorb-mask) del dLLM YA es self-supervised.**
Es, de hecho, el pretext task más potente que existe para representaciones de
texto (denoising a escala de token). Añadir "SSL sobre embeddings" al
entrenamiento del modelo es redundante SALVO que la señal venga de una fuente
que el modelo no ve.

Y aquí está el hallazgo que cambia la propuesta: **el corpus v4 de entrenamiento
es texto plano** (`text, pmid, domain, year, source` — verificado 2026-08-28,
head -1 de train_corpus_v4.jsonl). No tiene:

- MeSH terms / keywords
- journal / autores
- citaciones / co-citación
- co-autoría

Es decir: la única señal verdaderamente nueva que los embeddings podrían aportar
(relaciones entre papers que no están en el texto: co-citación, co-autoría,
misma temática MeSH) **hoy no existe en el input**. Los embeddings de un encoder
externo sobre el texto serían una compresión del MISMO texto que el modelo ya ve
— circular, sin señal nueva.

## 1. Desglose honesto de "SSL con embeddings"

| Vía | Señal nueva para el dLLM | Coste | Veredicto |
|---|---|---|---|
| Embeddings del propio modelo → SSL | NINGUNA (circular) | alto | ❌ no |
| Embeddings externos (SPECTER2/BGE/SciBERT) → SSL contrastivo sobre pares del texto | NINGUNA (misma info que los tokens) | alto | ❌ no |
| Embeddings externos → **curar datos** (dedup semántico, batches temáticos) | Indirecta: mejor señal por token, menos redundancia | **bajo** | ✅ sí |
| **Estructura del corpus** (MeSH/journal/citas unidas por pmid desde la DB FTS) → contrastivo aux | SÍ — relaciones no-textuales | medio | ⚠️ experimento después |
| Embeddings externos → **retrieval en inferencia** (RAG) | SÍ — contexto adicional en uso, no en train | bajo | ✅ sí, pero NO es SSL |

## 2. Realidad de los datos (verificado)

- Corpus pretrain activo: `2_pretrain_3` = train_corpus_v3.jsonl (1,011,449 docs,
  ~1.7B tok) y su tokenización `2_ids_3`. El v4 (train_corpus_v4.jsonl, 20GB,
  ulterior) se usa por `moe-v4-bw` AHORA MISMO (job 27477651 corriendo).
- **El corpus está EN USO**: cualquier cambio de input ⇒ re-tokenizar
  (`pre_tokenize.py` → `train_ids_v5.npy`, ~6.4GB) ⇒ relanzar el training. Coste
  de re-tokenización: ~30-60 min en login node (sin GPU).
- La DB `pubmed_fts.db` (97GB) SÍ tiene por pmid: title, abstract, journal,
  authors, mesh_terms, keywords, pub_year, language. Unir estos campos al corpus
  por pmid = enriquecimiento estructural barato (SQL join, 1h CPU, sin GPU).
- El corpus v4 mezcla fuentes (source: v3/ecoevorxiv/...): el dedup semántico
  es la única forma de limpiar duplicados transversales (mismo paper en PubMed
  y GBIF/PMC con pmid distinto).

## 3. El orden correcto (crítico)

**Primero hay que tener un EVAL estable, después tocar el input.**
Hoy no existe EcoBench-EVAL ejecutado sobre el dLLM (está en Fase 4, sin hacer).
Sin una métrica de referencia, cualquier cambio de input es indescifrable: no
podrás decir si el dedup/curriculum/contrastivo mejoró o empeoró nada.

Secuencia recomendada (cada fase con criterio de éxito medible):

### Fase 0 — Baseline EVAL (pre-requisito, no opcional)
- Ejecutar EcoBench-EVAL / bench de la Fase 4 sobre el checkpoint actual de
  `moe-v4-bw` (o el mejor disponible) → métrica piso.
- Criterio: número reproducible; sin esto, nada de lo siguiente tiene lectura.

### Fase 1 — Curación con embeddings (la parte barata y de valor inmediato)
Coste estimado: 1 pasada de embedding sobre 1M docs con un modelo compacto
(BGE-small ~33M o similar) en 1 GPU Blackwell/Q6000: **~10-20 min** +
indexado HNSW (faiss, CPU) + join de metadatos.
1. **Dedup semántico**: pares con cos ≥ 0.95 → colapsar (guardar el doc más
   rico: full-text > abstract, más reciente). Esperado: eliminar 1-5% de docs.
   Criterio: % tokens eliminados < 3% y loss a 500 steps NO empeora vs baseline.
2. **Batches temáticos** (curriculum suave): agrupar por embedding (k-means
   o HNSW-near) y muestrear batches coherentes. Criterio: A/B de loss curves
   (mismo nº de steps, semilla fija) — el temático NO debe empeorar, idealmente
   baja más rápido.
3. **Enriquecimiento estructural** (join por pmid con pubmed_fts.db): añadir
   mesh_terms/journal/authors al corpus. Esto habilita la Fase 2. No cambia el
   objetivo de entrenamiento; es metadata para señal futura y para análisis.

### Fase 2 — Experimento contrastivo (SEÑAL NUEVA genuina, si la 1 paga)
- Aux loss contrastivo en el encoder del dLLM con pares definidos por
  **estructura**: positivos = mismo MeSH principal / misma revista / co-autoría
  (NO por similitud de texto, que es circular). Hard negatives = misma temática
  pero dominio distinto.
- Esto SÍ toca `train_mdlm_moe.py` (añadir rama de loss) y requiere relanzar
  entrenamiento. Criterio: mejorar EcoBench-EVAL de Fase 0 sin degradar la
  loss de denoising (monitorizar ambas).

### Fase 3 — RAG en inferencia (si sobra apetito)
- Embeddings → retrieval de papers relacionados como contexto al generar
  razonamiento. No es SSL ni toca training; es un cambio de inferencia.

## 4. Qué NO hacer (resumen de trampas)

1. **No** contrastive sobre pares de texto con embeddings externos: la señal es
   circular (el modelo ya ve el texto). Pérdida de tiempo y GPU.
2. **No** tocar el corpus mientras `moe-v4-bw` entrena sobre v4 sin haber
   guardado un checkpoint evaluable. El cambio de input es cambio de datos, y
   los datos están en producción.
3. **No** re-tokenizar a la ligera: cada cambio de corpus = 6.4GB de npy nuevo.
4. **No** empezar "mejorar el input" sin EcoBench-EVAL: es diseñar a ciegas.

## 5. Siguiente paso concreto (barato, sin GPU, hoy)

1. Escribir `dedup_embed.py`: lee train_corpus_v4.jsonl, genera embeddings
   (modelo compacto), indexa HNSW, marca pares ≥0.95, reporta % duplicados
   (sin escribir nada todavía — solo diagnóstico).
2. Escribir `enrich_pubmed.py`: join por pmid contra pubmed_fts.db → reporta
   cuántos docs del corpus tienen MeSH/journal disponibles (expectativa >90%).
3. Con esos dos números (duplicados reales %, cobertura MeSH %) decidir si la
   Fase 1 merece la pena: si dedup <0.5%, la curación no paga; si cobertura
   MeSH <50%, la vía estructural es débil y hay que repensar.

> Regla de labor: nada de esto se ejecuta sin OK explícito. Los scripts de
> diagnóstico (paso 5) son solo generación de números de decisión.