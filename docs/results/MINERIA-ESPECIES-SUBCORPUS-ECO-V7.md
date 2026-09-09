# Minería de especies — subcorpus ECO del v7 — RESULTADO (2026-09-09)

Fecha: 2026-09-09 · Jobs: 28995803/28996619 (extractor) + tagger GBIF local

## Qué se hizo

1. Extractor de entidades sobre el fulltext PMC completo (`fulltext_corpus_all.jsonl`,
   97,012 docs, job 28995803): 1.32M candidatas binomiales, 61 regiones, 104K
   candidatos doc×tool.
2. Tagger GBIF (species/match, rank=SPECIES, conf≥85) sobre las 600 más frecuentes:
   29 reales — pero el **top era biomédico** (Arabidopsis, S. aureus, levadura,
   ratón). El PMC completo es dominio biomédico, no ecológico.
3. **Subcorpus eco del v7 curado** (job 28996619): filtrado por domain ∈
   {eco, phylo, bioc, climate, conserv, marine, plant, palaeo, microbio} del
   `train_corpus_v7_curated.jsonl` → **919,700 docs (63.1%)** → extractor sobre
   400K docs → 1.19M candidatas, 59 regiones.
4. Tagger GBIF (con retry backoff; el run completo previo murió 291/600 por
   throttling) sobre las 600 top del subcorpus eco → **84 especies reales
   (0 errores)**.

## Hallazgo estructural (importante para la FASE 3)

Incluso el subcorpus "eco" del v7 está dominado por patógenos/organismos modelo
(S. aureus, M. tuberculosis, C. elegans, levaduras) y cultivos de interés clínico.
Solo ~23 de las 84 son ecológicas reales:

- **Cultivos/agrícolas**: Oryza sativa, Zea mays, Triticum, Solanum ×2, Glycine,
  Brassica, Vitis, Hordeum, Phaseolus, Nicotiana ×2
- **Ganadería**: Sus scrofa, Bos taurus
- **Ecológicas de campo**: Apis mellifera, Canis lupus, Daphnia magna, Salmo salar,
  Oncorhynchus mykiss, Bombyx mori, Cyprinus carpio, Oreochromis niloticus,
  Xenopus laevis

**Lección**: los dominios GRUESOS del v5/v7 (12 ILIKE) no separan eco de biomed
en el texto PMC — el 63% "eco" sigue siendo mayormente microorganismos clínicos.
La etiqueta FINA real ecológica está en el v5 (`domain_fine`: climate_ecology,
population_ecology...) que el v7 NO heredó. Para un poso genuinamente ecológico
(minería real de ecología de campo: SDM, macroecología, filogeografía) hay que
filtrar por domain_fine, o usar las fuentes eco-nativas del corpus (EcoEvoRxiv,
ecoseek-litdump, arXiv ecológico).

## Pool de especies ecológicas REALES confirmadas por GBIF (23)

(ver `species_eco_v7_tagged.json` — 84 totales con usageKey/confianza/menciones)

## Artefactos

- `/beegfs/a474r867/ecoreasoner/docs/results/entities_fulltext_corpus.json` (PMC completo, 200K top)
- `/beegfs/a474r867/ecoreasoner/docs/results/entities_eco_v7.json` (subcorpus eco, 200K top)
- `docs/results/species_eco_v7_tagged.json` (84 reales, GBIF-confirmadas, 0 errores)
- Scripts: extract_entities_lit.py (--top), extract_eco_v7.py, tag_entities_fulltext.py (retry), f3_eco_extract.slurm
- Candidatos doc×tool del subcorpus eco: 38,868 (listos para curación de tool-calls)