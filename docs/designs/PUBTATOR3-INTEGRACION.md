# Integración PubTator3 (NCBI) — Minería de especies en literatura (FASE 3)

Fecha: 2026-09-09 · Autor: Hermes/Ángel · Estado: ✅ OPERATIVO (verificado)

## Qué aporta

PubTator3 (NCBI) es un sistema de anotación de entidades bio-*con* un tagger
entrenado (AIONER / GNormPlus) que anota **Species** y las normaliza a **NCBI
Taxonomy**. A diferencia del regex de binomios, reconoce sinónimos y
abreviaturas ("S. meliloti" → taxon 382, igual que "Sinorhizobium meliloti")
y NO produce falsos positivos de texto genérico.

Para la FASE 3 (minería de literatura → 500+ tool-calls) es la vía correcta
para obtener especies REALES de los textos del corpus.

## API (verificada 2026-09-09, endpoint REAL)

Export de anotaciones por pmid (sin API key, funciona):

    GET https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=25340565,27303257
    # formatos: biocjson | biocxml | pubtator   (full=true para full-text en xml/json)

- Documentación de NCBI: https://www.ncbi.nlm.nih.gov/research/pubtator3/api
  (la página es una app JS; el endpoint real está en el snippet de búsqueda y
  verificado por curl).
- Límite: ~3 req/s (espaciar; el script usa 0.4s).
- Para volumen alto: contactar NCBI (chih-hsuan.wei@nih.gov) o usar el FTP
  bulk (ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3) — anotaciones completas de PMC.

Anotar TEXTO PROPIO (alternativa RESTful clásica de NCBI, sin key):
    curl -d '{"sourcedb":"x","sourceid":"1","text":"..."}' \
      https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Species/Submit/
- ⚠️ Este endpoint dio 500 en la prueba (2026-09-09); el camino SÍ verificado
  es export por pmids con biocjson.

## Llave NCBI

- Archivo: `/home/reumanlab/env/ncbi-key` (37 bytes, tipo token corto).
- NO es necesaria para el export por pmids (probado sin ella).
- Si el volumen crece (muchos pmids), usarla si NCBI la requiere; para el
  export público no hace falta.

## Script operativo

`scripts/annotate_pubtator3.py`:
    python3 scripts/annotate_pubtator3.py --pmids-file pmids.txt --out annot.jsonl
        [--limit N] [--batch 40]

- Input: archivo con un pmid por línea (de fulltext_corpus_*).
- Output: jsonl con {pmid, title, species:[{ncbi_id, name, count}]}.
- Verificado: 18/20 pmids anotados, 11 anotaciones (6 Species + 5 Chemical)
  en 2 docs de prueba, ids NCBI presentes (382=Sinorhizobium meliloti,
  3879=alfalfa, 9655?/9606=Homo sapiens).

## Cómo integrarlo en la FASE 3 (patrón)

1. `annotate_pubtator3.py` → anotar un lote de pmids del corpus ecológico
   (los de ecoreasoner/data/fulltext_corpus_*).
2. Filtrar especies: quitar Homo sapiens (9606), los ids "" (no normalizados,
   revisar a mano), y los taxones no ecológicos (viruses, modelos lab).
3. Las especies anotadas (con NCBITaxonId) alimentan
   `build_toolcalls_500.py` (lista de especies REALES con proveniencia).
4. El resolver gbif_occurrence YA usa GBIF (usageKey) — se puede enlazar
   NCBITaxonId ↔ usageKey vía la API GBIF si se quisiera validación cruzada.

## CONTRASTE con el tagger GBIF (tag_species_gbif.py)

| | PubTator3 | GBIF species/match |
|---|---|---|
| Normalización | NCBI Taxonomy | GBIF usageKey |
| Detección | tagger entrenado (sinónimos/abreviaturas) | requiere binomial correcto |
| Falsos positivos | mínimos (entrenado) | filtrados por rank=SPECIES+conf |
| Acceso | export por pmid (pmids del corpus) | por texto candidato |
| Dominio | biomédico (PubMed/PMC) | todo el árbol de la vida |

Uso ideal: PubTator3 para extraer las especies anotadas del corpus (rápido,
normalizado), GBIF para validar/especies restantes cuando no hay pmid.

## Ficheros tocados

- NEW `scripts/annotate_pubtator3.py` (annotator operativo)
- `docs/PROMPT-DEVIN-TOOLCALLS-LIT.md` (referencia)
- `docs/results/` (resultado de prueba /tmp/annot_pt3.jsonl)
- Este doc: `docs/designs/PUBTATOR3-INTEGRACION.md`