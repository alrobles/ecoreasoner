# EcoReasoner

Hybrid AR-diffusion language model for scientific agents — ecology, species distribution modeling, and computational biology.

## Architecture

```
EcoReasoner-Hybrid
├── AR Core (causal reasoning, tool-calling)    → Qwen3.5-35B-A3B / GLM-4.7-Flash
├── dLLM Core (fast draft, summarization)        → LLaDA-8B / LLaDA-MoE-7B
├── Router (task-based dispatch)                  → heuristic + scoring
└── Integrator (schema validation, AR fallback)   → format checker + corrector
```

## Components

- **AR Core**: Fine-tuned LoRA on Qwen3.5-35B-A3B (EcoReasoner swarm, MI210)
- **dLLM Core**: LLaDA-8B-Instruct (MIT license), deployed via vLLM on MI210
- **Distillation pipeline**: Multi-teacher CoT generation from PubMed + GBIF literature
- **maxentcpp**: C++ MaxEnt engine with R interface (github.com/alrobles/maxentcpp)

## Documentation

- [Feasibility: Hybrid dLLM architecture](docs/feasibility-hybrid-dllm.md)
- [LLaDA-8B strategy (server + fine-tune, no from-scratch)](docs/llada-8b-strategy.md)
- [dLLM-MoE PoC (0.5B masked-diffusion MoE)](docs/dllm-moe-poc.md)
- [dLLM feasibility on KUHPC (model choice, cost)](docs/dllm-feasibility-kuhpc.md)
- [dLLM literature map (v3, 34 refs)](docs/dllm-literature-map.md)
- [dLLM experimental protocol (RQ1-RQ4, gates go/no-go)](docs/dllm-experimental-protocol.md)
- [EcoReasoner design doc](docs/ecoreasoner-DESIGN.md) (from HPC)
- [Fase 2 curriculum](docs/fase2-curriculum.md) (Block A/B/C)

## Hardware

- KU CRC HPC: AMD MI210 (gfx90a, 64GB HBM2), 132 nodes in sixhour partition
- ReumanLab mesh: Q6000 Ada (48GB), PRO6000 Blackwell (96GB)
- ROCm 6.4.1+, vLLM rocm images, enforce-eager, AITER=0

## Related repositories

- [alrobles/dLLM](https://github.com/alrobles/dLLM) — Literature review on diffusion LLMs (v3, DiDAL + corpus integration) + experimental protocol for MI210
- [alrobles/ecoseek-litdump](https://github.com/alrobles/ecoseek-litdump) — Paper corpus and CoT traces
- [alrobles/maxentcpp](https://github.com/alrobles/maxentcpp) — MaxEnt SDM engine (C++/R)

## Corpus (PubMed/PMC multi-dominio)

**Nomenclatura canónica de datasets:** ver **[docs/dataset-registry.md](docs/dataset-registry.md)**
— cada dataset tiene un código estable `ETAPA__TIPO` (p.ej. `2_pretrain_3` = corpus v3,
`2_ids_3` = su pre-tokenizado, `3_distill_r3` = trayectorias de destilación round 3).
Para auditar el estado vivo de los datos en el cluster: `python3 scripts/dataset_catalog.py --group`.

Pipeline de ingesta en beegfs (ver `scripts/mine_pubmed_duckdb.py`, `map_pmc_fulltext.py`, `port_pubmed_parquet.py`):

| Artefacto | Path (HPC) | Contenido |
|---|---|---|
| Parquet PubMed | `/beegfs/a474r867/litdump/pubmed/parsed/parquet/` | 30.8M abstracts, particionado por año (6.3GB) |
| PMC-ids.csv.gz | `/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz` | 11.4M pmid→pmcid (252MB) |
| `eco_corpus.jsonl` | `/beegfs/a474r867/ecoreasoner/data/` | abstracts multi-dominio (ecología/filo/genómica/bioc) |
| `fulltext_corpus.jsonl` | `/beegfs/a474r867/ecoreasoner/data/` | **11,137 artículos full-text PMC, ~153M tokens (580MB)** |

Fuentes: abstracts vía DuckDB sobre Parquet; full-text via `pmc-oa-opendata.s3.amazonaws.com` (AWS Cloud, sin FTP/captcha).

## License

MIT License — A.L. Robles Fernández
