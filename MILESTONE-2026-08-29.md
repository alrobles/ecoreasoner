# MILESTONE 2026-08-29 — EcoReasoner: bw1v5 COMPLETE + L1 pipeline launched

Date: 2026-08-29 19:10 CDT
Status: base training (L0) finished; tool-use adaptation (L1) launched.

================================================================================
1. MAIN MILESTONE: BW1V5 COMPLETED 6000 STEPS (COMPLETE flag)
================================================================================
Job: 27579314 (moe-v4-bw-v5, wave 3) — COMPLETED 01:05:59
- VANILLA trainer train_mdlm_moe.py + corpus v5_clean (train_ids_v5_clean.npy)
- MoE 4 experts top-1, world=2 intra-node, 2x RTX PRO 6000 (r30r24)
- 6000/6000 steps, final loss ~5.2 (evolution 11.9 -> 8.8 -> 6.5 -> 5.2)
- final checkpoint: outputs/bw1v5/checkpoint-g6000
- flag: outputs/bw1v5/training_complete.flag = COMPLETE
- loss per wave:
  wave 1 (11:57-17:41): ~5000 steps, SIGUSR1@300s correct, checkpoint g5101
  wave 2 (17:42, 27579259): FAILED 00:04:52 — resume loaded g5101 CORRUPTED
  (SIGUSR1 race: 2 ranks writing to the same dir -> torch.load failed)
  wave 2b (17:51, 27579280): FAILED 00:04:52 — after corruption-tolerant resume
  fix, detected g5101 corrupt, resumed g5051, but died on FileNotFoundError
  in tmp+rename (both ranks renaming the same tmp)
  wave 3 (18:03, 27579314): COMPLETED — after fix2 (tmp per PID + ONLY rank 0
  saves on SIGUSR1), resumed g5101 cleanly and completed 6000.

The 2 checkpoint races caught today (documented forever):
  R1: torch.save direct to the same path from 2 ranks -> interleaved file (corrupt)
  R2: tmp+rename without unique PID -> FileNotFoundError renaming the other rank's tmp
  FINAL FIX (train_mdlm_moe.py):
    - _save_checkpoint: tmp_m = model.pt.tmp.{PID}, atomic os.replace
    - _handle_sig: ONLY rank 0 saves on SIGUSR1 (world>1)
    - resume(): _try_load with try/except skipping corrupt checkpoints,
      tries state.json first then checkpoint-g* by descending step.

================================================================================
2. PHYSICS CORPUS (arXiv) — DOWNLOADING
================================================================================
Goal: 300K balanced general-physics full-text (physics is NOT in PubMed).
- Metadata: HF jackkuo/arXiv-metadata-oai-snapshot (2.7M papers, 4.58GB) downloaded
- Balanced selection (build_arxiv_select.py -> _arxiv_selected.jsonl): 300,000
  papers, 9 groups: phys-cond 70K, phys-hep 64K, phys-astro 48K, phys-physics 46K,
  phys-quant 26K, phys-grqc 15.5K, phys-mathph 11.5K, phys-nucl 10K, phys-nlin 8K
  Licenses: 188K arXiv-license + 112K CC (BY, BY-NC, CC0, SA)
  Years: 1986-2025 (peak 2024: 18K)
- Fetch: 8 shard array (job 27579383), LaTeX source via export.arxiv.org/e-print/<id>
  (gzip, 200KB each) -> plain text with strip_latex (stdlib, no pandoc)
  Rate: ~2.5K docs/min x 8 shards -> 300K in ~2h. At doc close: 143K/300K.
  Partial outputs: data/arxiv/fulltext/fulltext_c{0..7}.jsonl
- Next: build_arxiv_corpus.py -> train_corpus_phys.jsonl, then merge into v6+
  (phys-* domains).

================================================================================
3. L1 PIPELINE (TOOL-USE ADAPTATION) — LAUNCHED
================================================================================
Goal: the dLLM learns to GENERATE valid tool calls + REPAIR after error
(masked-diffusion, partial-plan revision). NOT competing with deepseek on general
agency; being the specialized eco-knowledge + repair engine inside the agent.

Raw material (2 formats):
  A) 85 conversational traces with real tool_calls JSON
     (distill_data.jsonl + distill_v4_round2.jsonl): gbif_occurrence,
     bioclim_download, maxent_train. Format [prompt -> trajectory(tool_calls) -> final]
  B) 2370 massive-distillation traces context->reasoning->code (sci_v2_b1.jsonl,
     code_valid=True): python eco/SDM analysis code.

Synthesis (build_l1_synth.py, l1_synth.slurm): per real tool call:
  A. gen:   [INSTRUCTION]+[CONTEXT] -> [ACTION] {json}
  B. repair: [INSTRUCTION]+[CONTEXT]+[ERROR] <mutation> -> [ACTION] {json}
     5 synthetic mutations: M1 truncated JSON, M2 broken quote, M3 value typo,
     M4 function typo, M5 broken id (deterministic over the JSON)
  C. final: [INSTRUCTION]+[CONTEXT] -> [RESPONSE] <text>
  Format B: [INSTRUCTION] context -> [ACTION] <plain python code> (ecocode)
Summary: 15,775 docs total (~8.5M tok)
  - 2,615 gen + 5x2,615 repair + 85 final (tool call JSON)
  - 14,178 ecocode (code generation)
  Functions: ecocode 14K, gbif_occurrence 672, bioclim_download 462, maxent 378
  Corpus: data/l1/train_corpus_l1.jsonl (36.5MB) -> data/l1/train_ids_l1mix.npy

L1 training (moe_v4_l1.slurm): continues from bw1v5 checkpoint-g6000,
MIXED dataset (L1 + 1/50 anchor of v6 ~40K docs), LR 1e-4 (low, don't destroy L0),
warmup 200, TARGET_STEPS 1500, vanilla trainer world=2, SIGUSR1 waves.

Job chaining (autonomous):
  l1_synth (DONE) -> l1_pretok (job 27579486, PENDING) -> moe_v4_l1

================================================================================
4. KEY FIXES DEPLOYED TODAY (git moe-v4-blackwell)
================================================================================
- atomic checkpoint + corruption-tolerant resume (R1/R2 above) [dd6a8c0]
- OOB token guard in build_batches (v5/v6 had 2 tokens 126082) [4852476]
- v6 corpus built: 2,018,444 docs (v5 + 113K fulltext 2025-2026) [155347d]
- train_ids_v5_clean.npy + train_ids_v6_clean.npy (OOB clamped)
- Blackwell slurm to overlay :ro (fix lock that killed t2 tests in 1s) [96d799d]
- NCCL 2.26.2 identified as IMA cause in Blackwell DDP (pytorch #152780) —
  fix pending: pip install nvidia-nccl-cu12>2.26.2 in /bb/bwvenv when the
  overlay is not in use (wait for L1 training to finish)

================================================================================
5. PENDING / NEXT STEPS
================================================================================
1. l1_pretok completes -> launch moe_v4_l1 (L1 training, ~1500 steps, ~2-3h)
2. arxiv-fetch completes (~150 more min) -> build_arxiv_corpus -> merge v6+phys
3. L1 smoke: given [INSTRUCTION]+[CONTEXT], does the dLLM generate the correct
   tool call JSON? Metric: valid JSON + repair rate (this is L1's success criterion)
4. Integrate service: FastAPI wrapper of the L1 model (generates/repairs tool calls)
   -> Emily/ecoseek.org: the agent calls the service when it needs
   eco knowledge or to repair a failed tool call
5. NCCL fix + validate top-2 (when the overlay is free)
6. Multi-model EcoBench comparison table (missing qwen)

================================================================================
KEY PATHS
================================================================================
| Path | What it is |
|---|---|
| outputs/bw1v5/ | COMPLETE base run (checkpoint-g6000, COMPLETE flag) |
| data/train_ids_v5_clean.npy | Clean v5 IDs (run base) |
| data/train_corpus_v6.jsonl | v6: 2,018,444 docs (21.4GB) |
| data/train_ids_v6_clean.npy | Clean v6 IDs |
| data/arxiv/ | Metadata + selection + fulltext (300K arXiv physics) |
| data/l1/train_corpus_l1.jsonl | L1 dataset (15,775 tool-use docs) |
| data/l1/train_ids_l1mix.npy | L1+anchor IDs (when pre-tokenized) |
| scripts/build_l1_synth.py | L1 synthesis (2 trace formats) |
| scripts/l1_synth.slurm | Synthesis job (DONE) |
| scripts/l1_pretok.slurm | Pre-tokenize mix job (PENDING) |
| scripts/moe_v4_l1.slurm | L1 training (pending launch) |
| scripts/build_arxiv_select.py | Balanced 300K arXiv selection |
| scripts/fetch_arxiv_text.py | LaTeX fetch -> text (8 shards) |
| scripts/build_arxiv_corpus.py | Merge shards -> train_corpus_phys.jsonl |
| scripts/train_mdlm_moe.py | Vanilla trainer (fixes: atomic+resume+OOB guard) |
| scripts/moe_v4_bw_train_v5.slurm | bw1v5 run slurm (reference) |