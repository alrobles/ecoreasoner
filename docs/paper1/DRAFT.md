# DRAFT — Paper 1 (systems)

**Title:** Training Diffusion Language Models on a Heterogeneous Shared HPC GPU Pool: Exact Per-Token Gradient Weighting, Scavenged-Pool Continuous Pretraining, and a Falsification-Oriented Evaluation Harness

**Status:** working draft, 2026-09-08; results arc updated 2026-09-17; v5-1b phase added 2026-10-09. All numbers verified against `docs/results/`, `docs/designs/`, `ROADMAP.md`, `scripts/`, `harness/`, and live cluster state.

---

## Abstract

Small diffusion language models (dLLMs) are an attractive substrate for studying scientific-reasoning capabilities from scratch, but academic groups rarely control a homogeneous accelerator allocation: they inherit whatever a shared HPC cluster happens to have free, across GPU families with different VRAM, throughput, interconnect, and even CUDA stacks. We describe a system for training a 154.8M-parameter masked-diffusion LM on such a heterogeneous NVIDIA pool (A100, RTX PRO 6000 Blackwell, L40, Q6000) under Slurm. The trainer assigns each rank a micro-batch sized by *measured* forward+backward VRAM consumption, and applies a per-rank gradient scaling `SCALE_I = world·micro_i/Σmicro` so that the optimizer step is the **exact** per-token average over the global batch — identical to a homogeneous run — while a *global* gradient-accumulation counter keeps DDP all-reduce lock-step across heterogeneous ranks. A multi-family launcher (one Slurm job per GPU family, TCP rendezvous via torchrun, forced Ethernet NCCL to survive an IB/no-IB fabric mix) and a wave-based checkpoint/resubmit pattern keep runs alive inside short queue windows. We pair this with a declarative evaluation harness (pairwise inferential discrimination, generative completion, curve tracking, automated verdicts). The flagship heterogeneous run (F1) scored at chance — a NO-GO against the pre-registered ≥0.55 criterion — but the micro-vs-F1 contrast isolated the cause: optimization shape, not scale. Following exactly that prescription (many small updates on a single GPU), a nine-generation evolutionary search over masking recipes converged on a configuration (`hinrcf10`) that scores **0.632 pairwise accuracy on the inferential level of a leak-cleaned holdout** — above the falsification floor — with a verified corpus ablation (raw Spanish text degrades discrimination by ~5σ; English-only translation recovers it) and a recipe port to a 676M-parameter MoE that recovers inferential discrimination (L3 0.600 clean holdout) while retaining the best surface scores — showing the structured objective, not scale or corpus volume, is what carries inference — and, at flagship scale, a 1.13B-parameter (285M-active) 16-expert masked-diffusion MoE pretrained ~1M optimizer steps (~12B tokens) entirely on scavenged six-hour queue windows via a self-resubmitting wave chain whose checkpoints resume across world sizes and GPU families, surviving weeks of preemption and a full-day storage outage. On top of that run we report the instrument layer that makes its decisions auditable: a corrected binding battery (context-necessity corruption profiles, iterative remask decoding) hardened through a documented instrument crisis, and two **pre-registered, equal-dose structural falsifications** — a fine-grained-MoE arm and a block-diffusion semi-AR arm — both negative against the same init, step budget, and battery, localizing the model's binding bottleneck to capacity/data rather than attention structure. An external scientific-decision trajectory (sciev) keeps converting exposure into capability after the internal pairwise metric plateaus at ~0.744, while a controlled SFT probe shows chat remains degenerate at this scale — SFT serves as a measurement instrument, not a product path. The systems contribution made the negative measurable; the harness made the pivot a decision rather than a guess.

---

## 1. Introduction

Diffusion-based language models — absorbing-state discrete diffusion in the D3PM/MDLM lineage, exemplified at scale by LLaDA — offer an alternative to autoregressive pretraining that is particularly interesting for scientific reasoning: denoising bidirectionally-conditioned text may capture *argument structure* rather than next-token surface statistics. Our broader project asks whether a small dLLM trained from scratch on the *skeleton* of scientific arguments (IMRaD-style stage sequences) can learn to discriminate the inferentially correct continuation of a reasoning chain from plausible but incorrect alternatives.

The obstacle is not algorithmic but operational. Our cluster is a shared, contended HPC system whose free accelerators at any moment span four NVIDIA generations — A100 (80 GB), RTX PRO 6000 Blackwell (102 GB), L40 (48 GB), Q6000 (48 GB) — partitioned across nodes with heterogeneous interconnect (some nodes expose InfiniBand, others Ethernet only) and heterogeneous software stacks (the Blackwell nodes require a cu128 PyTorch build; the rest run cu121). Standard DDP assumes homogeneous ranks: identical micro-batches, identical accumulation schedules, a working common interconnect. Naively launched on this pool, distributed training either crashes at rendezvous, deadlocks in gradient sync, or silently trains N independent models.

This paper describes the systems we built to make the pool usable anyway, and the evaluation harness that turns trained checkpoints into a decision. Our contributions:

1. **A heterogeneous trainer** (`train_mdlm_moe_hetero.py`) combining (a) per-rank micro-batch autosizing from a real fwd+bwd VRAM probe, (b) exact per-token gradient weighting via `SCALE_I`, and (c) a global accumulation schedule that keeps all-reduce synchronous.
2. **A multi-family Slurm launcher** (`f1_hetero.slurm`): one job per GPU family, torchrun TCP rendezvous across jobs, NCCL transport fixes for mixed IB/no-IB fabrics, and a SIGUSR1 wave pattern for checkpoint + auto-resubmit under short partition limits.
3. **A falsification-oriented eval harness** (`harness/` + `eval_curve.py` + `verdict_f2.py`): reproducible pairwise discrimination on 256 real pairs drawn from the training corpus, per-checkpoint curves, and a pre-registered GO/NO-GO verdict.
4. **Measured, honest pool characterization**: per-family tok/s for the actual model (not projections from FLOPs), and a documented measurement pitfall that initially underestimated some families by 2–5×.
5. **Scavenged-pool continuous pretraining** (v5 phase): an elastic single-island DDP recipe — family-agnostic placement, VRAM-adaptive micro-batching, ZeRO-1 cross-world checkpoint resume, and a self-resubmitting wave chain — that carried a 1.13B MoE through ~1M steps on queue fragments, plus the local watcher/watchdog layer that turns checkpoints into a self-driving eval curve.
6. **An equal-dose falsification protocol** with two executed arms (fine-grained MoE, block-diffusion): pre-registered gates, shared init, identical budget and battery — a template for turning architecture questions into closed decisions at laboratory scale.

We report results as they are: the flagship heterogeneous run (F1, 17 ranks, L40+Q6000, 1000 steps) produced chance-level discrimination — a NO-GO against the pre-registered ≥0.55 criterion — and the contrast against a single-GPU micro-run isolates the cause as optimization configuration rather than scale.

## 2. Related Work

**Diffusion LMs.** Discrete diffusion for text was established by D3PM (Austin et al., 2021), with continuous-embedding variants such as Diffusion-LM (Li et al., 2022). Masked/absorbing-state formulations were simplified and scaled by SEDD (Lou et al., 2024) and MDLM (Sahoo et al., 2024); LLaDA (Nie et al., 2025) demonstrated masked diffusion competitive with AR LMs at 8B scale. Our models are MDLM-style masked-diffusion LMs spanning 154.8M dense to a 1.13B-parameter (285M-active) 16-expert MoE.

**Distributed training.** PyTorch DDP (Li et al., 2020) averages per-rank gradients over world size — correct only when ranks carry equal token counts. ZeRO/DeepSpeed (Rajbhandari et al., 2020) shards optimizer state but still assumes homogeneous data-parallel ranks. Elastic and volunteer-compute training (e.g., SWARM parallelism, Ryabinin et al., 2023; DiLoCo, Douillard et al., 2023) addresses unreliable or heterogeneous nodes via pipeline reshuffling or infrequent outer synchronization; our setting instead needs *exact* dense-synchronous semantics on a small model where communication is cheap relative to correctness. We are not aware of prior work deriving the exact per-token weighting correction for heterogeneous micro-batches under standard DDP all-reduce.

**Scheduling and containers.** Slurm (Yoo et al., 2003) gang-schedules homogeneous allocations; multi-family composition is left to the user. Apptainer/Singularity (Kurtzer et al., 2017) provides the container layer, though overlay + multi-rank `srun` interactions required a single-task + torchrun-inside design (§3.2).

## 3. Method

### 3.1 Model and objective

The model is a dense 154.8M-parameter masked-diffusion transformer: hidden 512, 8 layers, 8 heads, ff_mult 4, sequence length 768, vocabulary 126,080 (LLaDA tokenizer) + 1 MASK token. Training follows the absorbing-state objective: contiguous 64-token spans are masked at 15% ("span64@15%", the winner of our micro-sweep), and cross-entropy is computed on masked positions only. Data is a 129.5M-token pre-tokenized corpus of scientific-argument skeletons (315,299 documents with ≥3 labeled stages over five labels: OBSERVACION/HIPOTESIS/PREDICCION/EVIDENCIA/CONCLUSION), extracted by `build_skeleton.py` from IMRaD-structured and structured-abstract sources.

### 3.2 Heterogeneous trainer

Three mechanisms make a single optimizer semantics hold across unequal GPUs.

**Per-rank micro-batch autosizing (`--autosize`).** Each rank builds a probe model and runs a *complete forward + backward* on a 2-sample micro-batch, measuring peak allocated VRAM. The measured peak is decomposed as `peak ≈ fixed + b·per_sample` and extrapolated linearly against `total_memory × (1 − headroom)` (headroom 20%). The probe must include backward: the Linear→vocab head over 126,080 classes retains enormous logits only when gradients exist — a `no_grad` probe underestimates activation memory and produced a first-step OOM in early testing. The resulting micro-batch `micro_i` is clamped to `[min_batch, 512]` and is deterministic (a formula, not a search).

**Exact per-token weighting (`SCALE_I`).** Standard DDP computes `grad = (1/W)Σ_i grad_i` — an average over *ranks*. With unequal micro-batches this overweights small ranks. Each rank instead scales its local loss by

```
SCALE_I = world · micro_i / Σ_j micro_j        (Σ via one all-reduce at init)
```

Since DDP still divides by `world`, the reduced gradient becomes `(1/Σ_j micro_j)Σ_i micro_i·grad_i` — the exact per-token average over the global batch, identical to what a single large-batch GPU would compute. This is not an approximation; it is the same gradient, partitioned by VRAM.

**Global gradient accumulation.** `grad_accum` is identical across ranks (not per-rank adaptive). Accumulation counts differ per rank would place the DDP all-reduce on different global steps per rank — a deadlock. With `no_sync` on non-sync micro-batches, all ranks communicate on the same steps and the optimizer steps in lock-step. The batch shards are disjoint slices of the token stream per rank.

### 3.3 Multi-family launcher

Each GPU family is submitted as a separate Slurm job (`--nodes=1 --ntasks=1`), because an Apptainer overlay + multi-rank `srun` produced a rank-lock conflict; instead a single task launches `torchrun` inside the container, which spans all local GPUs and performs TCP rendezvous at `MASTER_ADDR:MASTER_PORT` shared by all jobs of the run. Two fabric problems had to be solved:

- **Mixed IB/no-IB nodes.** With an A100 (IB) master and an L40 node (`r32r25n01`, Ethernet-only) in the same job, NCCL chose IB and the L40 failed in `NCCLUtils.hpp:275`. Fix: `NCCL_IB_DISABLE=1` plus `NCCL_SOCKET_IFNAME` probed over `eth0/eth1/ib0/ens3/eno1`, with `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0`.
- **CUDA-stack split.** pro6000 (Blackwell) nodes run a cu128 PyTorch that cannot share a DDP process group with cu121 families (A100/L40/Q6000); they are excluded from mixed runs.

**Wave pattern.** The `sixhour` partition kills jobs at the wall clock; `#SBATCH --signal=B:USR1@300` delivers SIGUSR1 five minutes early, the trainer saves an atomic checkpoint (fcntl-locked, rank-0-only writer, tmp+rename, keep-last-2 retention) and exits 42, and the launcher auto-resubmits an identical job that resumes from `state.json`.

### 3.4 Evaluation harness

The harness turns checkpoints into a decision without touching a live trainer:

- `build_pairs.py` — 256 discrimination pairs from real skeleton docs (≥3 stages, seed 7331): `ctx` = stages 1..k−1 serialized, `ok` = the document's real stage k, `bad` = a same-position stage from a *different* document (plausible, inferentially wrong). Stored as pre-tokenized `{ctx, ok, bad}`.
- `suite_smoke.py` — scores each pair by masked-denoising loss under a fixed-seed mask; reports `pairwise_acc` = fraction where `L(ok) < L(bad)` and `mean_delta` = mean `L(bad) − L(ok)`; plus mask-predict iterative generation and fluency diagnostics (`rep4`, `uniq`). Same seed ⇒ comparable reports across runs.
- `run_micro.py` / `report.py` / `validate_configs.py` — declarative 1-GPU eval jobs, `report.json` + append-only `runs/index.jsonl` index, and config validation (a silent-eval-killing YAML bug motivated it).
- `eval_curve.py` — sweeps `checkpoint-gN` dirs, emits deduplicated `eval_curve.jsonl`.
- `verdict_f2.py` — automated verdict: HIT (acc ≥ 0.60), EXTEND (0.535 < acc < 0.60 with positive last-3 slope), FALSIFY (≤ 0.55 and below the 0.535 reference), NO-GO otherwise.

The pre-registered GO criterion is `pairwise_acc ≥ 0.55` on the 256 pairs.

## 4. Experiments

### 4.1 Pool characterization (measured, 2026-09-08)

[FIG: pool bar chart — measured tok/s per GPU family for the 154.8M dense model, sorted A100 > pro6000 > L40 > Q6000, annotated with VRAM.]

| Family | VRAM | GPUs on cluster | Measured tok/s |
|---|---|---|---|
| A100 (cu121) | 80 GB | ~18 | **~22,026** |
| pro6000 Blackwell (cu128) | 102 GB | 5 | **~12,553** |
| L40 (cu121) | 48 GB | 4 | **~11,443** |
| Q6000 (cu121) | 48 GB | ~29 | **~7,941** |

Method: timestamps in the training log, step 0 → step 190 at 6,144 tok/step (batch 8 × seq 768), excluding data-cache load. **Pitfall:** dividing by job walltime includes ~35–56 s of cache load and underestimates throughput 2–5× (an early L40 measurement read 2,560 tok/s — nearly 5× low). Two lessons: (a) never project tok/s from FLOPs for a small dense model — at batch 8 the model is compute-bound, not memory-bound (L40 used ~20 GB of 48), so raw device speed dominates and the A100 outperforms the newer Blackwell here, likely via the vocab-126,080 head; (b) per-VRAM batching alone does not capture the heterogeneity — throughput must be measured.

Projected cost of one skeleton-corpus epoch (129.5M tok): ~3 min on the theoretical max pool (56 GPUs, ~735K tok/s), ~8 min on a realistic 24-GPU slice (~279K tok/s), 15–22 min on L40+Q6000 alone (~100–140K tok/s).

### 4.2 Micro-sweep (F0): 6 single-GPU runs

Six micro-runs on 1× pro6000, 10K steps each (12,288 tok/update = batch 8 × accum 2 × 768 ≈ 122.9M tok ≈ 0.95 epochs), seed 7331:

| Run | Mask × data | pairwise_acc | mean_delta |
|---|---|---|---|
| f0-random-prosa | random 15% × v7 prose | 0.5117 | −0.032 |
| f0-span-prosa | span64@15% × v7 prose | 0.4492 | −0.064 |
| f0-spanhi-prosa | span64@60% × v7 prose | 0.4766 | −0.052 |
| f0-random-esqueleto | random 15% × skeleton | 0.5078 | +0.011 |
| **f0-span-esqueleto** | span64@15% × skeleton | **0.5352** | +0.015 |
| f0-spanhi-esqueleto | span64@60% × skeleton | 0.5312 | +0.030 |

No run met the ≥0.55 GO criterion (best: 0.5352, 1.5 p.p. short), but the pattern is informative: skeleton ≫ prose (prose runs show *negative* mean_delta — the model systematically prefers the wrong continuation), and on skeleton, span masking > random. Winner carried forward: span64@15% on the skeleton corpus.

### 4.3 F1: heterogeneous run and NO-GO

The F1 run exercised the full system: 5 jobs / 17 ranks (4× L40 + 13× Q6000), global batch 65,536 tok/update, lr 2e-4, warmup 100, 1000 optimizer steps (65.5M tok ≈ 0.5 epoch), completing within roughly three hours of shared-cluster wall time. The trainer, launcher, NCCL fixes, and wave machinery all worked as designed — the systems contribution was validated end-to-end.

[FIG: F1 discrimination curve — pairwise_acc (left axis) and mean_delta (right axis) vs step, 501–1000; horizontal lines at GO = 0.55 and micro-sweep reference 0.5352.]

| Step | pairwise_acc | mean_delta |
|---|---|---|
| 501 | 0.4844 | −0.0436 |
| 551 | 0.4883 | −0.0405 |
| 701 | 0.5000 | −0.0300 |
| 751 | 0.4961 | −0.0266 |
| 801 | 0.4922 | −0.0220 |
| 851 | 0.5039 | −0.0148 |
| 901 | 0.5078 | −0.0108 |
| 951 | 0.4961 | −0.0129 |
| 1000 | 0.5000 | −0.0066 |

`pairwise_acc` stayed at exact chance (~0.5; max 0.5078, final 0.5000). `mean_delta` improved monotonically (−0.044 → −0.0066): the model stopped preferring the wrong continuation but never learned to prefer the right one. Training loss plateaued at ~7.1 from ~step 100 — consistent with our standing rule that loss is not a proxy for the discrimination signal. **Verdict: NO-GO.**

### 4.4 The contrast: configuration, not scale

| | tok/update | updates | total tokens | acc |
|---|---|---|---|---|
| micro-span-esqueleto (winner) | 12,288 | 10,000 | 122.9M | **0.5352** |
| F1-hetero | 65,536 | 1,000 | 65.5M | 0.5000 |

Same corpus, same 154.8M model, same mask recipe, same lr. The only substantive difference is the optimization shape: the micro-run made ~10× more, smaller updates per token. Averaging each gradient over 65K tokens dilutes the sparse structural signal of the skeleton corpus, and with only 1K updates the model never sees each fine pattern enough times. The actionable conclusion is **not** "more GPUs"; it is *more small optimizer updates* — the exact micro recipe at 50K–100K steps.

### 4.5 From NO-GO to a verified inferential frontier (updated 2026-09-17)

The §4.4 prescription — many small updates, single GPU — was executed as a nine-generation evolutionary search (EVOG0–G8) over the masking recipe space, ~90 runs of 10K steps each (12,288 tok/update), selected on a frozen dev set and verified on a document-disjoint holdout. Two harness upgrades mattered for what followed: a **dense scoring mode** (mask 100% of the candidate stage, reconstruct conditioned on context — the shallow 15%-mask scorer had diluted the inferential signal ~6×) and a **leakage audit** (embedding-based dedup + source-document separation found the original holdout ~97% clean; the cleaned 7,509-item holdout is what we report).

The search converged on `hinrcf10`: *random* token masking drawn from U(0.30, b_h) with b_h ramped 0.30→0.95 by curriculum, *role_mask off*, and *candidate_focus* = 1.0 (every skeleton document masks one whole argument stage — the exact condition of the dense eval). Span masking, role-weighted masking, capacity (~150M), more steps (20K), schedules, and loss-weighting all tied or degraded; the frontier responds to *volume of inferential practice*, not architecture.

**Verified frontier (clean holdout, dense mode):** `hinrcf10` (n=2 seeds): L0 0.546 / L1 0.529 / L2 0.676 / **L3 0.632**, with per-subtype `number` 0.389 / `negation` 0.645 — L3 clears the pre-registered 0.55 falsification floor; the fine-logic subtypes (`number` ~0.31–0.42 across all configs) remain the open boundary. Cost note: the reasoning gain is not free — denoising pseudo-PPL degrades ~46% vs the earlier champion, so the model trades LM quality for discrimination.

**Corpus ablation (the ES-crudo lesson).** A v3 corpus augmented with 48.6K raw-Spanish UNAM thesis chunks degraded L3 to 0.583 (~5σ below the 0.626 backbone on the same dev set); the English-only sibling (c1e, 116.9M tok) recovered 0.627. Raw Spanish *helps* LM quality (better pseudo-PPL) while *hurting* English discrimination — the corpus enters translation-first (HY-MT1.5-7B, in-cluster, $0), never raw.

**The MoE inversion (v4), resolved by the recipe port (v4s).** Scaling to a 676M-param MoE (hidden 768, 12 layers, 8 experts top-1, ~279M active) continued-trained on the EN-augmented corpus (v4en, 118.5M tok) with the *flat* masking recipe inverted the profile: L0 0.696 / L1 0.634 (best surface discrimination to date) but L2 0.503 / **L3 0.496 — chance**. Porting the full `hinrcf10` recipe onto that same checkpoint (+6K steps; candidate-stage masking on skeleton docs, dense random masking on prose) **recovered the inferential levels while retaining the surface advantage**: L0 0.612 / L1 0.566 / L2 0.694 / **L3 0.600** on the clean holdout — and L3 0.627 on the dev set, tying the dense champion (0.626) with uniformly better surface discrimination and a much better denoising pseudo-PPL (926 vs ~2988). Negation jumped 0.290 → 0.574; `number` (~0.38) remains the hard floor across every configuration. The flat arm had *fallen* to chance over the same extra-step budget, so the delta is the objective, not the steps: the structured recipe transplants to a larger architecture and corpus, and the two regimes do not fundamentally compete — a single model can carry both surface fluency and inferential discrimination.

### 4.6 v5-1b: the flagship run on the scavenged pool (added 2026-10-09)

The v4s endpoint motivated the flagship question directly: if the recipe carries inference, what does it do at ~1B scale over a full corpus pass? `v5-1b` is a 16-expert top-1 masked-diffusion MoE — hidden 768, 12 layers, 1.13B total / 285M active parameters (the same per-token compute as the 676M dense design; top-1 routing buys capacity for free) — trained from scratch on `train_ids_v5pdb.npz`: 11.96B tokens / 17.38M documents (papers_db EN: 285K fulltext + 2.79M abstracts + 40.7K synthetic + 2K translated UNAM, plus 327K v4hneg skeletons, split-long into 768-token windows), target 1.3M optimizer steps at a constant 9,216 tokens/step.

The run's systems problem is not the model but the substrate: there is no allocation — only whatever `sixhour` islands happen to be free. Three mechanisms make continuous pretraining possible anyway:

**Elastic single-island DDP.** The launcher requests `gres=gpu:3` with an NVIDIA-family constraint and lands on *any* free island (Q6000, A100, L40, A40, Q8000, PRO6000). The trainer auto-adapts to what it finds: ≥40 GB VRAM → micro-batch 4, flat; <40 GB → micro-batch 1, grad-accum 4, gradient checkpointing, and **ZeRO-1 optimizer sharding** (the 1.13B fp32 model + AdamW needs ~18 GB fixed — it does not fit a 24 GB card unsharded; ZeRO-1 brings fixed state to ~12.4 GB/rank). Effective batch is held constant at 9,216 tok/step across every island shape, so the recipe is invariant to where the wave lands.

**Cross-config checkpoint resume.** Optimizer state is stored in standard ZeRO-1 format (consolidated on rank 0 after `zero_grad`); a wave that dies on 4 ranks resumes on 3 — verified in production. Combined with EMA (decay 0.999) saved alongside raw weights, every checkpoint is simultaneously a resume point and an eval point.

**The AUTO_RESUBMIT wave chain.** Each wave runs inside a six-hour window, checkpoints atomically on SIGUSR1, and resubmits itself; an early race (SIGTERM path exited without resubmit) was found and fixed by routing `finalize` through the TERM trap. The chain then ran for weeks unattended — including through a ~24 h cluster storage outage (10-07→10-08) that cost one day of compute and was absorbed without data loss.

Local **watcher daemons** close the loop: ssh-polling scripts (now supervised by a cron watchdog after a post-outage silent death) relaunch dead fleets and trigger downstream eval — the eval curve is produced *by the infrastructure*, not by an operator remembering to launch it. Throughput: ~47–90 K steps/day depending on which island the chain currently occupies (large NVIDIA islands are contended; a ~81-GPU AMD MI210 pool — 2.5× all fast NVIDIA combined — remains untouched pending a ROCm spike; measured inter-node bandwidth ~200 MB/s rules out naive multi-node DDP at this batch size: 4.5 GB of fp32 gradients per sync would cost ~22 s against ~1 s of compute).

### 4.7 The instrument crisis and the corrected battery (added 2026-10-09)

The v5 phase's most consequential engineering was not training — it was discovering that early instruments *lied*, and rebuilding them. The sequence is a case study in why eval infrastructure deserves first-class status:

1. **Leakage audit.** Embedding-based dedup plus source-document separation showed the original holdout ~97% clean — but only after auditing; the cleaned 7,509-item holdout is what all current numbers report.
2. **Dense scoring.** The original 15%-mask scorer diluted the inferential signal ~6×; masking 100% of the candidate stage (dense mode) is what exposed the frontier in §4.5.
3. **Context-necessity (ctxnec) battery.** Pairwise discrimination alone cannot distinguish *using* context from ignoring it. The corrected suite corrupts the context (entity/value/role/direction rebinds) and measures whether predictions track the corruption — plus a **corruption profile** (corrupt_0 → corrupt_0.5) whose endpoint delta is the "C4 drop", a fragility measure that later proved decisive for the §4.8 verdicts.
4. **Remask decoding.** An iterative decode (4 steps × 2 rounds × frac 0.25, confidence-ordered) provides a generative binding measure (`remask gx`) that pure pairwise scoring misses.
5. **Live-checkpoint race fix.** Batteries launched against *live* checkpoints raced the trainer's writes; the fix snapshots each checkpoint (hardlink) into a pinned dir before submitting eval jobs, plus a completeness filter on partially-written states.
6. **Known residual caveat.** One subtype (`entity`) currently audits at n=1 — a documented limitation, not a hidden one.

### 4.8 Equal-dose falsification protocol: two structural arms (added 2026-10-09)

With a trustworthy battery in place, the program could ask the structural question cleanly: *is the residual binding gap an architecture problem?* The protocol is deliberately rigid: warm-start each arm from the **same EMA checkpoint**, train the **same step budget** (30K), and evaluate on the **same battery** against the mainline point at equal total exposure — gates pre-registered before launch.

**Arm fgmode (fine-grained MoE).** 32 fine experts at ff/2 with top-2 routing + 1 shared expert at ff/4 — DeepSeekMoE-style specialization at equal FLOPs/token. Warm-start g316K + 30K steps. **Verdict: FLAT** — no primary metric moved beyond noise.

**Arm blkdiff (block-diffusion, BD3-LM-style semi-AR).** Context→target asymmetry wired into the attention mask, warm-start g683K + 30K. **Verdict: FALSIFIED with regression** — ctxnec pairwise 0.737→0.696 (−4.7pt, ~3.5σ), cons_inf 0.659→0.643, remask gx 0.328→0.270, gen_exact 0.277→0.242. The arm's one "improvement" — a record C4 drop of −10.1 — is on inspection *fragility*, not binding: the model became more dependent on clean context and worse at using it, so corruption cuts deeper into a degraded baseline. Every ctxnec subtype fell below mainline (role_swap 0.839 vs 0.89, entity_rebind 0.772 vs 0.82, value_rebind 0.618 vs 0.66).

The paired negatives close the architecture axis at this scale and dose: neither attention structure nor expert specialization is the binding bottleneck — consistent with the capacity/data reading that the exposure curve (§4.9) tests directly.

### 4.9 Exposure plateau and the external trajectory (added 2026-10-09)

The mainline's corrected curve runs from g207K and is still extending; the recent stretch defines the plateau:

| ckpt | denseL3 | cons_inf | ctxnec pw | gen_exact ctx | remask gx |
|---|---:|---:|---:|---:|---:|
| g710001 | 0.517 | 0.668 | 0.743 | 0.290 | 0.335 |
| g738065 | 0.517 | 0.665 | 0.738 | 0.293 | 0.335 |
| g763501 | 0.516 | 0.669 | 0.742 | 0.286 | 0.333 |
| g791001 | 0.506 | 0.652 | 0.743 | 0.279 | 0.333 |
| g816501 | 0.523 | 0.662 | 0.744 | 0.287 | 0.339 |

`ctxnec` pairwise has held a 0.728–0.744 band for ~250K steps; `cons_inf` ~0.66; `gen_exact` — the last live slope — flattened (three consecutive points below the 0.293 peak). Internal metrics say exposure stopped buying binding.

The external trajectory disagrees — usefully. On sciev (scientific-decision evals vs a LLaDA-8B-Instruct ceiling):

| eval | g316K | g495K | g738K | LLaDA-8B |
|---|---:|---:|---:|---:|
| choice elite (n=448) | 0.473 | 0.520 | **0.540** | 0.870 |
| GPQA main (n=441) | 0.256 | 0.247 | **0.293** | ~0.315 |
| GPQA diamond (n=193) | 0.275 | 0.269 | **0.326** | ~0.33 |
| SciFact score (n=304) | 0.286 | 0.398 | 0.368 | 0.624 |

`choice` keeps climbing (57% of the headroom to LLaDA-8B extracted); GPQA moved for the first time at g738K — the backbone essentially **tied LLaDA-8B** there (0.293/0.326 vs ~0.315/0.33). The honest reading: parametric knowledge continues accumulating even where the binding instruments plateau — the two measure different things, and the paper reports both rather than choosing one.

### 4.10 SFT as a measurement instrument (added 2026-10-09)

Two controlled SFT runs (same 36,677-pair sciq+smoltalk dataset, 2 epochs, LLaDA-recipe masked-response objective) bracket the backbone's maturation. On the held-out generative eval (eval_devin_hard, 3,000 items, gold_recall):

| tipo | sft-v1 (init g94K) | sft-v2 (init g738K EMA) |
|---|---:|---:|
| multihop (n=800) | 0.031 | 0.084 |
| numerical (n=1055) | 0.052 | 0.153 |
| negation (n=324) | 0.036 | 0.125 |
| causal (n=820) | 0.032 | 0.086 |

Parametric recall roughly tripled with backbone maturity — but a direct chat probe on the sft-v2 checkpoint produced degenerate looping output on all prompts (the chat template is respected; the content is not usable), and the binding battery shows a small real tax vs its init (ctxnec 0.738→0.733; gen_exact 0.293→0.263; remask 0.335→0.297). Verdict: **chat=no at 1.13B/this exposure** — SFT's value here is as a calibrated probe of what the backbone stored, and the sft-v1→v2 delta is the cleanest available curve of knowledge-vs-exposure on this model.

## 5. Discussion

[FIG: autosize + SCALE_I diagram — per-rank VRAM probe → micro_i → SCALE_I weighting → single synchronous all-reduce → one optimizer step equal to the global per-token average.]

[FIG: v5-1b systems stack — sixhour wave chain (AUTO_RESUBMIT + atomic ckpt + ZeRO-1 cross-world resume) → pinned snapshot → 3-job battery (dense / cons_inf / ctxnec+remask) → watcher daemons → cron watchdog; side lane: UNAM-EN fleet (scrape → curate → OLMo-13B vLLM translate → QA-gen → SFT-v3).]

**What the systems side enables.** The pipeline converts an idle, fragmented, heterogeneous slice of a shared cluster into a single exact-semantics data-parallel run in minutes, with measured (not projected) capacity, and survives queue preemption via the wave pattern. The SCALE_I derivation is a general recipe: it applies to any DDP workload where ranks must carry unequal micro-batches, requires no custom collectives, and preserves the exact optimizer semantics of a homogeneous run. The measurement pitfall (§4.1) is worth stating in print: small-model throughput on contended clusters is routinely mismeasured by job walltime, and the error is large (2–5×) and directionally misleading — it inverts the apparent ranking of hardware generations.

**The honest negative, and what it bought.** The flagship run falsified *this configuration*, not the thesis: discrimination stayed at chance, and the verdict tooling recorded a clean NO-GO against a pre-registered criterion. This is the designed behavior of a falsification-oriented harness — the experiment produced a decision with cheap evidence, and the micro-vs-F1 contrast localized the failure to optimization shape rather than to hardware, data, or masking objective. The payoff is documented in §4.5: the corrective the contrast prescribed (many small updates) is exactly what the subsequent single-GPU recipe search executed, and it converged on a verified inferential frontier. The negative was load-bearing twice — once as a verdict, once as a diagnosis.

**Negatives compound (v5 addendum).** The equal-dose protocol produced a second, matched pair of clean negatives (§4.8): with the instrument trusted, each arm cost 30K steps and returned a decision instead of an ambiguity. The deeper methodological claim of this paper is therefore not "our trainer works" but that *a small lab can run a falsification-grade research program on scavenged capacity* — provided the instrument layer is treated as load-bearing: every near-miss in this project that later resolved (the diluted scorer, the raced checkpoints, the leaky holdout, the fragility-vs-binding confound) was an instrument bug or instrument upgrade, not a model change.

**Operations as a contribution.** The wave chain + watcher + watchdog stack is unglamorous but is what converted ~40 GPU-days of fragmented sixhour windows into a single uninterrupted 1.3M-step run; its failure modes (the SIGTERM resubmit race, the silent watcher death, the storage outage) are documented because they are the actual cost model of this class of infrastructure.

**Limitations.** cu121/cu128 families still cannot share a process group, so Blackwell capacity is unreachable in mixed runs; per-rank data shards are static slices rather than a sampler; SCALE_I equalizes token weight but not step latency — the slowest rank still gates each synchronous step; measured inter-node bandwidth (~200 MB/s) confines v5-1b to single islands — multi-node only pays at ~×30 larger effective batch, which is a different recipe, not a free scale-up; the battery-curve gap g841K–g975K is permanent (rolling checkpoint retention); and the `entity` subtype audit remains at n=1.

## 6. Future Work (updated 2026-10-09)

The two items listed here in the 09-08 draft are resolved: the f2-spanes bridge run completed (0.5273, FALSIFY — archived), and the controller/verificator fallback was validated independently (98.2–98.6% arg-match on 500 real tool-calls). Current work:

1. **Close the v5-1b arc:** terminal battery and the 4th sciev point on the g1.3M checkpoint (expected ~10-13); then the UNAM-EN lane completes end-to-end — 4,873 curated theses → OLMo-2-13B vLLM translation (~89% done) → automated QA generation → SFT-v3 on the terminal checkpoint, giving the third point of the SFT-as-probe curve.
2. **Frontier push on `number`:** every configuration plateaus the fine-logic subtype near 0.38–0.42; candidate axes are fine-dosed corrective corruption (p≈0.02–0.05 — the p=0.10 dose showed adversarial gradients), hard-negative mining from the precomputed 16-NN edges, and objective-level changes rather than recipe tweaks.
3. **Capacity axis after the plateau:** with both structural arms falsified and internal metrics flat, the open levers are corpus (translation-first UNAM-EN integration; reasoning traces) and scale — the MI210/ROCm pool (~81 GPUs) is the untapped capacity, blocked only by an unverified software stack.
4. **Systems extensions:** cross-stack training (per-family homogeneous subgroups + activation-level parallelism or a unified cu128 build), throughput-aware micro-batch sizing (autotune by measured tok/s, not only VRAM, since the model is compute-bound), localSGD-style infrequent sync (`sync_every`) for batch-×30 runs that make multi-node pay on a ~200 MB/s fabric, and elastic membership to absorb mid-run pool fluctuation.

---

## Bibliography

1. J. Austin, D. D. Johnson, J. Ho, D. Tarlow, R. van den Berg. "Structured Denoising Diffusion Models in Discrete State-Spaces." *NeurIPS 2021*.
2. X. L. Li, J. Thickstun, I. Gulrajani, P. Liang, T. Hashimoto. "Diffusion-LM Improves Controllable Text Generation." *NeurIPS 2022*.
3. A. Lou, C. Meng, S. Ermon. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution." *ICML 2024*.
4. S. Sahoo, M. Arriola, Y. Schiff, I. Gokhale, E. Mehrvar, Z. Chen, A. Wang-Satterthwaite, C. Xiong, R. Feris, V. Kuleshov. "Simple and Effective Masked Diffusion Language Models." *NeurIPS 2024*.
5. S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, C. Li. "Large Language Diffusion Models." *arXiv:2502.09992*, 2025.
6. S. Li, Y. Zhao, R. Varma, O. Salpekar, P. Noordhuis, T. Li, A. Paszke, J. Smith, B. Vaughan, P. Damania, S. Chintala. "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." *VLDB 2020*.
7. A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS 2019*.
8. S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC 2020*.
9. M. Ryabinin, T. Dettmers, M. Diskin, A. Borzunov. "SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient." *ICML 2023*.
10. A. Douillard et al. "DiLoCo: Distributed Low-Communication Training of Language Models." *arXiv:2311.08105*, 2023.
11. J. Hoffmann et al. "Training Compute-Optimal Large Language Models." *NeurIPS 2022*.
12. A. B. Yoo, M. A. Jette, M. Grondona. "SLURM: Simple Linux Utility for Resource Management." *JSSPP 2003*.
13. G. M. Kurtzer, V. Sochat, M. W. Bauer. "Singularity: Scientific Containers for Mobility of Compute." *PLoS ONE 12(5)*, 2017.
14. NVIDIA. "NVIDIA Collective Communications Library (NCCL)." Software documentation, docs.nvidia.com.
15. M. Arriola, S. Gokhale, A. K. Sagar, A. Tong, Z. Yang, J. M. Hernandez-Lobato. "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models." *ICLR 2025*.
16. D. Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." *ACL 2024*.
17. T. OLMo et al. "2 OLMo 2 Furious." *arXiv:2501.00656*, AI2, 2025.
18. D. Rein et al. "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." *arXiv:2311.12022*, 2023.
19. D. Wadden et al. "Fact or Fiction: Verifying Scientific Claims (SciFact)." *EMNLP 2020*.
