# EcoReasoner — Progress Report & Metric Verdict

Date: 2026-09-23 · Scope: from-scratch dLLM pivot (2026-08-25 → present, ~4 weeks)
Status: active. Frontier model `v5-1b` training (g171K+/1.3M). Paired battery
extended to +20K steps running.

---

## 1. Mission

Build, from scratch, a masked diffusion language model that (a) performs
context-bound scientific inference — verifies a claim against visible
evidence, derives conclusions forced by premises — and (b) chats usefully.
Corpus (~12B tokens scientific skeleton text) and HPC infrastructure are
solved; capability is the open variable.

## 2. Infrastructure delivered

- **Trainer** (`scripts/train_mdlm_moe_v2.py`): MDLM absorbing-mask denoising,
  MoE 16×top-1, DDP multi-node, ZeRO-1, EMA, wave orchestration
  (SIGUSR1 + auto-resubmit + atomic checkpoints), curriculum masking,
  candidate-focus stage masking, role-weighted masking, corrective
  corruption, contrastive hinge loss, MRAS adaptive masking.
- **Fleet ops**: VRAM-adaptive batching, island targeting
  (Q6000/L40/A40/A100/PRO6000), migration watchdog.
- **Corpus**: PMC OA S3 pipeline (~3M articles), v5pdb 12B-token skeleton
  corpus, embedding-based dedup/leak audits, ngram leak checker.
- **Eval harness**: L0–L3 pairwise battery, `consistency` + generative argmax
  modes, subtype breakdown, inferable/provenance split.

## 3. The instrument crisis (pivotal discovery)

The canonical `dense` metric masked the **entire candidate** — the ok/bad
inputs were literally identical, so it measured "reconstruct a plausible
argument," not "verify a slot against context." Consequences:

- `number` subtype was **structurally unwinnable** (72% provenance-only)
- `entity` collapse was n=1 noise
- 11 generations of evolutionary search had converged on the optimum of a
  broken instrument; "structural ceiling ~0.40" was a measurement artifact

## 4. Corrected metrics (A1/A2) and re-baseline (A3)

- **`consistency`**: context visible, only the mutated span masked,
  **sum-CE** (fixed a real length bias — `number` moved 0.42→0.66 on
  correction).
- **Generative argmax**: does the model *produce* the correct slot content?
- **Split**: 1,628 inferable / 139 provenance pairs (all provenance =
  `number`).

Re-baseline on inferable L3 (clean holdout):

| ckpt | dense (old) | consistency | gen_exact |
|---|---|---|---|
| hinrcf10 (155M) | 0.632 | 0.531 | 0.000 |
| retrain_v4s (676M) | 0.600 | 0.561 | 0.007 |
| **v5-1b (1.1B, 280M act)** | 0.531 | **0.652** | **0.296** |
| sft-v1 | 0.492 | 0.582 | 0.227 |

The ranking **inverted**: v5-1b — mediocre under dense — is the real
frontier, and the only model that *generates* forced answers (~30%).

## 5. Falsifiable experiments run

| experiment | intervention | inferable L3 | verdict |
|---|---|---|---|
| B2-synth | +100K synthetic derivation docs @20.5% tok | 0.635 vs ctrl 0.654 | **FALSIFIED** |
| B2-qa | +35.7K real extractive QA @20.8% | 0.657 vs ctrl2 0.652 | sub-σ positive |
| B2-corr | corrective_p=0.08 + hinge 0.5 | 0.652 = ctrl2 | null at dose |

All interventions plateau at ~0.65. Extensions to +20K steps running
(jobs 30103851-53). Remaining suspects: duration, scale (280M active),
or architecture/decoding.

## 6. Assets

- `corpus_deriv_v1.jsonl`: 100K synthetic derivation docs — 100%
  forcedness-verified, max mold 1.1%, 0 ngram-leak
- `deriv_qa_v1.jsonl`: 35,684 span-verified real extractive QA
- Mixed sets: `train_ids_b2base` 420M / `b2deriv` 529M / `b2qa` 531M tok
- Docs: REPLANTEO-2026-09-22, DERIV-CORPUS-SPEC, OBJETIVO-ARQ-2026-09-22

---

## 7. VERDICT ON OUR METRIC — where `consistency` can lie

The corrected metric is the best instrument we have, but it is not neutral.
Documented biases, by severity:

### 7.1 Fluency-vs-truth confound (high severity)

Pairwise accuracy rewards picking the *fluent/typical* continuation. A model
can score well by detecting "which option reads like corpus text" rather
than verifying the slot against evidence. The two abilities correlate on
inferable pairs but are NOT identical — a fluency detector would also score
well on provenance pairs (it does: ~0.68). **The gap inferable−provenance
(−0.03) is suspiciously small: our "verification" signal may be mostly
stylometry.**

### 7.2 Subtype dominance (medium)

`direction_word` is n=778 of 1628 inferable pairs (48%). The headline 0.65
is largely "does the model know if the result went up or down." The weakest
subtypes — `negation` (0.57-0.63) and `causal_phrase` (0.57-0.59) — sit at
the true capability boundary and are diluted in the aggregate.

### 7.3 Pseudo-joint artifact (medium)

Sum-CE over a masked span treats masked positions as conditionally
independent — it is not the true joint conditional likelihood for
multi-token spans (TER/SER theory, arXiv 2502.09622). Unequal span lengths
between ok/bad can still inject residual bias; this is the likely source of
the below-chance `negation` readings.

### 7.4 Mutation-engine style bias (medium)

All pairs come from ONE builder with ONE mutation grammar. The model may
learn "what a mutation looks like" (lexical fingerprint) rather than "what
truth looks like." An adversarial mutation style would deflate scores; an
easier one inflates them.

### 7.5 Metric divergence is itself a warning

`pairwise 0.65` vs `gen_exact 0.30` — the 2× gap between ranking and
producing means the ranking metric *flatters* the skill. Any capability
claim must cite the generative number.

### Verdict

Keep `consistency`-sum as the **primary paired-experiment metric** (it is
sensitive, fast, and fixed the dense pathologies) — but it measures
"preferring the plausible-and-correct option," not verification in the
strong sense. **No capability claim should rest on it alone.**

## 8. CONTRAST METRIC BATTERY — the anti-blindness instruments

Four independent instruments, each attacking a different bias of §7:

| id | instrument | bias attacked | cost |
|---|---|---|---|
| **C1** | **Fluent-negative pairs**: mutations rewritten to be grammatically natural in both variants (e.g., "rose by 30%"→"rose by 45%", not ungrammatical breaks) | 7.1 fluency confound | builder update + 1 eval job |
| **C2** | **External scorer cross-check**: LLaDA-8B-Instruct (independent, 8B) scores the same pairs; report agreement κ and our model's acc conditioned on LLaDA's answer | 7.4 builder-style bias, plus sanity anchor vs a real dLLM | eval-side only, ~hours |
| **C3** | **Domain-transfer holdout**: rebuild pairs from a held-out domain slice (physics/social, out of the eco/med training mass) | distribution overfit to eval domain | pairs rebuild + 1 eval job |
| **C4** | **Multi-timestep consistency profile** (DiffScore-style, arXiv 2605.11601): score the mutated span at k context-masking rates; report the *curve*, not a point | 7.3 pseudo-joint + fragility of a single t | eval code change |

**Decision rule (pre-registered)**: an intervention only counts as a win if
it (i) beats control +2σ on `consistency` inferable AND (ii) does not
regress on ≥2 of the four contrast instruments. A pairwise gain that
evaporates under C1/C3 is declared a metric artifact, not a capability.

## 8b. CONTRAST BATTERY RESULTS (2026-09-23)

Implementation: `harness/build_pairs_fluent.py` (C1/C3), `--hf_model` backend
(C2, LLaDA-8B-Instruct local snapshot), `consistency_profile` mode (C4),
`scripts/c_contrast.slurm`. Pair sets: `runs/pairs_l3_fluent` (n=1200),
`runs/pairs_l3_phys` (n=387, all `phys-*` docs held out), `runs/pairs_l3_inf`
(n=1628). Checkpoints: b2-corr@g173001, b2-qa@g175501, b2-ctrl2@g178001
(EMA; ckpt cleanup forced slightly different steps — comparability caveat).

### C1 — fluent negatives (n=1200, closed-class grammatical swaps)

| scorer | pairwise |
|---|---|
| b2-ctrl2 | **0.751** |
| b2-qa | 0.750 |
| b2-corr | 0.750 |
| LLaDA-8B | 0.604 |

Our model does *better* on fluent negatives than on the hard builder
(0.75 vs 0.65). Fluency is NOT the channel the metric rewards — the
negatives are grammatical and the model still separates them, and an
external 8B dLLM agrees they are the easier set (0.604 > 0.507).

### C2 — independent scorer (LLaDA-8B-Instruct, same pairs)

| pair set | LLaDA-8B | ours (ctrl2) |
|---|---|---|
| inferable (n=1628) | **0.507 ≈ chance** | 0.652 |
| fluent (n=1200) | 0.604 | 0.751 |
| phys-holdout (n=387) | 0.631 | 0.669 |

LLaDA-8B is at chance on our inferable pairs but above chance on both
contrast sets. Reading: the hard-pair negatives encode a *builder/corpus
fingerprint* our model learned and a general dLLM cannot see — but our
model's edge survives domain transfer and fluency control, so the learned
discrimination is not purely artifact.

### C3 — domain holdout (phys-*, n=387)

| arm | pairwise |
|---|---|
| b2-ctrl2 | 0.669 |
| b2-qa | **0.685** |
| b2-corr | 0.656 |
| LLaDA-8B | 0.631 |

Accuracy holds on a domain never seen in training or eval construction —
the discrimination is not locked to bio/eco conventions. `b2-qa` leads on
the held-out domain (+1.6 vs ctrl2), consistent with its small inferable
edge — still sub-σ (n=387, σ≈2.4pts).

### C4 — context-corruption profile (inferable, n=1628)

| arm | 0% | 10% | 25% | 50% |
|---|---|---|---|---|
| b2-ctrl2 | 0.662 | 0.660 | 0.652 | 0.624 |
| b2-qa | 0.659 | 0.654 | 0.647 | 0.616 |
| b2-corr | 0.650 | 0.659 | 0.647 | 0.620 |

**This is the most diagnostic result of the battery.** Destroying half the
non-slot context costs only ~3-4 points. If the pairwise decision were
context-bound verification, corrupting the context should cripple it —
instead the model barely notices. The metric is largely ranking
*candidate-side plausibility*; context contributes maybe ~4pts of signal.

### Metric verdict (updated)

**Partially trustworthy — and weaker than hoped.** What the 0.65 IS:
real discrimination that generalizes across domains (C3), is not a fluency
confound (C1), and beats an independent 8B dLLM (C2). What it is NOT:
evidence of context-forced inference — C4 shows the decision is nearly
insensitive to context integrity. The "verification" framing of §4 is not
supported; "domain plausibility ranking" is the honest description.

Consequences: (a) `gen_exact` (0.30) is the more honest measure of
context-forced derivation — it requires *producing* the slot; (b) next
pair sets must enforce context-necessity — e.g., pairs where both
candidates are equally fluent and only context disambiguates (C4-flat
curves are the signature to reject); (c) a C4 profile gate should join the
pre-registered decision rule: a claimed reasoning gain whose curve is flat
is not context-bound.

### Arms status

All three interventions remain statistically indistinguishable on every
instrument (σ≈2-2.4pts at these n). Extensions to g193001 running
(ctrl2 g178K+, qa g176K+, corr g173K+) — duration test still open.

---

## 9. Literature synthesis (2025-2026, post-pivot)

### Is it just training time? — strongest signal

**Quokka** (arXiv 2510.03280, first DLM scaling law): DLMs are **2-5× more
data-hungry than AR models** at equal compute; data-constrained validation
is U-shaped in epochs (`e_opt ∝ U_D^0.39/N^0.55`). v5-1b has seen ≈1.6B
tokens (~13% of ONE epoch over the 12B corpus) — we are genuinely early.
The 0.65 plateau may be exposure, not a capability ceiling. The running
+20K extension directly tests this.

### Better metrics

**DiffScore** (arXiv 2605.11601): masked-reconstruction eval with
multi-timestep profiles + PMI fluency/faithfulness decomposition +
structured masking — validates our paradigm and supplies the C4 design.

### Different designs (the field's convergent answer)

- **Block diffusion**: LLaDA2.0 (arXiv 2512.15745 — 100B MoE via AR→dLLM
  conversion, 3-phase block-size WSD) and Fast-dLLM v2 (NVIDIA, ~1B-token
  adaptation, hierarchical KV cache). Prime E-arq candidate.
- **"Masks Can Be Distracting"** (arXiv 2511.21338): MDLMs show *less*
  primacy bias and more global context use than AR — our context→slot
  verification plays to the paradigm's strength.
- **TER/SER theory** (arXiv 2502.09622): sequence-level correctness needs
  denoising steps ∝ length — formally explains our pairwise/generative gap.

### Reasoning post-training (the roadmap exists)

- **d1 / diffu-GRPO** (NeurIPS 2025, arXiv 2504.12216, open code): masked
  SFT + critic-free GRPO — our trace-SFT plan is its stage 1; diffu-GRPO
  is the natural stage 2.
- **Theory** (arXiv 2510.13117): MDMs ≡ padded looped transformers — can
  express everything CoT can, sometimes more efficiently.
- **EoS latent reasoning** (arXiv 2603.05197): masked dLLMs compute inside
  EoS-token representations — padding generation with EoS improves GSM8K
  and two-hop reasoning. **Free inference-time trick, testable on v5-1b.**

### Inference-time self-correction

**ReMDM** (NeurIPS 2025, arXiv 2503.00307, open code): remasking sampler —
generate → remask low-quality tokens → refine. Improves factual/reasoning
tasks. Our planned verifier-decoding (E-obj2) is ReMDM-conf with the
consistency score as the confidence signal.

## 10. Open questions (ranked)

1. **Duration/exposure** — does the plateau break with more tokens?
   (extension jobs running; mainline checkpoint evals at g200K+)
2. **Scale** — is 280M active the ceiling for this skill?
3. **Architecture** — block diffusion if 1-2 fail.
4. **Chat** — gated on 1-3; trace-SFT + diffu-GRPO is the roadmap.

## 11. Trendline (corrected metric)

```
date        model            consistency-inf   gen_exact
09-14       hinrcf10 155M        0.531            0.000
09-17       retrain_v4s 676M     0.561            0.007
09-22       v5-1b @g147K         0.652            0.296
09-23       b2-ctrl2/qa/corr   0.652-0.657      0.284-0.303
            └── plateau ~0.65 under attack ──┘
```
