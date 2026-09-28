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
instrument (σ≈2-2.4pts at these n). Extensions to g193001: ctrl2 and corr
COMPLETED (ckpts preserved under `runs/contrast/ckpts/`); qa timed out at
g188501 and was relaunched.

## 8c. CONTEXT-NECESSARY PAIRS (2026-09-24)

Post-C4 instrument: `harness/build_pairs_ctxnec.py` — negatives use
context-attested material so only binding decides truth:

| strategy | n | mechanism |
|---|---|---|
| value_rebind | 608 | number → another number attested in ctx |
| entity_rebind | 334 | entity → same-class entity attested in ctx |
| role_swap | 211 | A<rel>B → B<rel>A, both attested in ctx |
| direction_rebind | 47 | direction swap only if antonym in ctx |

### Results (n=1200)

| scorer | ctxnec | (orig inferable) |
|---|---|---|
| b2-ctrl2@g193001 | 0.710 | 0.652 |
| b2-qa@g188001 | 0.705 | 0.657 |
| b2-rbd@g198001 | 0.705 | — |
| b2-corr@g193001 | 0.700 | 0.652 |
| LLaDA-8B | 0.513 ≈ chance | 0.507 |

### C4 profile on ctxnec — the binding probe

| arm | 0% | 10% | 25% | 50% | drop |
|---|---|---|---|---|---|
| ctrl2 | 0.710 | 0.696 | 0.689 | 0.618 | −9.2 |
| qa | 0.705 | 0.691 | 0.673 | 0.621 | −8.4 |
| rbd | 0.705 | 0.691 | 0.673 | 0.621 | −8.4 |
| corr | 0.700 | 0.692 | 0.676 | 0.614 | −8.6 |
| (orig pairs) | ~0.66 | ~0.66 | ~0.65 | ~0.62 | −3.5 |

**Reading**: ctxnec pairs degrade 2.5× more under context corruption than
the original set — they DO engage context. But ~0.62 survives at 50%
corruption: the honest decomposition of our model's "verification" is
≈0.62 plausibility prior + ≈9pts context binding. The binding component is
real but small; the plateau is mostly a plausibility ceiling.

### Follow-up launched

**b2-rbd** (job 30237078): `--rebind_p 0.7` added to the trainer — numeric
corrupt positions take replacements from other numbers IN THE SAME
SEQUENCE (ctxnec-as-objective). Init b2-corr@g193001, +5K steps on the
identical base stream. Prediction to falsify: if the plateau is missing
binding supervision, b2-rbd should lift ctxnec accuracy AND steepen the
C4 profile beyond what ctrl2/corr show; a flat result falsifies the
objective hypothesis at this dose.

### b2-rbd result (2026-09-25) — objective hypothesis FALSIFIED at this dose

b2-rbd@g198001 (EMA, +5K steps done): ctxnec **0.705**, C4 profile
0.705/0.691/0.673/0.621 (drop **−8.4**). Flat vs every control arm —
no accuracy lift over ctrl2/corr, no profile steepening (all drops
inside −8.4..−9.2; σ≈1.3pts at n=1200). In-sequence rebind supervision
during corrective corruption did not move the binding component.
Reading: 5K steps of ctxnec-style objective at rebind_p=0.7 does not
break the ~0.62 plausibility prior — either the dose is far too small
or the binding gap lives in the eval/decoding surface, not the
objective. Elimination ranking for the next lever: exposure (v5-1b
extension — found stopped 09-25: optimizer state_steps device-mismatch
on cross-island resume killed wave 30252378; fix deployed, wave
relaunched) > architecture (block diffusion) > objective dose.

## 8d. EXPOSURE POINT g184K + INFERENCE-TIME ARMS (2026-09-25)

Mainline v5-1b@g184001 (EMA, ~37.5K steps / ~290M tokens after the
g146.5K baseline) on the corrected instruments:

| instrument | g146501 | g184001 | Δ |
|---|---|---|---|
| consistency-inf (n=1628) | 0.652 | **0.658** | +0.6 (sub-σ) |
| gen_exact_ok (inf) | 0.296 | **0.294** | −0.2 (flat) |
| ctxnec pairwise (n=1200) | — | **0.688** | — |
| ctxnec C4 drop | — | **−7.2** | in-family (−8.4..−9.2 arms) |
| inf C4 drop | ~−3.5 (arms) | **−3.9** | flat |

**Reading**: the 0.65 plateau is not moving with exposure at this window
(+37K mainline, and ctrl2@g193K=0.652 closes the other side). Quokka
caveat stands (~13% of one epoch is early), but the local slope is ~0 —
exposure alone does not break the plateau within a 25K-step horizon, so
non-exposure levers must carry the next gain.

**EoS-latent padding (arXiv 2603.05197) — FALSIFIED for this skill.**
ctxnec@g184001 with k EOS pads appended after the candidate:
pad0/32/128 → pairwise 0.6875/0.6892/0.6883, gen_exact_ok
0.2308/0.2333/0.2283 — all inside noise. Latent-compute slots do not
help a model whose bottleneck is context binding, not compute depth.
Cheap test, clean null.

**Verifier decoding (E-obj2) — PARTIAL POSITIVE.** `remask` mode
(steps=4, rounds=2, frac=0.25) on v5-1b@g185001:

| set | gen_exact_ok 1-paso | gen_exact_ok remask | Δ | gen_exact_bad |
|---|---|---|---|---|
| ctxnec (n=1200) | 0.231 | **0.274** | **+4.3 ≈ +3.3σ** | 0.073→0.064 |
| inf (n=1628) | 0.294 | 0.286 | −0.75 (null) | 0.081→0.033 |

Iterative refinement helps exactly where the pair needs context binding
(ctxnec) and is flat where plausibility suffices — the decode surface
DOES hold extra binding capacity the 1-step argmax leaves on the table.
`gen_exact_bad` nearly halves on inf (0.081→0.033): the loop rejects
mutations rather than polishing them. Subtype lift (gen_exact_ok,
1-paso→remask): entity_rebind 0.275→0.374 (+9.8), role_swap
0.038→0.115 (3×, aun así el peor), direction_rebind 0.362→0.370 y
value_rebind 0.263→0.266 (planos). Escalation: dose-response
(steps=8/rounds=4/frac=0.3) on ctxnec running; if it keeps rising,
remask decoding becomes the eval standard for capability claims and a
candidate for inference-time verification.

Ops fixes: trainer keeps ~2 ckpts → pinned evals raced to deletion;
pinned evals now snapshot to `runs/contrast/ckpts/`, and the battery
watcher (repaired evaled-parse bug) launches consistency+profile on
inf+ctxnec at every curve point — the corrected-metric exposure curve
is now automatic (next point ~g206K).

## 8e. EXPOSURE CURVE g207K–g288K + remask dose-response (2026-09-27)

The automatic battery now delivers corrected-instrument points every
~25K steps of mainline training (EMA evals):

| ckpt | dense L3 | cons_inf | gen_exact_ok inf | ctxnec | gen_exact_ok ctx | ctx C4 drop |
|---|---|---|---|---|---|---|
| g184001 | — | 0.658 | 0.294 | 0.688 | 0.231 | −7.2 |
| g207501 | 0.546 | 0.657 | 0.307 | 0.690 | 0.240 | −6.7 |
| g234001 | 0.533 | 0.657 | 0.302 | 0.701 | 0.244 | −7.8 |
| g261001 | 0.539 | 0.652 | 0.305 | 0.702 | 0.248 | −7.0 |
| g288001 | 0.527 | 0.654 | 0.320 | 0.712 | 0.255 | −8.4 |

**Reading (104K-step window):** `cons_inf` is dead flat (0.652–0.658,
inside σ≈1.1pt) — the plausibility ceiling does not move with exposure.
The honest metric tells a different story: `gen_exact_ok` creeps up on
both sets (+2.6pts inf, +2.4pts ctxnec over the window, ~2σ cumulative),
and ctxnec pairwise gained +2.4pts. The model is slowly improving at
*producing* forced content while its ranking stays saturated. Exposure
is not a breakthrough lever but its slope is not zero either — the
generative numbers are the only ones still moving.

**Remask dose-response — saturates at first dose.** E-obj2 at doubled
dose (steps=8/rounds=4/frac=0.3 vs 4/2/0.25) on ctxnec@g185001:
gen_exact_ok 0.2738→0.2782 (+0.4pt, ≪σ), gen_exact_bad 0.0643 (same),
subtype deltas all inside noise. The +4.5pt remask gain is a one-shot
refinement — more iterations do not extract more binding. Decision:
adopt `remask` (4/2/0.25) as the reporting standard for generative
claims; no further dose escalation.

Ops: the g313001 battery point failed at 20s — the watcher's `latest`
and both slurms' `CKPT_DIR` picked the checkpoint *directory*, which
exists ~2min before `model.pt` is renamed into it (observed save
sequence model→opt→ema). Fix deployed 09-27: completeness filter
(`checkpoint-g*/model.pt` glob) in watcher + both eval slurms, and
`scripts/battery_point.sh` now hardlink-snapshots model.pt+ema_model.pt
to `battery_curve/ckpts/g<N>/` and pins `CKPT_DIR` — the measured
checkpoint exactly matches the curve label and survives trainer pruning
(same inode, no extra space).

## 8f. ARM LAUNCHED: v5-1b-moefine — DeepSeekMoE (2026-09-27)

The architecture lever is now active (roadmap A2): **fine-grained MoE +
shared expert** as the controlled comparison against the v5-1b mainline.

- **Config**: 32 routed experts @ff/2, top-2 + 1 shared expert @ff/4
  always-on (vs 16@ff top-1). Routed FLOPs/token identical
  (2×ff/2 = ff); total params 1144.1M vs 1129M (+1.3%); active
  293.9M vs ~279.5M (+5%, the shared expert's cost).
- **Warm start**: `--init_from` partial load from the pinned EMA
  g316001 — copies key+shape-matching tensors (emb/pos/attn/ln/head =
  294/1698 tensors, 223M params); gate/experts/shared reinit. Same
  recipe, same data stream → directly comparable at equal added dose.
- **Why this arm**: pairwise consistency is flat under exposure; the
  capacity-specialization hypothesis is the cheapest structural probe
  (finer experts = less redundancy, shared = common knowledge path).
  Falsification budget: 30K steps (~276M tokens, dose comparable to
  the b2 arms).
- **Launch**: job 30587597, 3×L40, bs4 → ~9.2K tok/s, auto-resubmit
  waves to g30000. Step-0 loss 4.54 (vs ~11.7 random-init → warm
  start real; experts relearning).
- **Eval hooks**: `EVAL_CFG=eval-moe-v5-moefine.yaml` +
  `RUN_OUT=runs/v5-1b-moefine` on the same battery
  (`battery_point.sh`, `c_contrast.slurm`, `g_eval_holdout_v5.slurm`)
  → identical instruments (dense/cons_inf/cons_ctxnec) on pinned
  snapshots under `runs/v5-1b-moefine/battery_curve/`.
- Files: `scripts/v5_1b_moefine.slurm`, `scripts/smoke_moefine.slurm`,
  `harness/configs/eval-moe-v5-moefine.yaml`, MoEMLP/Block/MdLMMoE +
  `--n_shared/--fine_ff_div/--shared_ff_div/--init_from` in
  `scripts/train_mdlm_moe_v2.py`.

Decision rule: if moefine lifts `gen_exact_ok`/`cons_ctxnec` above the
mainline's matched-dose trajectory → adopt as the new trunk; if flat,
the capacity-MoE route is falsified and block diffusion (§9) becomes
the structural candidate.

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
09-25       v5-1b @g184K         0.658            0.294   <- exposicion plana
09-26       v5-1b @g207-234K     0.657            0.302-0.307
09-27       v5-1b @g261-288K     0.652-0.654      0.305-0.320
            └── pairwise saturado; gen_exact es la unica pendiente != 0 ──┘
```
