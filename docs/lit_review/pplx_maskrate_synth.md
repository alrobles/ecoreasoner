## Bottom line

There is **no single universally best corruption rate or schedule**. The empirical picture is:

1. **Classic 15% MLM masking is a historical default, not an optimum.** For BERT-scale pretraining, **30–40% masking is usually a better region**, while even **80% masking can retain most downstream performance** if training is adjusted appropriately.
2. **Discrete diffusion LMs deliberately expose the model to the entire range from nearly 0% to nearly 100% masking**, rather than choosing one fixed masking rate. Their strongest results generally come from **absorbing-mask corruption**, with the exact schedule and loss weighting determining how much capacity is spent at low versus high noise.
3. **Independent random masking is the standard diffusion corruption**, because it gives a tractable factorized forward process and supports arbitrary infilling. Span/block corruption can improve learning of longer dependencies, but it is more task- and architecture-dependent.
4. The best-supported schedule trend is **not “emphasize the hardest, highest-noise examples.”** In language, performance often improves when training gives relatively more weight or sampling probability to **low-to-middle noise**, where accurate reconstruction is more difficult and the signal is more informative.
5. **Easy-to-hard curricula have promising evidence in generic diffusion models**, but there is still limited direct evidence that a strict easy-to-hard curriculum is optimal for large masked diffusion language models.

---

## 1. Masking rates: 15% versus 40–99%

### BERT-style masked language modeling

| Setting | Empirical result | Interpretation |
|---|---|---|
| BERT-large, 15% versus 40% | **40% outperformed 15% on GLUE and SQuAD** in the controlled study by Wettig et al. | 15% is not intrinsically optimal; higher corruption can produce a stronger pretraining signal. [web:31][web:35] |
| BERT-large, 80% | 80% masking preserved roughly **95% of the fine-tuning performance** of the 15% model, despite extremely poor reconstruction perplexity. | MLM reconstruction quality and downstream representation quality can diverge substantially. [web:31] |
| RoBERTa-base/large, up to 50% | Increasing masking from 15% to 50% allowed substantially cheaper pretraining **without GLUE degradation**; the authors report that **40% worked best on average**. | Around 40% is a strong practical default for encoder MLM pretraining. [web:38] |
| Dynamic schedule | A linearly decreasing schedule, approximately **30% → 15%**, improved average GLUE by up to **0.46 percentage points for BERT-base** and **0.25 points for BERT-large** over fixed-rate baselines. | Starting with harder corruption and ending with conventional corruption can be better than using a fixed rate. [web:39] |
| Other MLM experiments | Models trained with **25% or 35% masking** exceeded 15% models by more than 1 SQuAD F1 point in the reported experiments. | Moderate increases above 15% are repeatedly beneficial, although the optimum is implementation-dependent. [web:95] |

The main controlled study concludes that **40% is a better general choice than 15% for BERT-large**, while 80% remains surprisingly viable. However, 80% should not be interpreted as the new universal optimum: it retained most downstream quality but did not generally beat 40%. [web:31][web:98]

A useful distinction is:

- **Moderate high masking: 25–50%** — often improves downstream performance or training efficiency.
- **Very high masking: 70–80%** — can preserve strong representations, but usually makes the denoising problem much harder and does not consistently produce the best downstream scores.
- **99–100% masking** — important for diffusion generation as an endpoint, but generally not a good *single fixed MLM rate*, because the input contains little or no information from which to reconstruct the sequence.

Thus, for ordinary encoder MLM pretraining, the evidence favors **roughly 30–40% masking**, not either the historical 15% default or an always-near-total masking regime.

### Discrete diffusion language models

Masked diffusion models differ fundamentally from BERT MLMs: they do not normally train at one fixed masking rate. Instead, a noise level \(t\) is sampled and each token is independently masked with a probability determined by the schedule. This exposes the model to masking rates spanning approximately **0% to 100%**.

- **LLaDA** samples the masking ratio \(t\) uniformly from \([0,1]\), so the model sees all corruption levels, including nearly fully masked examples. [web:64][web:65]
- **MDLM** uses an absorbing-mask process and a weighted masked cross-entropy objective across noise levels. [web:16][web:14]
- **D3PM** established the discrete diffusion framework and found that the absorbing-state/mask transition was stronger for text than uniform categorical corruption in the reported experiments. Its absorbing model achieved substantially better LM1B perplexity than its uniform-noise counterpart: **76.9 versus 137.9** in one reported comparison. [web:46][web:47]
- **SEDD** also found the absorbing transition substantially better than a uniform transition; one reported One Billion Word comparison gives **32.79 perplexity for SEDD-Absorb versus 40.25 for SEDD-Uniform**. [web:77][web:83]

The relevant conclusion is therefore not that diffusion LMs prefer, for example, 80% masking. Rather:

> They benefit from training over the whole masking continuum, but the distribution of training effort over that continuum matters greatly.

Very high noise is necessary because generation starts from a fully masked sequence, but the model must also learn the low-noise regime in which small reconstruction errors and fine-grained local decisions dominate.

---

## 2. Uniform random masking versus span/block masking

### BERT and encoder MLMs

Independent token masking is attractive because it creates many prediction targets while preserving a broad context. Its main weakness is that neighboring tokens often make the answer locally easy.

Span masking removes contiguous context and forces the model to reconstruct longer units:

- **SpanBERT** masks contiguous spans while retaining a roughly 15% total corruption rate. It reported gains over comparable BERT baselines, including approximately **+2.0 and +2.8 SQuAD F1** over its BERT baseline on SQuAD 1.1 and 2.0, respectively. The gains were particularly relevant to span-sensitive tasks such as question answering and coreference. [web:91]
- **PMI-Masking** argues that uniform token masking encourages models to exploit shallow local signals. It masks statistically associated n-grams and reported better downstream performance than prior masking approaches on **SQuAD2.0, RACE, and GLUE**, while reaching comparable performance in roughly half the training time. [web:43][web:100]
- The advantage is not universal: one study reports that PMI-Masking performed worse than random masking on LAMA, showing that structured masking can trade off general lexical-probing performance for better relational or phrase-level learning. [web:94]

The practical trade-off is:

| Masking strategy | Main advantage | Main cost |
|---|---|---|
| Independent random tokens | Broad, simple, computationally convenient; many targets per example | Often leaves local clues and may under-train phrase/span reasoning |
| Random contiguous spans | Forces longer-range reconstruction and phrase-level representations | Fewer independent prediction sites; can make reconstruction unnecessarily difficult |
| Whole-word/entity/phrase/PMI spans | Better alignment with semantic units and relation-sensitive tasks | Gains are task-dependent; may hurt lexical or fact-recall probes |
| Block masking in diffusion | Can support structured or sequential generation and possibly cache reuse | Reduces the fully parallel, globally revisable nature of masked diffusion |

For ordinary encoder MLM, the evidence supports **mixing or selectively using spans when downstream tasks require phrase-level or relational reasoning**, rather than replacing random masking universally.

### Diffusion language models

Most MDLM, LLaDA, D3PM, and SEDD results use **independent token corruption**, not span corruption. This is not merely inherited convention:

- Independent masking gives a simple absorbing Markov process.
- Every token can be independently recovered at every denoising step.
- It naturally supports arbitrary infilling and parallel denoising.
- The forward and reverse objectives are mathematically tractable.

Consequently, the strongest direct empirical evidence in these papers concerns **the transition type and noise schedule**, not a head-to-head comparison of random-token versus span/block corruption.

Block or span diffusion changes the generation factorization. It may offer better compatibility with left-to-right serving or KV-cache reuse, but it restricts global revision and can reduce the ability to correct earlier decisions using later context. Therefore, the evidence currently favors:

- **Independent random masking** for general-purpose bidirectional MDLMs and arbitrary infilling.
- **Block/structured corruption** when serving constraints, streaming, or explicit span completion are primary objectives.
- No strong evidence yet that block masking universally improves language-model perplexity or downstream accuracy over independent masking in MDLMs.

---

## 3. Noise-level weighting and schedules

Let \(t\) denote the corruption level, with larger \(t\) corresponding to more masking. Two choices are easy to conflate:

1. **The forward schedule:** how masking probability changes with \(t\).
2. **The training weighting:** how frequently or strongly each noise level contributes to the loss.

They are not equivalent. Changing the sampling distribution over \(t\), changing the explicit loss weight, and changing the parameterization can produce different effective emphasis.

### Main schedules and empirical findings

| Model or study | Schedule / weighting | Empirical implication |
|---|---|---|
| D3PM | Tested several transition matrices and schedules; absorbing-mask corruption was strongest for text in the reported experiments. | The corruption *type* matters at least as much as the nominal masking rate. [web:46][web:47] |
| MDLM | Absorbing-mask diffusion with a weighted average of masked cross-entropies over noise levels. | Trains across the full noise range rather than at a single masking rate. [web:14][web:16] |
| LLaDA | Uniform \(t\sim U[0,1]\), with independent masking probability \(t\). | Equal sampling density in noise level, covering 0–100% masking. [web:65][web:69] |
| SEDD | Geometric and log-linear schedules were compared; the **log-linear schedule** was especially useful for SEDD-Absorb perplexity. | Allocating noise so that the expected number of changed tokens grows approximately linearly with time is better than an arbitrary schedule for the absorbing model. [web:77] |
| BabyLM masked diffusion study | Compared uniform/linear and cosine masking schedules. Cosine had mean masking around **0.36**, versus **0.50** for uniform/linear sampling, and consistently improved zero-shot likelihood-style results. | In that setting, emphasizing lower masking rates was better than treating all masking levels equally. [web:17][web:76] |
| Noise-schedule studies more generally | Importance sampling around intermediate log-SNR was more effective than simply increasing loss weights there under constrained compute. | Sampling allocation and explicit loss weighting should be optimized jointly. [web:13] |

### Does one want to emphasize high-noise training?

The empirical answer is **usually not exclusively**.

High-noise examples are essential because:

- They teach global reconstruction from sparse evidence.
- They make the model usable from the nearly or fully masked initial state.
- They prevent the model from becoming merely a local infiller.

But they are intrinsically information-poor. At 99% masking, the model often has little sequence-specific evidence, so the optimal prediction is close to a corpus prior. This makes the task useful for learning broad distributional structure but less useful for precise token-level discrimination.

The strongest language-specific results instead suggest emphasizing:

- **Low-to-middle noise**, where the model must make difficult, information-rich corrections.
- Or a **balanced schedule with explicit reweighting**, so high-noise examples remain present without dominating the objective.

The frequency-informed masked diffusion experiments explicitly report that a cosine schedule, which lowers the average masking rate from 0.50 to 0.36, improved zero-shot likelihood results over uniform scheduling. [web:17] This is evidence against the simple rule “more high-noise examples are always better.”

For MDLM-style objectives, the familiar inverse-noise-type weighting also reflects this asymmetry: low-noise steps contain fewer masked tokens, but each prediction can be more informative and harder to get exactly right. The effective weighting should therefore compensate for the changing number and difficulty of targets rather than merely count masked tokens.

---

## 4. Easy-to-hard curricula over noise level

A strict curriculum is conceptually appealing:

1. Start with low noise / easy reconstruction.
2. Gradually introduce higher noise.
3. Finish with the full 0–100% corruption range.

However, the empirical diffusion literature complicates this terminology. In many denoising systems, **high-noise examples converge faster and are considered easier**, because the target distribution is smoother; low-noise examples can be harder because they require precise reconstruction. A diffusion curriculum paper therefore organizes training from **higher timesteps/high noise to lower timesteps/low noise**, i.e. easy-to-hard in terms of denoising difficulty rather than visible corruption percentage. [web:4][web:6]

This leads to two different curricula:

| Curriculum | Order | Rationale |
|---|---|---|
| Corruption curriculum | Low masking → high masking | Starts with ordinary MLM-like examples and gradually teaches reconstruction from little context |
| Denoising-difficulty curriculum | High noise → low noise | Starts with smoother, easier denoising targets and gradually adds precise low-noise reconstruction |

The latter has stronger general diffusion evidence: the reported curriculum groups timesteps by difficulty and introduces them in descending ease, with experiments showing improved convergence and performance over simultaneous uniform-noise training. [web:4][web:6]

For masked diffusion LMs specifically, the evidence is less definitive:

- **LLaDA and MDLM generally train across the full noise range from the outset**, rather than using a strict curriculum. [web:14][web:65]
- Dynamic BERT MLM results support a **high-to-low masking schedule**—approximately 30% down to 15%—which is consistent with beginning harder and ending easier in terms of visible context, and it improves GLUE modestly. [web:39]
- The available diffusion-curriculum results suggest that an effective schedule should be based on **measured denoising difficulty**, not simply the numerical masking rate.

Thus, the most defensible recommendation is:

> Use broad noise coverage from the beginning, but allocate training adaptively or progressively toward the noise levels with the largest loss, slowest convergence, or greatest downstream sensitivity. If using a curriculum, high-noise-to-low-noise is better supported by diffusion-denoising evidence than a naive low-mask-to-high-mask curriculum.

---

## Practical recommendations by objective

| Objective | Recommended starting point |
|---|---|
| BERT/RoBERTa-style encoder MLM | Try **30–40% random masking** rather than automatically using 15%; test 25%, 40%, and 60% if compute permits. |
| Strong span reasoning, QA, coreference | Add **contiguous span or phrase masking**, possibly mixed with random token masking. SpanBERT-style objectives are especially well motivated here. [web:91] |
| General MDLM pretraining | Use independent absorbing-mask corruption with masking levels spanning **near 0% to 100%**. |
| LLaDA-like training | Uniformly sample the masking ratio over \([0,1]\) as a robust baseline. [web:65] |
| Compute-constrained diffusion training | Consider cosine or other schedules that allocate more examples to low-to-middle noise; the BabyLM experiments favor cosine over uniform in zero-shot likelihood evaluation. [web:17] |
| SEDD-like score modeling | Test absorbing versus uniform transitions and use a **log-linear schedule** as a serious baseline. [web:77][web:83] |
| High-noise robustness | Retain substantial high-noise exposure, but avoid allowing 90–100% masking to dominate the loss. |
| Curriculum | Prefer a measured easy-to-hard denoising curriculum—often high noise first, then lower noise—over a purely heuristic mask-rate ramp. [web:4][web:6] |

### Overall conclusion

For conventional MLMs, the best-supported answer is **approximately 40% masking**, with 15% remaining serviceable but rarely optimal and 80% surprisingly robust. For discrete diffusion LMs, the right analogue is **not one fixed rate**: train across the full masking continuum, use absorbing-mask corruption, and tune the schedule so that low-to-middle noise receives sufficient effective weight. Span/block masking is beneficial when the task demands longer-range reconstruction, but independent random masking remains the strongest general-purpose baseline for MDLM, LLaDA, SEDD, and D3PM-style systems.