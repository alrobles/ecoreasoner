# Literature review: RL post-training for masked diffusion LMs

## Executive conclusion

**RL post-training of masked diffusion LMs is now technically feasible, but the strongest evidence is still at the 1.7B–8B scale and mostly assumes an already capable pretrained model.** The original **d1** result showed that task-specific RL can improve an 8B masked diffusion LM, while **d2** substantially improved the likelihood-estimation problem and reported strong results without extra chain-of-thought SFT. However, there is not yet convincing evidence that a **~100M model trained from scratch can acquire strong mathematical reasoning from RL alone**. A 100M-scale diffusion LM can certainly be pretrained and RL-tuned mechanically, but the likely bottleneck is the quality and diversity of the pretrained policy—not merely the RL algorithm.

The most important conclusions are:

1. **d1/diffu-GRPO:** practical first demonstration of GRPO-style RL for masked dLLMs; uses an approximate one-step likelihood estimator and random prompt masking.
2. **d2:** replaces d1’s crude mean-field likelihood with trajectory-aware estimators; its best 8B results are much stronger, especially on Sudoku and Countdown.
3. **Compute:** d1 used LoRA/4-bit training and 8×A100-80GB for RL, but online sampling remained the major cost. d2 reports compute formulas and matched-FLOP comparisons rather than a simple total GPU-hour number.
4. **Reasoning gains:** d1’s gains were modest on math and large on logical/constraint tasks; d2 reported much larger gains, especially Sudoku.
5. **Small models:** d2 includes a 190M model, but its demonstrated task is toxicity steering, not mathematical reasoning. The reasoning experiments are principally 1.7B and 8B.
6. **Reward models:** there is work using external classifiers, AR perplexity models, correctness verifiers, and token-level verification rewards, but relatively little work training a conventional discriminative process-reward model whose native input is a partially denoised diffusion state.

---

## 1. d1: SFT followed by diffu-GRPO

### Method

d1 adapts **LLaDA-8B-Instruct**, an 8B masked diffusion LM, using:

1. **Masked SFT** on the 1,000-example `s1K` reasoning dataset.
2. **diffu-GRPO**, a critic-free policy-gradient method adapted from GRPO to masked diffusion generation. [web:2]

The key difficulty is that a masked diffusion model does not factorize a sequence naturally as

\[
\log p(x\mid q)=\sum_t\log p(x_t\mid q,x_{<t}),
\]

because multiple tokens may be generated in parallel and the order of unmasking is dynamic.

d1 therefore uses:

- a **mean-field approximation** to sequence likelihood;
- one-step per-token probability estimation;
- random masking of prompt tokens, with masking probability \(p=0.15\);
- multiple inner policy updates using different masked views of the same prompt-completion pair.

The random masking is important computationally: it acts as both regularization and data augmentation, allowing more gradient updates per sampled rollout and thereby reducing the number of expensive online generations.

### d1 training scale and compute

The reported configuration was:

| Component | Configuration |
|---|---|
| Base model | LLaDA-8B-Instruct |
| Parameters | 8B |
| SFT data | `s1K`, 1,000 reasoning problems |
| SFT hardware | 2× NVIDIA A6000 |
| SFT duration | 20 epochs, 2,460 steps |
| SFT sequence length | 4,096 |
| SFT adaptation | LoRA, rank 128 |
| RL hardware | 8× NVIDIA A100-80GB |
| RL sequence length | 256 tokens |
| RL adaptation | LoRA, rank 128; 4-bit quantization |
| RL steps | GSM8K 7,700; MATH500 6,600; Countdown 5,000; Sudoku 3,800 |
| RL learning rate | \(3\times10^{-6}\) |
| Online rollout length | Limited to 256 tokens because generation was expensive |

These are relatively modest parameter-update requirements because the backbone is frozen or nearly frozen through LoRA, but they should not be confused with cheap full-model training. The expensive part is repeated online diffusion sampling and likelihood estimation. d1 explicitly notes that scaling RL to substantially longer reasoning traces remained difficult because of slow online generation. [web:2]

For context, the underlying LLaDA-8B pretraining reportedly used approximately **2.3 trillion tokens and 0.13 million H800 GPU-hours**; that is foundation-model pretraining cost, not d1 post-training cost. [web:54]

### d1 benchmark results

The following are d1’s reported zero-shot accuracies; the main comparison is against the released LLaDA-8B-Instruct checkpoint.

| Model | GSM8K, 512 | MATH500, 512 | Countdown, 512 | Sudoku, 512 |
|---|---:|---:|---:|---:|
| LLaDA-8B-Instruct | 78.2 | 36.2 | 16.0 | 5.5 |
| LLaDA + SFT | 81.1 | 34.8 | 23.8 | 4.6 |
| LLaDA + diffu-GRPO | 81.9 | 39.2 | 37.1 | 11.0 |
| **d1: SFT + diffu-GRPO** | **82.1** | **40.2** | **42.2** | **9.5** |

The paper summarizes the d1 gains over the base model as approximately:

| Benchmark | Absolute gain reported by d1 |
|---|---:|
| GSM8K | +3.9 points |
| MATH500 | +4.0 points |
| Countdown | +26.2 points |
| Sudoku | +10.0 points |

Thus, d1’s improvement is **not uniform**:

- On GSM8K and MATH500, the gain is useful but relatively modest.
- On Countdown and Sudoku, which require strict constraint satisfaction and benefit from self-correction, the gains are much larger.
- SFT alone is unstable: it improves some settings but can hurt others.
- RL alone is stronger than SFT alone in most reported configurations.
- SFT followed by RL is generally the best recipe, although not in every individual sequence-length cell. [web:2]

The main limitation is that d1’s likelihood estimator is only an approximation. The mean-field assumption effectively treats token probabilities as conditionally independent given the prompt, which is poorly matched to a multi-step denoising trajectory.

---

## 2. d2: trajectory likelihood estimation

### What d2 changes

d2 argues that accurate trajectory likelihoods are central to policy optimization for masked diffusion models. It introduces two estimators:

1. **d2-AnyOrder**
   - Intended for any-order diffusion models.
   - Packs clean and masked tokens into a single attention structure.
   - Computes an exact or unbiased trajectory-likelihood estimate in one model pass for the supported architecture.

2. **d2-StepMerge**
   - Intended for ordinary masked diffusion models such as LLaDA.
   - Divides the full denoising trajectory into \(N\) contiguous segments.
   - Uses \(N\) model evaluations instead of evaluating every denoising step.
   - \(N=1\) is effectively the d1/diffu-GRPO approximation.
   - Larger \(N\) improves likelihood accuracy at greater compute cost.

The approximation error is bounded in terms of the number of skipped denoising steps and the change in token predictions inside each block. The reported practical compromise for Sudoku was \(N=16\). [web:3]

### d2 compute and configurations

d2 reports more explicit FLOP accounting than d1. For sequence length \(L\), group size \(G\), denoising steps \(T\), model size \(P\), and \(N\) likelihood segments, the approximate components are:

- on-policy sampling: \(LGT(2P)\);
- old-policy likelihood: \(LGN(2P)\);
- reference-policy likelihood: \(LGN(2P)\);
- current-policy likelihood and backpropagation: \(LGN(4P)\).

The LoRA overhead is treated as negligible relative to the frozen backbone. [web:3]

For the main LLaDA experiments, d2 used:

- LLaDA-8B-Instruct;
- LoRA rank 128, \(\alpha=64\), dropout 0.05;
- 4-bit quantization;
- group size 6;
- two tokens decoded per timestep;
- learning rate \(3\times10^{-6}\);
- prompt masking probability 0.15;
- generation lengths of 128–512 depending on task;
- matched or controlled FLOP budgets.

The paper does **not** give a single clean “total GPU-hours” number or a total RL-step count for the main d2 experiments. It reports configurations and FLOP formulas instead. [web:3]

### d2 reasoning results

The headline LLaDA results are:

| Method | Sudoku | Countdown | GSM8K | MATH500 |
|---|---:|---:|---:|---:|
| LLaDA | 11.8 | 19.9 | 75.7 | 35.4 |
| LLaDA 1.5 | 12.5 | 23.4 | 78.6 | 36.8 |
| d1 | 22.1 | 42.2 | 82.1 | 40.2 |
| wd1 | 25.2 | 51.2 | 82.3 | 39.0 |
| **d2** | **91.9** | **56.6** | **85.0** | **41.6** |

The d2 result is particularly striking on Sudoku:

- d1: 22.1%
- d2: 91.9%

The authors also report that d2 improves over diffu-GRPO on Sudoku, Countdown, and GSM8K, with a smaller but positive trend on MATH500. Importantly, the main d2 LLaDA comparison claims **no additional supervised chain-of-thought fine-tuning**; d2 is applied directly as RL post-training to LLaDA-8B-Instruct. [web:3]

On the 1.7B Qwen3 Set Diffusion model:

| Method | GSM8K |
|---|---:|
| Block-diffusion SFT | 48% |
| Any-order SFT | 62% |
| Any-order SFT + diffu-GRPO | 63% |
| **Any-order SFT + d2-AnyOrder** | **67%** |

On Dream-7B:

| Method | Sudoku | GSM8K |
|---|---:|---:|
| Dream 7B | 14% | 71% |
| Dream + diffu-GRPO | 79% | 73% |
| **Dream + d2-StepMerge** | **91%** | **80%** |

The reported improvement of d2-StepMerge over diffu-GRPO on Dream is therefore +12 points on Sudoku and +7 points on GSM8K. [web:3]

### Interpretation

d2 suggests that a major fraction of d1’s limitation was not “RL does not work for diffusion,” but rather that **the policy-ratio and KL terms were being estimated too crudely**. Better trajectory likelihoods produce substantially better credit assignment and policy updates.

However, d2’s large Sudoku gain should be interpreted carefully:

- Sudoku is a highly structured constraint task and may benefit disproportionately from improved denoising trajectory credit assignment.
- It does not establish a comparable 70-point improvement on general mathematical reasoning.
- GSM8K and MATH500 gains are more modest: roughly +9.3 and +6.2 points relative to the listed LLaDA baseline, respectively.

---

## 3. Related work

### MRO: multi-reward optimization

MRO uses LLaDA-8B and combines:

- a final answer/correctness reward;
- a token-verification reward;
- an AR-language-model perplexity reward;
- rejection sampling or REINFORCE;
- step-wise group reward optimization.

Its **token verification reward** masks one generated token at a time and asks the diffusion model to reconstruct that token from the remaining response. This is a form of self-consistency or leave-one-out verification, rather than a separately trained discriminative reward model.

MRO reports the following LLaDA results:

| Method | GSM8K, 512 | MATH500, 512 | Countdown, 128 | Sudoku, 64 |
|---|---:|---:|---:|---:|
| LLaDA | 79.4 | 34.4 | 14.1 | 11.2 |
| MRO rejection sampling | 82.6 | 36.2 | 22.0 | 17.2 |
| MRO RL | 81.8 | 37.4 | 27.2 | 20.2 |

Thus MRO-RL gives, relative to vanilla LLaDA:

- +2.4 GSM8K points;
- +3.0 MATH500 points;
- +13.1 Countdown points;
- +9.0 Sudoku points.

MRO also reports that its token-verification reward is particularly effective, while the perplexity reward can increase variance. [web:31]

### SPG and later policy-gradient methods

Later methods such as **SPG**, **GDPO**, **DiFFPO**, and related denoising-aware policy-gradient approaches attempt to improve one or more of:

- trajectory likelihood estimation;
- token-level credit assignment;
- stability of policy ratios;
- denoising-step efficiency;
- reward assignment to parallel token updates.

SPG reports improvements over prior dLLM RL methods of up to approximately:

- +3.6 points on GSM8K;
- +2.6 points on MATH500;
- +18.4 points on Countdown;
- +27.0 points on Sudoku. [web:68]

These results reinforce the general pattern: **logical/constraint tasks often show larger relative gains than standard math benchmarks**, likely because reward signals are more exact and the diffusion model’s parallel editing ability is directly useful.

### Process and denoising rewards

Other work explores rewards attached to intermediate denoising states rather than only to the final answer. Examples include:

- token-level verification;
- process reward models;
- denoising feedback;
- reward-weighted sampling;
- inpainting-guided policy optimization.

This direction is attractive because a final correctness reward is extremely sparse for a multi-step denoising trajectory. Intermediate rewards can potentially tell the model which partially filled states are promising before the final answer is available.

---

## 4. Are there discriminative reward models inside diffusion LMs?

### Short answer

**Yes, but the literature is still heterogeneous, and most examples are external or hybrid rather than a conventional diffusion-native discriminative process reward model.**

The main categories are:

### A. External discriminative classifiers

d2 includes a 190M-parameter masked diffusion model, **Eso-LM**, in a toxicity-steering experiment. The reward is supplied by a pretrained toxicity classifier. d2-AnyOrder substantially outperforms DDPO under matched compute, moving the toxicity score close to the target while DDPO remains far away. This demonstrates that diffusion-LM RL is not limited to exact math rewards, but it is not evidence of mathematical reasoning in a 190M model. [web:3]

### B. External AR perplexity models

MRO uses a pretrained autoregressive LM to score the perplexity of intermediate responses. This is a discriminative-ish quality signal, but it is not a learned task-specific verifier and does not directly judge mathematical correctness. [web:31]

### C. Task verifiers and correctness rewards

d1 and MRO use exact or semi-exact task rewards:

- answer matching;
- equation validity;
- Sudoku cell correctness;
- required-format checks.

These are verifiers, but not learned discriminative reward models.

### D. Diffusion-native token verification

MRO’s token-verification reward is generated by the diffusion model itself: mask a token, predict it from its neighbors, and use the probability as a compatibility reward. This is arguably a **discriminative score over partial sequences**, but it is not a separately trained reward network.

### E. Process reward models

More recent work applies an off-the-shelf process reward model to intermediate diffusion trajectories or denoising states. Such methods are closer to the requested “discriminative reward model inside diffusion LMs,” but the reward model is generally external to the diffusion LM and evaluates text states rather than being jointly trained as a diffusion-native discriminator. [web:34]

### Bottom line on reward modeling

The field has demonstrated:

- exact programmatic verifiers;
- toxicity classifiers;
- AR perplexity critics;
- token-level self-verification;
- external process reward models.

What is still comparatively underdeveloped is a **small, diffusion-native, discriminative process reward model trained directly on partially masked/noisy sequences**, with labels indicating whether the current denoising state is on a correct reasoning trajectory. Such a model could provide dense credit assignment without requiring the policy itself to estimate all trajectory probabilities.

---

## 5. Is RL-on-dLLM feasible for a ~100M model trained from scratch?

### Engineering feasibility: yes

A 100M masked diffusion LM is entirely feasible to pretrain from scratch. Public examples include masked diffusion models around 142M parameters trained on TinyStories, and other work reports models in the 169M range. [web:78][web:86]

RL fine-tuning at that size would also be computationally manageable:

- the model fits comfortably on a single modern GPU;
- LoRA is unnecessary or very cheap;
- rollout groups can be large;
- exact or high-\(N\) likelihood estimation is more affordable;
- reward computation can dominate only if the verifier is large.

### Reasoning feasibility: not yet established

The evidence is weaker for **strong reasoning**:

| Evidence | Scale | What it demonstrates |
|---|---:|---|
| d1 | 8B | Math and logical reasoning after SFT + RL |
| d2 reasoning | 1.7B and 8B | Math/logical reasoning with improved trajectory likelihood |
| d2 small model | 190M | Toxicity steering, not math reasoning |
| Scratch-trained small diffusion LMs | ~140–170M | Language modeling/pretraining feasibility, not RL reasoning |

Thus, a 100M model trained from scratch could plausibly learn a narrow task such as:

- 4×4 Sudoku;
- small Countdown;
- arithmetic word problems with short traces;
- formal-language or constraint-satisfaction tasks.

But it is unlikely to match 1.7B–8B models on GSM8K or MATH500 unless it receives unusually strong supervision, a narrow domain, or extensive distillation.

### Why RL alone is unlikely to be enough

RL cannot recover reasoning knowledge that the initial policy almost never samples. A 100M model trained from scratch is likely to have:

- weaker language understanding;
- lower probability of producing valid chain-of-thought formats;
- lower exploration quality;
- less ability to exploit sparse correctness rewards;
- greater sensitivity to reward hacking and formatting shortcuts.

The most plausible successful recipe is therefore:

1. pretrain the 100M masked LM on a sufficiently large and diverse corpus;
2. perform instruction or reasoning SFT/distillation;
3. use exact verifiers or a compact process reward model;
4. apply d2-style trajectory-aware RL;
5. focus first on narrow, verifiable tasks rather than broad mathematical reasoning.

### Compute expectation

The important distinction is:

- **Pretraining compute:** still substantial and scales roughly like ordinary LM pretraining; diffusion mainly provides inference-time parallelism, not a free reduction in training FLOPs.
- **Post-training compute:** potentially modest at 100M, especially with LoRA or full-parameter updates.
- **Rollout cost:** lower per forward pass than an 8B model, but diffusion requires multiple denoising passes, and RL needs many sampled completions.
- **Reward-model cost:** can dominate if using a large external PRM or AR evaluator.

A reasonable research-scale experiment would be feasible on a small multi-GPU setup, but a serious from-scratch 100M reasoning model still needs a meaningful pretraining token budget. The published literature does **not** yet provide a demonstrated “100M from scratch + diffusion RL → strong GSM8K/MATH500” result.

---

## Overall assessment

| Question | Assessment |
|---|---|
| Does RL work for masked diffusion LMs? | Yes; d1, d2, MRO and subsequent methods demonstrate this. |
| Is d1’s diffu-GRPO computationally practical? | Yes for LoRA post-training, but online rollouts remain expensive and likelihood estimates are approximate. |
| Is d2 a meaningful technical improvement? | Yes; trajectory-aware likelihood estimation substantially improves policy optimization, especially on constraint tasks. |
| Are gains real on math? | Yes, but generally several points rather than the dramatic gains seen on Sudoku/Countdown. |
| Are gains larger on logic/constraints? | Clearly yes in the reported results. |
| Are 100M models feasible to pretrain? | Yes. |
| Is strong RL reasoning demonstrated at 100M? | No. The closest d2 small-model evidence is 190M toxicity steering, not math reasoning. |
| Are discriminative reward models being used? | Yes, mainly external classifiers, AR perplexity models, process reward models, and task verifiers. |
| Is there a mature diffusion-native PRM architecture? | Not yet; this remains an open and promising direction. |

**Most defensible conclusion:** RL-on-dLLM is feasible at 100M as an engineering exercise and may work well for narrow, exactly verifiable tasks, but the current evidence does not support expecting a scratch-trained 100M masked diffusion model to develop broad mathematical reasoning through RL alone. The strongest near-term recipe is **competent pretraining + reasoning distillation/SFT + d2-style trajectory-aware RL + dense or discriminative process rewards**.