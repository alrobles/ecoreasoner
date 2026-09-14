# Literature review: block/semi-autoregressive diffusion LMs and corruption curricula for reasoning

**Bottom line:** The clearest evidence that a curriculum over corruption granularity improves reasoning in diffusion LMs comes from **DreamReasoner-8B** and, independently, **T⋆ (Progressive Block Scaling)**. Both progressively increase block size from fine-grained to coarse-grained decoding and report better mathematical-reasoning performance and/or training stability than direct large-block training. **BD3-LM** and **SSD-LM** establish the semi-autoregressive/block-diffusion design space, but neither studies reasoning curricula. In conventional text MLM pretraining, **MRD/PTW** and **concept-based curriculum masking** improve GLUE, SQuAD, NER, or related inference tasks, but the evidence for improved *reasoning* in the stronger chain-of-thought/math sense is indirect. **DSFT** provides more direct reasoning evidence, although it is diffusion-model supervised fine-tuning rather than MLM pretraining, and its curriculum component is not independently sufficient on all tasks.

## 1. Core block and semi-autoregressive diffusion models

| Work | Model/training structure | Main contribution and findings | Reasoning/curriculum evidence |
|---|---|---|---|
| **SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control** — Han, Kumar & Tsvetkov, ACL 2023 | Generates text in blocks autoregressively; performs diffusion in the vocabulary simplex within each block; supports local bidirectional context and flexible output length. | Demonstrates that simplex-space diffusion can support classifier guidance and modular control without adapting the classifier. On unconstrained generation, SSD-LM matches or exceeds GPT-2 baselines on reported quality/diversity metrics and outperforms prior diffusion baselines. [web:1] | **No curriculum or reasoning study.** SSD-LM is an architectural predecessor: sequential across blocks, bidirectional/diffusive within blocks. |
| **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models** — Arriola et al., ICLR 2025; **BD3-LM** | Factorizes the sequence autoregressively over blocks, while using masked discrete diffusion inside each block. Block size interpolates between token-level AR decoding and full-sequence diffusion. | Introduces efficient blockwise training, variance-reducing masking/noise schedules, vectorized training, KV caching, arbitrary-length generation, and improved likelihood relative to earlier diffusion LMs. The paper reports better generative perplexity than MDLM and substantially fewer function evaluations than SSD-LM in its comparisons. [web:31][web:32] | **No reasoning curriculum.** The paper evaluates perplexity, generation quality, length, and efficiency—not CoT, mathematics, or reasoning benchmarks. Its important relevance is conceptual: block size is an explicit **corruption/parallelism granularity** parameter. |

### BD3-LM’s position in the design space

BD3-LM partitions \(x\) into blocks \(x^1,\ldots,x^B\) and models

\[
p_\theta(x)=\prod_b p_\theta(x^b\mid x^{<b}),
\]

with discrete denoising diffusion used to parameterize each block conditional. Thus:

- block size \(1\) approaches ordinary autoregressive modeling;
- intermediate block sizes produce semi-autoregressive generation;
- a block covering the whole sequence approaches standard full-sequence diffusion.

The model uses block-causal attention and can cache previously generated blocks, retaining causal long-range conditioning while sampling tokens within the active block in parallel. [web:31]

This makes BD3-LM an important **substrate for curriculum learning**, but the original paper does not establish that varying block size during training improves reasoning.

## 2. Direct evidence for block-size curricula in diffusion reasoning models

### 2.1 DreamReasoner-8B

**DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models** — Wu et al., arXiv:2606.19257, June 2026 — is the most direct match to the question. The paper studies how training and inference block sizes affect long-chain-of-thought reasoning and reports that direct large-block training severely harms reasoning. [web:16][web:40]

The training formulation preserves autoregressive factorization between blocks but performs local bidirectional diffusion inside each block. The curriculum is:

1. train initially with a small block size, typically **4**;
2. transition to larger blocks, including **32**;
3. use mixed-granularity training over \(\{4,8,16,32\}\) in the final stage.

The motivation is that small blocks force the model to learn local causal dependencies and fine-grained reasoning patterns, whereas large blocks demand more simultaneous inference and can cause reasoning degradation.

### Evidence from the block-size ablation

The reported AIME results show a large gap between fixed small-block, fixed large-block, and curriculum training:

| Training regime | Inference block size | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Fixed block size 4 | 4 | 47.1 | 37.5 |
| Fixed block size 4 | 32 | 42.5 | 39.6 |
| Fixed block size 32 | 4 | 20.0 | 24.2 |
| Fixed block size 32 | 32 | 42.5 | 21.3 |
| Curriculum \(4\rightarrow32\) | 4 | **50.0** | **43.8** |
| Curriculum \(4\rightarrow32\) | 32 | **48.3** | **38.3** |

The important result is not simply that block size 4 is better. Rather, **curriculum training makes the model robust to both fine- and coarse-grained inference**, while direct training at block size 32 can collapse under small-block inference and perform poorly even at large-block inference. [web:16][web:40]

The final model also shows relatively stable performance across inference block sizes. DreamReasoner-8B reports AIME 2024 scores from **73.8 at block size 4** to **68.3 at block size 32**, and AIME 2025 scores from **65.0** to **63.3**, while maintaining competitive code-generation performance. [web:16]

**Interpretation:** DreamReasoner provides strong evidence for a *granularity curriculum*: fine-grained denoising first, followed by progressively more parallel/coarse denoising. However, the curriculum changes **block size**, not the underlying token corruption law or diffusion noise schedule. The authors do not establish that every form of corruption-granularity curriculum will help; the result is specific to blockwise diffusion reasoning.

### 2.2 T⋆: Progressive Block Scaling

**T⋆: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory-Aware Reinforcement Learning** — Xia et al., ACL 2026 — independently studies progressive block-size scaling. It starts from an AR-initialized small-block masked diffusion model, gradually increases block size, and re-optimizes the denoising policy at every stage. [web:42]

The abstract reports that T⋆:

- enables higher-parallelism decoding;
- limits degradation on mathematical reasoning benchmarks;
- consistently outperforms direct large-block TraceRL;
- is substantially more stable during training across two SDAR model scales and three benchmarks.

Unlike a simple supervised block-size schedule, T⋆ combines progressive granularity with trajectory-aware reinforcement learning. It therefore supports the same broad conclusion as DreamReasoner—**large parallel decoding is easier to acquire after fine-grained competence has been established**—but suggests that the policy may need to be re-optimized as granularity changes.

### 2.3 Related but weaker diffusion evidence

Several adjacent works modify masking or denoising structure for reasoning, but do not provide the same clean block-size curriculum comparison:

- **DSFT** uses number-first masking, span masking, an adaptive masking-ratio curriculum, and number-weighted loss for diffusion-model supervised fine-tuning. Its masking ratio increases linearly from **10% to 20%**. On LLaDA-1.5, the complete DSFT recipe improves GSM8K from **75.59 to 79.37**, MATH from **23.74 to 26.14**, GPQA from **29.69 to 31.50**, BBH from **51.27 to 51.99**, and ARC-C from **56.57 to 57.74**. [web:85]  
  However, the isolated curriculum-masking ablation is mixed: it raises GSM8K to **76.20** but lowers ARC-C slightly to **56.46**. Thus, the paper supports **curriculum plus information-aware masking and loss weighting**, not a standalone claim that masking-ratio curriculum alone improves reasoning.

- **LogicDiff** uses a dependency-ordered denoising scheduler—premises, connectives, derived steps, then conclusions—which is a curriculum over *logical dependency order*, not a pretraining block-size curriculum. [web:65]

- **Noise Dependent Granularity Control** proposes using diffusion noise level as a granularity cue: coarse groups at high noise and token-level refinement at low noise. This is conceptually compatible with DreamReasoner, but it is not the same as a demonstrated fine-to-coarse training curriculum. [web:47]

## 3. Curriculum masking in conventional MLM pretraining

### 3.1 Masking Ratio Decay and POS-Tagging Weighted Masking

**Learning Better Masking for Better Language Model Pre-training** — Yang, Zhang & Zhao, ACL 2023 — is the most directly relevant conventional MLM study. It proposes two time-varying masking methods:

1. **Masking Ratio Decay (MRD):** begin with a high mask ratio—approximately twice the conventional 15%—and decay it toward a low ratio or near zero.
2. **POS-Tagging Weighted (PTW) Masking:** adapt mask probabilities according to the model’s cumulative loss for different part-of-speech categories, preferentially masking persistently difficult words. [web:57][web:100]

The rationale is a broad-to-fine curriculum:

- high corruption early encourages global reconstruction and exploration;
- lower corruption later enables more precise local prediction;
- loss-weighted masking focuses training on currently difficult linguistic categories.

### Results

For BERT-base trained for 1M steps, cosine MRD improves the GLUE average from **81.1 to 82.8** and SQuAD F1 from **90.2 to 90.9** relative to fixed 15% random masking. PTW improves GLUE from **78.5 to 79.2** and SQuAD F1 from **87.1 to 88.1** for BERT-base trained from scratch; in the BERT-large continued-training setting, it improves GLUE from **83.4 to 83.9**, SQuAD from **91.3 to 91.6**, and CoNLL F1 from **94.9 to 95.4**. [web:57]

These are meaningful improvements on inference-oriented tasks such as:

- MNLI/RTE textual entailment;
- QNLI question-answer entailment;
- SQuAD extractive question answering;
- CoNLL named-entity recognition.

Nevertheless, the paper does **not** evaluate mathematical reasoning, chain-of-thought generation, theorem proving, or multi-step reasoning directly. The correct conclusion is that MRD/PTW improve downstream language understanding and inference, with reasoning relevance mainly through entailment and QA.

### 3.2 Concept-based Curriculum Masking

**Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking** — Lee et al., EMNLP 2022 — constructs an easy-to-hard masking curriculum using ConceptNet. It first masks frequent, highly connected concepts, then progressively adds concepts related to earlier masked concepts. [web:127]

The schedule uses:

- four curriculum stages;
- two-hop ConceptNet expansion;
- whole-concept masking;
- approximately 15% total masked tokens per example;
- an MLM warm-up before the curriculum stages.

The method improves BERT-base’s GLUE average from **80.4 to 82.3**, with gains on CoLA, SST, MNLI, QQP, QNLI, and most other reported tasks, although RTE decreases from **67.8 to 65.0** in the main BERT-base result. The authors also report comparable GLUE performance at roughly half the training cost and a **1.9-point** GLUE improvement at comparable computational cost. [web:127]

This is evidence that **which concepts are corrupted, and in what order, matters for transfer**, but it is not evidence for improved chain-of-thought reasoning. Its strongest relevance is to semantic inference, entailment, and general NLU.

### 3.3 BERT block-size curriculum

**Pre-training a BERT with Curriculum Learning by Gradually Increasing the Block-size of Input Text** increases the block size of input text during BERT pretraining, primarily to improve memory utilization and training efficiency. It reports faster convergence and improved downstream performance in low-resource settings. [web:119]

This should not be conflated with DreamReasoner:

- BERT’s curriculum changes the **attention/input sequence block size**;
- DreamReasoner changes the **diffusion generation and corruption granularity**;
- the BERT work does not demonstrate improved mathematical or long-CoT reasoning.

### 3.4 CurrMask: relevant masking curriculum, but not text MLM

**Learning Versatile Skills with Curriculum Masking** uses an adaptive EXP3 bandit over masking ratios and block sizes. It improves skill prompting and planning in offline reinforcement learning, but its data are state-action trajectories rather than natural-language text, so it is not conventional MLM pretraining. [web:70]

It is still conceptually relevant because it directly treats masking schemes \((r,b)\) as curriculum arms and learns which corruption granularity produces the greatest learning progress. The work reports that the selected block sizes tend to increase over training, consistent with a transition from local to longer-range dependencies. [web:70][web:75]

## 4. Evidence hierarchy

| Claim | Best supporting evidence | Strength |
|---|---|---|
| Block diffusion interpolates between AR and full diffusion | BD3-LM | Strong architectural evidence; no reasoning evaluation. |
| Semi-autoregressive diffusion supports flexible length and bidirectional within-block updates | SSD-LM and BD3-LM | Strong architectural evidence. |
| Direct large-block training can damage reasoning | DreamReasoner | Strong within-paper ablation on AIME and related tasks. |
| Fine-to-coarse block-size curriculum improves reasoning robustness | DreamReasoner | Strongest direct evidence currently located. |
| Progressive block scaling improves diffusion reasoning stability | T⋆ | Independent supporting evidence, using TraceRL and SDAR. |
| Increasing masking difficulty during diffusion SFT can improve math/logical performance | DSFT | Positive complete-recipe results, but weak causal evidence for the curriculum component alone. |
| Time-varying corruption improves MLM transfer | MRD/PTW | Good evidence on GLUE, SQuAD, and CoNLL. |
| Concept-structured masking curricula improve NLU | CCM | Good evidence on GLUE and efficiency; not direct CoT reasoning evidence. |
| MLM masking curricula improve mathematical or chain-of-thought reasoning | — | **No strong conventional-MLM pretraining evidence identified.** |

## 5. Synthesis and open questions

The papers suggest a coherent hypothesis:

> **Reasoning benefits from initially fine-grained corruption because it gives the model repeated exposure to local dependencies and incremental commitment; coarse corruption can then be introduced to acquire parallelism and efficiency.**

For diffusion LMs, the evidence is strongest when “granularity” means **block size**:

\[
\text{token-level / small blocks}
\;\longrightarrow\;
\text{larger blocks}
\;\longrightarrow\;
\text{mixed-granularity inference}.
\]

This differs from ordinary noise-schedule annealing. A noise schedule changes the probability or amount of token corruption at a diffusion timestep; DreamReasoner and T⋆ change the **number of positions jointly denoised or committed**. DSFT changes the mask ratio and mask content, but its strongest improvements also come from number-aware masking and loss weighting.

The main unresolved questions are:

1. **Causal isolation:** Does block-size curriculum help because of curriculum ordering, because of exposure to multiple block sizes, or simply because small-block training is a better optimization regime?
2. **Corruption versus decoding granularity:** Would the same gains arise by keeping the training block size fixed while varying only mask ratio, span length, or remasking order?
3. **Schedule direction:** DreamReasoner and T⋆ support fine-to-coarse progression, whereas some MLM results support high-to-low mask-ratio decay. These are not contradictory: block size and mask ratio control different aspects of difficulty.
4. **Reasoning domains:** Current evidence is concentrated on math, code, logical QA, and a small number of general reasoning benchmarks. Tool use, agents, long-horizon planning, and scientific reasoning remain underexplored.
5. **Adaptive granularity:** Fixed blocks ignore semantic boundaries. A promising next step is a curriculum from token-level denoising to spans, clauses, subproblems, and eventually adaptive semantic blocks.

**Overall assessment:** BD3-LM and SSD-LM provide the architectural foundation; DreamReasoner supplies the clearest demonstration that a fine-to-coarse block-size curriculum improves diffusion-LM reasoning; T⋆ independently reinforces that conclusion. MRD, PTW, and CCM show that curriculum corruption can improve conventional MLM transfer and inference tasks, but they should be cited as **indirect precedents**, not as direct demonstrations of improved chain-of-thought reasoning.