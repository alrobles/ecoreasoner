## Scope and main conclusion

The literature separates into **three related but non-equivalent ideas**:

1. **Replaced-token detection (RTD):** discriminate whether an observed token is original or generated as a replacement. This is the ELECTRA family.
2. **Contrastive/negative training:** assign higher likelihood or representation similarity to a preferred sequence/token than to a plausible but incorrect or undesirable alternative.
3. **Error detection for masked diffusion LMs:** add explicit supervision for *visible* tokens that are wrong, inconsistent, or model-generated, enabling targeted remasking. This is newer and is not induced reliably by the standard masked-diffusion denoising objective.

A useful conceptual distinction is:

\[
\text{RTD: } y_i=\mathbf{1}[\tilde{x}_i\ne x_i],
\]

where the label records the **source of corruption**, whereas diffusion error-detection objectives attempt to predict something closer to

\[
y_i=\mathbf{1}[\tilde{x}_i\text{ is incorrect or inconsistent with }x_{\setminus i}],
\]

which is a semantic or task-dependent correctness label. A generator-produced token can be plausible and correct, while a token that was not synthetically replaced can still be factually or grammatically wrong.

---

## 1. ELECTRA-style replaced-token detection

### Canonical ELECTRA

**ELECTRA** replaces selected tokens with samples from a small MLM generator and trains a discriminator to classify every input token as **original** or **replaced**, rather than reconstructing only the masked positions. The generator is trained with MLM; the discriminator receives the fully populated corrupted sequence and is trained with token-level binary cross-entropy. This provides a much denser learning signal than conventional MLM. [web:154]

The important limitation is that RTD is fundamentally a **provenance classifier**. It learns to detect âgenerator-likeâ substitutions, not necessarily all tokens that are linguistically or factually inconsistent. If the generator samples the correct token, the token is generally treated as original; if it samples a plausible but valid alternative, the discriminator is still trained to call it replaced.

### Important extensions

| Paper | Year | Discriminative objective | Relevance |
|---|---:|---|---|
| **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators** | 2020 | Binary classification of original versus generator-replaced tokens | Foundational RTD formulation; generator uses MLM and the main encoder is the discriminator. [web:154] |
| **Learning to Sample Replacements for ELECTRA Pre-Training** | 2021 | Makes generator replacements harder using hardness prediction and sampling improvements | Directly addresses the quality of negatives. The generator is encouraged to produce replacements that expose what the discriminator has not yet learned; focal loss reduces the problem of sampling the correct token as a ânegative.â [web:163] |
| **COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining** | 2021 | Token-level corrective LM plus sequence-level contrastive learning | Goes beyond binary RTD: the main model must detect **and correct** auxiliary-model replacements, while also contrasting corrupted and source-derived sequences. [web:151] |
| **DeBERTa-v3: Improving DeBERTa using ELECTRA-Style Pre-training with Gradient-Disentangled Embedding Sharing** | 2021/2023 publication lineage | RTD in place of the original DeBERTa MLM objective | Adopts RTD and adds gradient-disentangled generator/discriminator embedding sharing; the discriminator remains the principal pretrained encoder. [web:158] |
| **AMOS: Pretraining Text Encoders with Adversarial Mixture of Training Signal Generators** | 2022 | RTD against replacements from a mixture of auxiliary generators | Extends ELECTRAâs single-generator setup with multiple/adversarial signal generators, aiming to diversify and strengthen corruption patterns. [web:13] |
| **Multilingual/translation replaced-token detection** | 2022 | Detects replacements in multilingual or translation-related inputs | Applies RTD beyond monolingual corruption, including multilingual and translation-derived replacement signals. [web:5] |
| **FAST-ELECTRA for Efficient Pre-training** | 2024 | More efficient ELECTRA-style discriminator pretraining | Primarily an efficiency/implementation advance within the RTD family. [web:130] |

### What RTD contributes

RTD is attractive when the desired representation should be sensitive to **local contextual fit**:

- it trains on nearly every position rather than only masked positions;
- negatives are usually **plausible**, because they come from an LM rather than random vocabulary replacement;
- it provides a token-level discriminator that can be used independently of the generator.

However, its supervision is normally **synthetic and binary**. RTD does not by itself establish that the discriminator has learned factuality, entailment, global consistency, or the ability to identify a modelâs own inference errors.

---

## 2. Contrastive losses over plausible-but-wrong continuations

The closest general formulation is a pairwise sequence-ranking loss:

\[
\mathcal L_{\mathrm{rank}}
 =
 \max\left(0,\,
 \gamma-\log p_\theta(y^+\mid x)+\log p_\theta(y^-\mid x)
 \right),
\]

where \(y^+\) is preferred and \(y^-\) is a plausible but inferior continuation. Unlike RTD, this objective directly compares **semantic or preference quality**, not merely corruption provenance.

Most work in this category has been developed for **autoregressive sequence models**, rather than encoder-only MLMs or masked diffusion LMs.

| Paper | Year | Negative construction/objective | Relation to MLM/diffusion |
|---|---:|---|---|
| **Neural Text Generation with Unlikelihood Training** | 2020 | Explicitly lowers probability of undesirable token or sequence candidates while retaining likelihood training on gold text | Token/sequence negative training; not specifically an MLM or diffusion objective. [web:99] |
| **An Analysis of the Utility of Explicit Negative Examples to Improve the Syntactic Abilities of Neural Language Models** | 2020 | Adds a margin between correct and incorrect word log-likelihoods | A direct token-level discriminative analogue for syntax; uses explicit minimal or corrupted negatives. [web:95] |
| **Understanding by Understanding Not: Modeling Negation in Language Models** | 2021 | Adds unlikelihood training on automatically constructed negative/negated examples | Demonstrates task-specific negative supervision beyond ordinary likelihood training. [web:92] |
| **A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration** | 2022 | Contrastive token learning combines likelihood and unlikelihood: promote the target token over selected negative tokens | Especially relevant as a token-level alternative to pure cross-entropy; not restricted to MLM. [web:126] |
| **The CRINGE Loss: Learning What Language Not to Model** | 2023 | CRINGEâContRastive Iterative Negative GEnerationâuses generated negative tokens, often high-ranked model predictions, and contrasts them with positive targets | Explicitly trains on plausible model-generated wrong tokens. [web:119][web:93] |
| **CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning** | 2023 | Generates multiple continuations, labels them by quality, and applies a max-margin loss so preferred continuations receive higher sequence likelihood | Very close to âplausible-but-wrong continuationâ training, but designed for causal generation. [web:117] |
| **SLiC-HF: Sequence Likelihood Calibration with Human Feedback** | 2023 | Pairwise margin loss over preferred and rejected responses, plus reference-model/SFT regularization | A sequence-level discriminative objective for preference data; it explicitly lowers likelihood of negative responses. [web:108] |
| **PACT: Pretraining with Adversarial Contrastive Learning for Text Classification** | 2023 | Uses adversarial MLM perturbations to mine hard negative token and sequence representations, then applies contrastive learning | One of the clearest MLM-adjacent hard-negative approaches, although the end task is text classification rather than MLM scoring. [web:127] |
| **Trainable Hard Negative Examples in Contrastive Learning** | 2024 | Learns to generate or optimize difficult negative examples | General contrastive-learning method; relevant as a negative-mining mechanism rather than a masked-LM pretraining objective. [web:102] |
| **Validator-Guided Hard Negative Mining for Masked Language Modeling in Low-Resource Ancient Languages** | 2026 | Uses a validator to identify difficult incorrect MLM candidates and fine-tunes with hard negatives | Directly targets hard-negative training for an MLM, though it is a low-resource/task-specific fine-tuning study rather than a general replacement for MLM pretraining. [web:31] |

### Relationship to ELECTRA

The distinction can be summarized as follows:

| Property | ELECTRA RTD | Contrastive continuation training |
|---|---|---|
| Positive/negative unit | Individual token | Token, span, or complete continuation |
| Negative label | âWas generated as a replacement?â | âIs less correct/preferred/consistent?â |
| Negative source | Auxiliary MLM generator | Human rejection, model sampling, adversarial perturbation, validator, or unlikelihood candidate |
| Typical loss | Binary cross-entropy | Hinge/margin, InfoNCE, preference, or unlikelihood loss |
| Main signal | Contextual discrimination | Relative quality or semantic correctness |
| Typical architecture | Encoder discriminator plus small generator | Mostly autoregressive LM; some encoder/MLM variants |
| Main failure mode | Learns generator artifacts/provenance | False negatives, preference-label noise, and sequence-length/likelihood bias |

COCO-LM is a particularly useful bridge: it retains ELECTRA-style corruption but adds an explicit **correction** task and a **sequence contrastive** task. [web:151]

---

## 3. Hard negatives for MLMs and dLLMs

### MLMs

Hard negatives can be introduced at several levels:

1. **Generator-selected token replacements:** ELECTRAâs ordinary samples.
2. **Adaptive replacement sampling:** make the auxiliary generator seek replacements that are difficult for the discriminator. This is the explicit contribution of *Learning to Sample Replacements for ELECTRA*. [web:163]
3. **Adversarial representation perturbations:** PACT perturbs masked-token representations to reduce the probability of the correct token, producing nearby adversarial alternatives that serve as hard negatives. [web:127]
4. **Explicit wrong-token margin losses:** train the model to rank the gold token above a known incorrect token, as in the syntax-focused negative-example work. [web:95]
5. **Validator- or task-guided candidate mining:** generate plausible MLM candidates and retain those judged incorrect by a validator. The 2026 ancient-language work reports gains from validator-guided hard-negative fine-tuning. [web:31]

A caution is that **hardness is not correctness**. The most confusing candidate may be a legitimate alternative, a paraphrase, or a false negative. The best methods therefore need a validator, label source, or semantic constraint rather than simply choosing the modelâs highest-probability alternative.

### Diffusion LMs

For masked diffusion LMs, the standard objective is still masked-token reconstruction. A typical masked diffusion process replaces tokens by \([MASK]\) and trains the model to predict the clean token at masked positions; this objective can be viewed as a weighted collection of MLM-like losses. [web:50]

The central issue is that, after a token has been filled in, ordinary MDLM training does not necessarily teach the model to judge whether that **visible** token is wrong. Recent work addresses this with explicit correction or error-detection supervision.

| Paper | Year | What is detected | Training mechanism | Status/relevance |
|---|---:|---|---|---|
| **Corrective Diffusion Language Models** | 2025 | Visible tokens corrupted by uniform random substitutions or otherwise incorrect tokens | Mixture corruption combines absorbing-mask noise with visible-token corruption; the model is explicitly supervised to assign lower confidence to visible incorrect tokens while retaining masked reconstruction | The clearest direct answer to the question: a masked diffusion LM is trained to detect incorrect visible tokens, not only denoise masks. [web:166][web:177] |
| **Self-Reflective Remasking for Diffusion Language Models / RemeDi** | 2025/2026 | Incorrect already-generated tokens | Remask-aware SFT trains the model to identify/remask incorrect tokens as well as predict masks; later training optimizes generation/refinement behavior | Joint confidence prediction and self-remasking rather than ordinary denoising alone. [web:179] |
| **Guided Star-Shaped Masked Diffusion** | 2025/2026 | Errors made by the diffusion model during generation | A lightweight learned error detector is trained on model-generated errors and controls remasking; it is deliberately trained on targeted model errors rather than uniform random substitutions | Strong example of a separate discriminative error detector over a frozen or lightly tuned MDLM. [web:46][web:53] |
| **Teach Diffusion Language Models to Learn from Their Own Mistakes: Decoupled Self-Correction** | 2026 | Subtle errors produced by a high-quality DLM | Freeze the generation model and train a correction head on its generated errors; Future-Context Augmentation supplies less-corrupted/future context during detector training | Explicitly decouples generation from token-error detection. [web:168] |
| **Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts** | 2026 | Unsupported, stale, or inconsistent visible summary tokens | A linear token classifier, **[Mask]-Disc**, is trained on synthetic diffusion-style corruptions and source-document-conditioned errors; detected tokens are remasked and regenerated | Application-specific but directly implements token-level discriminative detection on top of an MDLM. [web:174] |
| **NAVIRA: Decoupled Stochastic Remasking for Masked Diffusion Language Models** | 2026 | Unreliable committed tokens | Inference-time quality scoring chooses tokens for stochastic remasking; the paper is primarily a decoding policy, not a new detector-training objective | Relevant to deployment of discriminative confidence but not, by itself, evidence of discriminative pretraining. [web:180] |
| **Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement** | 2026 | Suspected erroneous committed tokens | Uses probability-, trigger-, and temporal-difference-based detection; resets suspected tokens to \([MASK]\) | Mainly training-free detection/remasking, useful as an inference baseline rather than a learned discriminative LM objective. [web:52] |

### The strongest direct matches

The most direct precedents for âtrain a masked diffusion LM to detect corrupted/inconsistent tokensâ are:

- **Corrective Diffusion Language Models:** explicit supervision of visible incorrect tokens. [web:166]
- **Guided Star-Shaped Masked Diffusion:** a learned error detector trained on model-generated errors. [web:46]
- **Decoupled Self-Correction:** a separately trained correction head for the frozen diffusion generator. [web:168]
- **DetectâRemaskâRepair:** token-level discriminator for unsupported or stale tokens in diffusion-based summarization. [web:174]

These differ from ELECTRA in an important way: their positive/negative labels are intended to represent **correctness under a corruption or generation process**, not simply whether a token came from an auxiliary generator.

---

## 4. Pairwise and pseudo-likelihood evaluation of masked LMs

### Pseudo-log-likelihood

Because an MLM does not define a left-to-right sentence probability, the standard evaluation device is **pseudo-log-likelihood (PLL)**:

\[
\operatorname{PLL}(x)
=
\sum_{i=1}^{T}
\log p_\theta(x_i\mid x_{\setminus i}),
\]

where \(x_i\) is masked one position at a time and predicted from the remaining sequence. Salazar et al. formalized this as an out-of-the-box scoring method for BERT-like models and showed that PLL can be useful for ranking sentences and rescoring hypotheses. [web:136]

Common variants include:

- **token-normalized PLL**, useful when comparing sequences of different lengths;
- **word-level PLL**, which masks all subword pieces of a word together or aggregates them carefully;
- **masked pseudo-likelihood with multiple-token masks**, which trades off computational cost and conditional fidelity;
- **AUL/all-unmasked likelihood**, which scores all tokens from an unmasked input, but should not be conflated with standard PLL because the model is not evaluated under the same one-token-removed conditional. [web:64]

PLL is computationally expensive because it requires one or more forward passes per position. Recent work therefore proposes word-aware or more efficient variants; one such study reports that a word-level left-to-right adaptation improves on the original PLL baselines. [web:65]

### Pairwise evaluation

The usual pairwise evaluation rule is:

\[
\operatorname{Acc}_{\mathrm{pair}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\mathbf{1}\!\left[
\operatorname{PLL}(x_n^+)
>
\operatorname{PLL}(x_n^-)
\right].
\]

Here \(x^+\) and \(x^-\) are minimal pairs, such as grammatical versus ungrammatical sentences, preferred versus dispreferred substitutions, or stereotype versus anti-stereotype variants.

Representative uses include:

| Evaluation setting | Pairwise procedure | What it measures |
|---|---|---|
| **BLiMP and related minimal-pair syntax tests** | Compare PLL of grammatical and ungrammatical sentences | Whether the MLM ranks the grammatical member higher; the same scoring principle is used for minimal-pair syntactic evaluation. [web:72] |
| **Winograd Schema, WinoGrande, CommonsenseQA and forced-choice tasks** | Substitute competing candidates into the same context and compare PLL/normalized PLL | Which alternative the MLM considers more probable under bidirectional context. [web:150] |
| **CrowS-Pairs and StereoSet-style bias tests** | Compare PLLs of matched sentences differing in demographic or stereotype terms | Relative preference/bias, although PLL is sensitive to lexical frequency and other confounds. [web:79] |
| **Sentence likelihood and ASR/N-best rescoring** | Rank competing hypotheses by PLL, often with length normalization or interpolation | Whether the MLM supplies useful global sequence scores. [web:140] |
| **Pairwise pseudo-perplexity / pairwise PPL** | Evaluate conditional probabilities for paired or jointly masked tokens | Extends MLM-derived scoring to pairwise token dependencies and provides an alternative to ordinary unary PLL. [web:83] |

### Interpretation and limitations

Pairwise PLL evaluation is **discriminative evaluation**, but normally not discriminative **training**. It asks whether a pretrained MLM ranks \(x^+\) above \(x^-\); it does not update the model with a pairwise margin.

Important caveats are:

- PLL is a pseudolikelihood, not a normalized joint sentence likelihood.
- Raw PLL favors shorter sequences; token normalization can change rankings.
- Subword tokenization makes word-level comparisons nontrivial.
- Minimal-pair datasets can contain lexical-frequency artifacts.
- A model can rank a corrupted sentence lower without possessing an explicit token-level detector.
- PLL evaluates the whole sentence by repeatedly masking tokens; it is therefore different from asking a model to classify already-visible tokens as corrupted.

---

## 5. Overall assessment and research gap

The field has progressed through the following chain:

\[
\text{MLM reconstruction}
\rightarrow
\text{ELECTRA provenance detection}
\rightarrow
\text{adaptive/adversarial hard negatives}
\rightarrow
\text{sequence-level preference contrast}
\rightarrow
\text{diffusion error detection and remasking}.
\]

The best-established approach is still **ELECTRA-style RTD**, especially for efficient encoder pretraining. Its key weakness is that the discriminatorâs label is tied to the corruption mechanism. Adaptive replacement sampling and COCO-LM reduce this weakness by making replacements harder and adding correction/contrastive signals. [web:163][web:151]

For **plausible-but-wrong continuations**, the most mature objectives are sequence-level margin losses such as CLICK and SLiC-HF, and token-level negative objectives such as unlikelihood, CRINGE, and contrastive token learning. [web:117][web:108][web:99][web:119]

For **masked diffusion LMs**, explicit visible-token error supervision is a genuinely newer direction. CDLM, Guided Star-Shaped Masked Diffusion, DSC, and DetectâRemaskâRepair provide the clearest evidence that a diffusion LM can be trained or augmented to detect errors in committed tokens rather than merely predict tokens at masked positions. [web:166][web:46][web:168][web:174]

The most promising unified objective would combine:

\[
\mathcal L
=
\mathcal L_{\mathrm{mask\text{-}denoise}}
+
\lambda_{\mathrm{RTD}}\mathcal L_{\mathrm{source\text{-}detect}}
+
\lambda_{\mathrm{err}}\mathcal L_{\mathrm{visible\text{-}error}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{pairwise}},
\]

with negatives drawn from both auxiliary generators and the modelâs own diffusion trajectories. The crucial design requirement is to label a token as negative because it is **wrong under the task/context**, not merely because it was sampled by the generator.', 'type': 'output_text