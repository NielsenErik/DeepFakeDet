# Where to push the architecture — literature review against our measured weaknesses

*2026-08-04. Written after the four-arm experiment. Every "we measured" below
refers to numbers in `POC.md` / `STATUS.md`. The point of this document is to
turn our weaknesses into a ranked, concrete architecture plan.*

---

## 0. Our position, stated bluntly

**Strengths (measured).**
- Exact inference at scale: 44.5× over the reference implementation, d = 6080
  on one 16 GB card, log Z = −1.1e-06 after training, equivalence-tested.
- Exact arbitrary conditionals at 114k queries/s — *no competitor can express
  this query*.
- Structure learning pays in likelihood: Forman −169 nats, ORC −46, Chow-Liu −5
  vs random, reproduced on three representations.
- The likelihood-ratio construction works: 0.953 on the shift it was trained on.

**Weaknesses (measured).**
| # | weakness | evidence |
|---|---|---|
| W1 | no detection advantage over a GMM | PC-ratio 0.828 vs GMM-ratio 0.830 |
| W2 | localization loses to a memory bank | PC 0.568 vs PatchCore 0.670 patch-AUC |
| W3 | likelihood ≠ discrimination | dilution +0.093, inversion 55.4% |
| W4 | pseudo-fakes don't match real forgeries | graphics methods inverted (0.36–0.45) |
| W5 | capacity plateaus | more epochs/structure ⇒ better NLL, flat AUC |
| W6 | leaves are weak | univariate Gaussian mixtures over PCA'd features |

The literature has a direct, named answer for W1, W3, W5 and W6, a partial one
for W2, and a *warning* about W4 that we must take seriously.

---

## 1. W1 + W3 — the fix is discriminative circuits, and it is well-trodden

We trained by joint NLL and then hoped the density would discriminate. The PC
literature abandoned that hope in 2012.

- **[Discriminative Learning of Sum-Product Networks](https://papers.nips.cc/paper/2012/hash/573f7f25b7b1eb79a4ec6ba896debefd-Abstract.html)**
  (Gens & Domingos, NeurIPS 2012). Optimises *conditional* log-likelihood with
  a backprop-style gradient. Crucially: **the class of tractable discriminative
  SPNs is broader than the tractable generative ones** — you lose nothing
  structurally by going discriminative.
- **[Learning Logistic Circuits](https://arxiv.org/pdf/1902.10798)** (Liang &
  Van den Broeck, AAAI 2019). The discriminative counterpart of a PC; parameter
  learning reduces to *convex* logistic regression.
- **[Discriminative Bias for Learning PSDDs](https://web.cs.ucla.edu/~guyvdb/slides/IDA20.pdf)**
  (D-LearnPSDD). Encodes the class↔feature relation directly in the structure.
- **[Conditional SPNs](https://www.sciencedirect.com/science/article/pii/S0888613X21001766)**
  (Shao et al.). Neural **gate functions** make the circuit's weights a
  function of the input: `p(Y | X)` with a PC over Y and a DNN producing its
  parameters. They note that **sum-product-quotient networks** express
  `P(Y|X) = P(Y,X)/P(X)` as a ratio of two SPNs — i.e. *our two-circuit ratio
  is a known construction*, and CSPNs represent it more compactly.
- **Hybrid objectives**: trade off generative likelihood against conditional
  likelihood or a margin term
  ([Ng & Jordan-style analysis](http://ai.stanford.edu/~ang/papers/nips03-hybrid.pdf),
  [maximum-margin SPNs](https://arxiv.org/pdf/2303.09065)).

**Why this should beat the GMM tie.** The ratio fixed *scoring* but both
circuits were still fitted to model everything, so capacity is still spent on
the 6000 nuisance dimensions. A discriminative objective spends capacity only
where the two classes differ — the thing the ratio has to achieve by
cancellation, learned directly instead. A GMM trained the same way has far less
capacity to exploit it.

**Concrete plan.** Keep the two compatible circuits; replace the two separate
NLL fits with one objective

```
L = −λ · [ log p_real(x_real) + log p_blend(x_blend) ]        (generative anchor)
    − (1−λ) · log σ( log p_blend(x) − log p_real(x) ) · y…     (discriminative)
```

λ = 1 recovers today's model, λ = 0 is a pure discriminative circuit; sweep it.
Both circuits stay exactly normalised, so every exact query survives.

---

## 2. W5 + W6 — capacity: the plateau is a known, solved problem

- **[Scaling Up PCs by Latent Variable Distillation](https://arxiv.org/abs/2210.04398)**
  (Liu et al., ICLR 2023). Diagnoses exactly our symptom: *"as the number of
  parameters increases, performance immediately plateaus"* — the hierarchical
  latent space is too hard to optimise. Fix: take embeddings from a deep model,
  k-means them, and use the cluster ids as **supervision for the circuit's
  latent variables**, then finetune. Result: a 25M-parameter PC beats 400M
  without it.
  **We are unusually well placed**: we already have the SBI encoder to distill
  from, and our region graph is patch-hierarchical, which is what they use.
- **[Probabilistic Flow Circuits](https://ml-research.github.io/papers/sidheekh2023uai.pdf)**
  (Sidheekh, Kersting, Natarajan, UAI 2023). Replace univariate leaves with
  **normalizing flows**, under a condition they call **τ-decomposability** (a
  transform over a product node must transform each child scope independently).
  Tractability is preserved. This directly targets W6: our leaves are
  univariate Gaussian mixtures over whitened features, which is the weakest
  part of the model.
- **[Sum of Squares Circuits](https://arxiv.org/abs/2408.11778)** (Loconte,
  Mengel, Vergari, AAAI 2025). SOS circuits are exponentially more expressive
  than both monotone and squared PCs. **Important caveat from
  [Wang & Van den Broeck (AAAI 2025)](https://starai.cs.ucla.edu/papers/WangAAAI25.pdf):
  squared PCs can also be *less* expressive than monotone ones** — so SOS is
  not a free win and must be measured, not assumed. (Our `scripts/sos_experiment.py`
  does exactly this.)
- **[Monarch matrices](https://arxiv.org/pdf/2506.12383)** (2025) and
  **[PyJuice](https://arxiv.org/abs/2406.00766)** for raw scale, if we ever need
  more than d ≈ 10⁴.

---

## 3. W2 — localization: our unique query, currently losing

Two things from the literature reframe this.

- **[May 2026 deepfake review](https://www.insightface.ai/blog/may-2026-deepfake-detection-papers)**:
  frozen foundation features are discriminative for *full-face synthesis* but
  hit *"fundamental limits"* on **localized face edits**. That is our hardest
  case too (NeuralTextures, the smallest edit, is worst in every arm). So the
  representation, not the readout, may be the binding constraint on
  localization.
- **[Neural Probabilistic Circuits](https://arxiv.org/pdf/2603.01372)** (2025/26):
  a **neural attribute predictor** feeding a PC that reasons exactly over
  *attributes* and the class. This is the shape our explainability claim wants:
  the neural net supplies forensic attributes (blending-boundary strength,
  spectral-peak presence, noise-floor consistency) per patch, and the circuit
  does exact, auditable reasoning over them. Accuracy comes from the front-end;
  exactness and explanation come from the circuit — and the explanation is over
  *human-meaningful attributes*, not raw PCA coordinates.

Localization SotA to be aware of: the IJCAI-2025 detection+localization
challenge leader reports AUC 0.963 / F1 0.756 / IoU 0.819, though cross-domain
localization collapses (some methods to IoU ≈ 11%). Our derived-mask numbers
are not comparable to those, and we should say so.

---

## 4. W4 — a warning we must not ignore

**[The Alpha Blending Hypothesis: Compositing Shortcut in Deepfake Detection](https://arxiv.org/pdf/2605.10334)**
(2026) argues that detectors trained on self-blended images learn to recognise
**alpha-blending artifacts** rather than genuine forgery indicators — high
performance on blended data, poor transfer to forgeries made by other
compositing routes. They recommend training on diverse *real* forgeries.

This is precisely our measurement seen from the other side: our `p_blend`
scored 0.953 against its own blends, 0.83 against neural swaps, and **inverted**
on the two graphics-rendered manipulations. We independently reproduced their
hypothesis. Consequences:

1. Any headline resting on self-blends alone will be attacked on exactly this
   point. We should pre-empt it with the per-method table.
2. A *mixture* `p_blend` over several pseudo-fake families (blend-type,
   render-type, resample-type) is the natural PC answer — a circuit is a
   mixture model, so it can hold several forgery processes natively, and the
   mixture weights are themselves an interpretable output ("this forgery looks
   like a graphics render, not a neural swap").
3. A supervised variant, where `p_blend` is fitted on *real* forgeries, is a
   legitimate second setting to report alongside the real-only one — it upper
   bounds what the construction can do.

---

## 5. The competitive bar

Numbers we would be measured against (FF++ → Celeb-DF-v2 cross-dataset,
video-level AUC unless noted):

| method | CDF2 | in-dataset |
|---|---|---|
| [PhaseForensics](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1670833/full) | **91.2** | — |
| SBI (CVPR'22) | ~93 | ~99 |
| [Anti-Deepfake Transformer](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1670833/full) | 84.97 | 96.30 (FF++ HQ) |
| [LNCLIP-DF](https://arxiv.org/abs/2508.06248) — tune only LayerNorm (0.03% of params) + L2-normalised hyperspherical manifold + metric learning | strong across 14 benchmarks (2019–2025) | — |
| **ours today (SBI features, ratio)** | not measured | **0.828** |

Sobering: a *frozen CLIP + linear probe* is a respectable baseline in this
field, and our best in-dataset number is below every published cross-dataset
number. Competing on raw accuracy means adopting their front-ends, not
inventing a new density model.

---

## 6. The plan, ranked by (impact × feasibility)

### P0 — cheap, decisive, do first (< 1 day)
1. **Per-patch likelihood ratio for localization.** Inherits nuisance
   cancellation *and* exact conditioning. The one place we can beat PatchCore.
2. **Occlusion robustness.** Black out k% of patches; exact marginalization vs
   the baselines' imputation. Circuit-unique, trivially measurable, and a real
   deployment argument.
3. **Region-wise KL between p_real and p_blend.** The two circuits share a
   region graph, hence are *compatible*, so information-theoretic divergences
   between them are tractable
   ([Compositional Atlas](https://arxiv.org/abs/2102.06137), Vergari et al.
   NeurIPS 2021). Gives a **global** explanation — which face regions carry the
   discriminative information — that no baseline can produce at all.

### P1 — the accuracy fix (2–4 days)
4. **Hybrid generative/discriminative training of the circuit pair** (§1). This
   is the principled answer to "the GMM ties us", and it keeps exactness.
5. **Latent variable distillation from the SBI encoder** (§2). Directly targets
   the plateau; we already have the teacher.

### P2 — expressiveness (1 week)
6. **Flow leaves (τ-decomposable)** — replaces the weakest component.
7. **SOS circuits**, measured not assumed, given the Wang & Van den Broeck
   caveat. `scripts/sos_experiment.py` is written and ready.

### P3 — reframing (as needed)
8. **NPC-style architecture**: neural forensic-attribute predictor + PC over
   attributes + class. Best 2026-shaped story — *"the frontier of PC research is
   no longer about replacing neural networks but integrating with them"*
   ([ICLR 2026 blogpost](https://iclr-blogposts.github.io/2026/blog/2026/probabilistic-circuits-for-uncertainty-quantification/)).
9. **Mixture `p_blend` over forgery families** (§4), with the mixture posterior
   as an interpretable "which kind of forgery is this" output.

---

## 7. What the honest paper looks like, given all this

Not "PCs beat SotA deepfake detection" — that race is being won by
frozen-foundation-model probes with clever fine-tuning, and we would be
adopting their front-end anyway.

The defensible contribution is **tractable forensic reasoning**:

> one exactly-normalised model that detects, localizes with calibrated
> probabilities, conditions on context, survives missing regions by exact
> marginalization, and explains *at the model level* which regions and feature
> groups separate real from forged — with a measured diagnosis (dilution and
> inversion) of why naive likelihood fails, and a likelihood-ratio construction
> that repairs it for *every* density model, not only ours.

The P0 experiments decide whether the middle clause survives contact with
PatchCore. That is the next thing to run.

---

## Sources

- [Discriminative Learning of Sum-Product Networks](https://papers.nips.cc/paper/2012/hash/573f7f25b7b1eb79a4ec6ba896debefd-Abstract.html) — Gens & Domingos, NeurIPS 2012
- [Learning Logistic Circuits](https://arxiv.org/pdf/1902.10798) — Liang & Van den Broeck
- [Discriminative Bias for Learning PSDDs](https://web.cs.ucla.edu/~guyvdb/slides/IDA20.pdf)
- [Conditional SPNs: modular probabilistic circuits via gate functions](https://www.sciencedirect.com/science/article/pii/S0888613X21001766)
- [Classification with Hybrid Generative/Discriminative Models](http://ai.stanford.edu/~ang/papers/nips03-hybrid.pdf)
- [Maximum margin learning of t-SPNs](https://arxiv.org/pdf/2303.09065)
- [Scaling Up Probabilistic Circuits by Latent Variable Distillation](https://arxiv.org/abs/2210.04398) — ICLR 2023
- [Probabilistic Flow Circuits](https://ml-research.github.io/papers/sidheekh2023uai.pdf) — UAI 2023
- [Sum of Squares Circuits](https://arxiv.org/abs/2408.11778) — AAAI 2025
- [On the Relationship Between Monotone and Squared PCs](https://starai.cs.ucla.edu/papers/WangAAAI25.pdf) — AAAI 2025
- [Scaling PCs via Monarch Matrices](https://arxiv.org/pdf/2506.12383) — 2025
- [Scaling Tractable PCs: A Systems Perspective (PyJuice)](https://arxiv.org/abs/2406.00766)
- [A Compositional Atlas of Tractable Circuit Operations](https://arxiv.org/abs/2102.06137) — NeurIPS 2021
- [Causal Neural Probabilistic Circuits](https://arxiv.org/pdf/2603.01372)
- [Probabilistic Circuits for Uncertainty Quantification](https://iclr-blogposts.github.io/2026/blog/2026/probabilistic-circuits-for-uncertainty-quantification/) — ICLR 2026 blogpost
- [The Alpha Blending Hypothesis](https://arxiv.org/pdf/2605.10334) — 2026
- [Deepfake Detection that Generalizes Across Benchmarks (LNCLIP-DF)](https://arxiv.org/abs/2508.06248)
- [Decoding deception: state-of-the-art deepfake detection](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1670833/full)
- [May 2026 deepfake detection papers](https://www.insightface.ai/blog/may-2026-deepfake-detection-papers)
- [Anomaly Detection with Generative Models and SPNs in Mammography](https://arxiv.org/pdf/2210.06188)
