# CW-MPVR: Confidence-Weighted Multi-Path Visual Reasoning

Does a vision-language model answer multi-step visual questions better if it reasons several times and weights each attempt by **two separate confidence signals**: *is the evidence grounded in the image?* and *how sure is the model of the answer?*

Zero-shot **BLIP-2** on **GQA**, evaluated against greedy decoding, self-consistency, answer-confidence weighting and **LXMERT**.

> COMP6001 Computer Vision and Multimodal Machine Learning · Final Project · Semester 2, 2026

---

## Key findings

**1. The two confidence signals do different jobs (a double dissociation).**

| Job | Answer confidence | Grounding confidence |
|---|---|---|
| *Which path is right?* (within-question AUC, testdev) | 0.51: chance | **0.54**: useful |
| *Should the final answer be trusted?* (question-level AUROC) | **0.70**: useful | 0.48: chance |

**2. Decomposed confidence improves *knowing when to trust* an answer.** Combining path agreement with answer confidence raises AUROC from 0.681 to 0.727 (all signals vs agreement: +0.046, 95% CI 0.033–0.059, p < 0.001; grounding adds nothing on top). Answering only the most confident half gives 58.1% vs 54.2%.

**3. It does not significantly improve accuracy on new images.** Cross-validated CW-MPVR reaches 41.0% vs 40.7% for majority vote (+0.3 pts, n.s. after Holm correction), although a correct answer is present in some path for 59.8% of questions.

**4. A VLM's grounding confidence is inflated on its own pretraining images.** Validation images are Visual Genome (seen in BLIP-2 pretraining); testdev images are not. Grounding separates right from wrong paths far better on seen images (AUC 0.603 vs 0.544), so weights tuned on validation *lowered* testdev accuracy (39.6%).

**5. ITM checks what is in the image, not whether an answer is correct.** Grounding scores rise when evidence names objects that exist in the scene graph (ρ = 0.22, p < 0.001), but scoring the question–answer pair directly gives no signal (AUC 0.50).

**6. Gains do not grow with reasoning depth** (−0.36 pts per extra step, p = 0.32, controlling for question type).

---

## Results

![Final accuracy](figures/05_final_accuracy.png)

**Accuracy on 2,500 GQA testdev-balanced questions** (95% CI: bootstrap resampling images)

| Method | Training on GQA | Accuracy (%) | 95% CI |
|---|---|---|---|
| Greedy (BLIP-2) | none | 40.20 | 38.16 – 42.23 |
| Majority vote (self-consistency) | none | 40.68 | 38.62 – 42.74 |
| Answer-only weighting | none | 41.12 | 39.01 – 43.22 |
| CW-MPVR, tuned on val (pre-specified) | none | 39.60 | 37.51 – 41.66 |
| Answer-only (CV) | none | 41.04 | 38.92 – 43.13 |
| Grounding-only (CV) | none | 40.64 | 38.62 – 42.69 |
| **CW-MPVR (CV)** | none | **40.96** | 38.89 – 42.99 |
| LXMERT | fine-tuned | 59.20 | 56.99 – 61.35 |
| *Oracle: a correct path exists* | – | *59.80* | – |

*CV = 5-fold cross-validation on testdev, folds split by image (`data/cv_folds.csv`). LXMERT reproduces its published 60.0% using the authors' original features.*

**When to trust an answer** (CW-MPVR answers, out-of-fold)

| Confidence signal | AUROC ↑ | AURC ↓ | Accuracy when answering 50% |
|---|---|---|---|
| Agreement between paths | 0.681 | 0.481 | 54.2 |
| Answer confidence | 0.704 | 0.433 | 55.4 |
| Grounding confidence | 0.479 | 0.625 | 39.4 |
| Agreement + answer | **0.727** | **0.422** | **58.2** |
| All three | 0.727 | 0.423 | 58.1 |

![Calibration and abstention](figures/05_calibration.png)

---

## Method

For each question, BLIP-2 FlanT5-XL samples **k = 10 paths**: an evidence sentence (two prompts × five samples, T = 1.0, top-p 0.95), then an answer decoded from that evidence. Each path *i* gets:

- **answer confidence** $a_i$: mean log-probability of the answer tokens
- **grounding confidence** $g_i$: log P(match) of the evidence from BLIP-2's image–text matching head

$$w_i = \text{softmax}_i(\beta a_i + \gamma g_i), \qquad \hat{y} = \arg\max_y \sum_{i:\, y_i = y} w_i$$

One rule covers every variant: β = γ = 0 is majority vote, γ = 0 is answer-only, β = 0 is grounding-only, β = γ is a single combined confidence. β and γ are chosen by grid search on held-out data only. Aggregation is implemented from scratch (`src/aggregate.py`, `src/cwmpvr.py`).

```
GQA question + image
      │
      ├── BLIP-2 greedy ────────────────────────────► Greedy baseline
      │
      └── BLIP-2 × 10 paths: evidence → answer
               │            │
         answer conf.   ITM grounding
               └──── weighted vote ────────────────► CW-MPVR
                     (generated once, cached; every method reads the same cache)
```

## Dataset

**GQA** (Hudson & Manning, 2019): real images with compositional questions generated from scene graphs.

| Split | Use | Questions | Images |
|---|---|---|---|
| testdev-balanced | evaluation (stratified by structural × semantic type) | 2,500 of 12,578 | 392 |
| val-balanced | calibration (pre-specified protocol), scene-graph analysis | 500 of 132,062 | 478 |

Samples are proportional stratified draws (fixed seed), checked for representativeness (χ² p = 0.94 and 0.92). Questions share images (~6 per image), so all confidence intervals resample **images**.

## Benchmarks

| Method | Why |
|---|---|
| Greedy | Standard single-answer BLIP-2 inference: the floor |
| Majority vote (Wang et al., 2023) | Same paths, unweighted: isolates the value of weighting |
| Answer-only weighting (Wang et al., 2023) | Same paths, answer confidence only: isolates the value of grounding |
| LXMERT (Tan & Bansal, 2019) | Established GQA model, fine-tuned on GQA: supervised reference point |

## Repository structure

```
cw-mpvr/
├── notebooks/
│   ├── 00_setup.ipynb              Colab + Drive + GitHub setup
│   ├── 01_data_eda.ipynb           GQA loading, EDA, stratified samples, images
│   ├── 02_generate_paths.ipynb     BLIP-2 paths + both confidence signals (GPU)
│   ├── 03_baselines.ipynb          Greedy, majority vote, answer-only, LXMERT
│   ├── 04_cwmpvr.ipynb             CW-MPVR, tuning, ablations, k-sweep, complexity, diagnosis
│   ├── 04b_answer_grounding.ipynb  Pre-declared follow-up: question–answer grounding
│   └── 05_analysis.ipynb           Significance, depth, scene graphs, calibration, failures
├── src/
│   ├── gqa.py         loading, stratified sampling, scene-graph statistics
│   ├── paths.py       BLIP-2 generation, log-probabilities, ITM/ITC scoring, cache
│   ├── aggregate.py   weighted voting (from scratch)
│   ├── cwmpvr.py      CW-MPVR scoring, grid search, confidence features
│   ├── evaluate.py    accuracy, image-clustered bootstrap, McNemar
│   ├── errorlog.py    log_error() → ERROR_LOG.md
│   ├── gitsync.py     commit/push from Colab
│   └── utils.py       config, seeding
├── configs/default.yaml   all settings (sample sizes, k, temperature, models, seed)
├── data/                  sample ID lists and CV folds
├── results/               cached paths (.jsonl.gz), tables (.csv), tuned weights
├── figures/               all report figures
└── ERROR_LOG.md           every development error, its cause and its fix
```

## How to reproduce

Run in Google Colab, in order. Every notebook starts with the same setup cell and ends by pushing to GitHub.

| Notebook | Runtime | Time |
|---|---|---|
| 00, 01 | CPU | ~10 min (image download) |
| 02 | **L4 GPU** | ~80 min for 3,000 questions |
| 03, 04b | T4 GPU | ~5–10 min each |
| 04, 05 | CPU | ~1–2 min each |

**Without a GPU**: notebooks 03 (except LXMERT), 04 and 05 run from the cached paths in `results/paths_*.jsonl.gz`.

**Setup**: add a GitHub token with *Contents: read and write* as a Colab secret named `GITHUB_TOKEN`, open `notebooks/00_setup.ipynb`, set `GITHUB_USER`, run all. Raw GQA files and images go to `MyDrive/cw-mpvr-data/` (not in git).

## Complexity

- **Generation**: O(N · k · (L_e + L_a)) decoder steps plus O(N · k) ITM passes; ≈1.6 s per question on an L4; ≈11 GB GPU memory, constant in N. Cost is linear in k.
- **Aggregation**: O(N · k) time and memory; 28 µs per question, 4 MB for 25,000 paths, no GPU.

## Limitations

- Evidence sentences are short (~3 words), so grounding mostly checks object presence, not relations.
- Calibration and evaluation images come from different sources; the pre-specified tuning protocol does not transfer, so cross-validation on testdev is reported alongside it.
- The scene-graph check (H4) uses validation images, which BLIP-2 saw during pretraining.
- One backbone (BLIP-2 FlanT5-XL), one dataset and a 2,500-question sample; gains below ~1 point cannot be resolved.
- Exact-match accuracy after light normalisation; synonyms count as wrong.

## Error log

Full details in [`ERROR_LOG.md`](ERROR_LOG.md).

| # | Problem | Fix |
|---|---|---|
| E01 | `autoreload` fails on Colab's Python 3.13 (`imp` removed) | Removed; restart after editing `src/` |
| E02 | Paths nearly identical (mode collapse) | Sampling T 0.7 → 1.0, top-p 0.9 → 0.95 (tuned on val) |
| E03 | LXMERT demo detector deleted from Hugging Face | Authors' original GQA testdev features |
| E04 | Val-tuned weights hurt testdev accuracy | Diagnosed image-source shift; added image-grouped CV |
| E05 | Same CV gave slightly different results across notebooks | Fixed folds saved to `data/cv_folds.csv` |

## Deviations from the proposal

- Evaluation on **testdev** instead of val: LXMERT was fine-tuned on val; val is used for calibration only.
- LXMERT is **run** on the same questions rather than quoted.
- Added **answer-only weighting** to isolate the grounding signal.
- "Multiple samplings" implemented as image-clustered bootstrap with Holm-corrected paired tests.
- Added cross-validated evaluation after the pre-specified protocol failed (both reported).

## References

- Hudson, D. A., & Manning, C. D. (2019). GQA: A new dataset for real-world visual reasoning and compositional question answering. *CVPR*.
- Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML*.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR*.
- Tan, H., & Bansal, M. (2019). LXMERT: Learning cross-modality encoder representations from transformers. *EMNLP*.
- Anderson, P., et al. (2018). Bottom-up and top-down attention for image captioning and visual question answering. *CVPR*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*.
