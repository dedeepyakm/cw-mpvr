# CW-MPVR: Confidence-Weighted Multi-Path Visual Reasoning

Compositional visual question answering on **GQA** with **BLIP-2**, where several reasoning paths are aggregated by two separate confidence signals: **visual-grounding confidence** and **answer confidence**.

> COMP6001 Computer Vision and Multimodal Machine Learning, Final Project, Semester 2 2026

---

## Idea

- Single greedy decoding is fragile on multi-step questions: one wrong step breaks the answer.
- Sampling several paths and voting helps, but every path counts equally.
- CW-MPVR weights each path by:
  - **Grounding confidence**: does the path's evidence match the image? (BLIP-2 image-text matching head)
  - **Answer confidence**: how probable is the answer? (length-normalised token log-probability)
- Hypothesis: decomposed confidence beats both greedy decoding and unweighted voting, and the gain grows with reasoning depth.

## Methods compared

| Method | Description |
|---|---|
| Greedy | BLIP-2, single path |
| Self-consistency | Same *k* paths, unweighted majority vote |
| Answer-only weighting | Paths weighted by answer confidence only |
| LXMERT | GQA fine-tuned external baseline |
| **CW-MPVR (ours)** | Paths weighted by grounding + answer confidence |

## Repository structure

```
cw-mpvr/
├── notebooks/        # Colab notebooks, run in order
├── src/              # reusable code imported by the notebooks
├── configs/          # experiment settings
├── data/             # subsample ID lists only (raw data lives on Google Drive)
├── results/          # metrics tables and cached outputs (small files only)
├── figures/          # figures used in the report
├── ERROR_LOG.md      # every error met during development, with cause and fix
└── requirements.txt
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 00 | `00_setup.ipynb` | Mount Drive, clone repo, install, GPU check |
| 01 | `01_data_eda.ipynb` | Download GQA, stratified subsample, dataset analysis |
| 02 | `02_generate_paths.ipynb` | Sample *k* paths with BLIP-2, score and cache them |
| 03 | `03_baselines.ipynb` | Greedy, self-consistency, LXMERT |
| 04 | `04_aggregation.ipynb` | CW-MPVR aggregation, tuning, ablations |
| 05 | `05_analysis.ipynb` | Results, calibration, statistical tests, error analysis |

## How to run (Google Colab)

**One-time setup**

1. Create a GitHub fine-grained token with *Contents: read and write* on this repo.
2. In Colab, open the 🔑 **Secrets** panel and add it as `GITHUB_TOKEN`.
3. Open `notebooks/00_setup.ipynb` from GitHub in Colab (*File → Open notebook → GitHub*), set `GITHUB_USER`, and run it. This clones the repo into `MyDrive/cw-mpvr/`.

**Every session after that**

- Open notebooks from `MyDrive/cw-mpvr/notebooks/` (not from GitHub), so edits are saved in the repo folder.
- Run the standard setup cell at the top of the notebook.
- Finish with `push("what changed")` to commit and push.

**Where things live**

| Location | Contents | In git? |
|---|---|---|
| `MyDrive/cw-mpvr/` | code, notebooks, configs, small results | yes |
| `MyDrive/cw-mpvr-data/` | GQA questions, images, cached model outputs | no |

## Reproducing results

```bash
pip install -r requirements.txt
```

Then run `notebooks/01` to `05` in order. All settings (sample size, *k*, temperature, seed) are in `configs/default.yaml`.

## Results

_To be added._

## Error log

All development errors, their causes and their fixes are recorded in [`ERROR_LOG.md`](ERROR_LOG.md).

## References

- Hudson & Manning (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. CVPR.
- Li et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.
- Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR.
- Tan & Bansal (2019). LXMERT: Learning Cross-Modality Encoder Representations from Transformers. EMNLP.
