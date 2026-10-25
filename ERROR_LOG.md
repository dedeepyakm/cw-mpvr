# Error Log

Every error met during development, in one place.
Entries are added with `log_error(...)` from `src/errorlog.py` (or by hand, same format).

**Entry format**

- **Where**: notebook and step
- **Error**: the message, shortened
- **Cause**: why it happened
- **Fix**: what solved it
- **Lesson**: what to do differently (optional)

---

<!-- entries below -->

### E01 (2026-10-25)
- **Where**: 00_setup / step 4 (standard setup cell)
- **Error**: `ModuleNotFoundError: No module named 'imp'`
- **Cause**: Colab moved to Python 3.13; its bundled IPython autoreload extension still imports `imp`, which was removed in Python 3.12
- **Fix**: Removed %load_ext autoreload; after editing src/ files, restart the session and rerun the setup cell
- **Lesson**: Check the runtime's Python version first; prefer explicit restarts over magic extensions

### E02 (2026-10-25)
- **Where**: 02_generate_paths / step 6 (pilot, 30 calibration questions)
- **Error**: `Low path diversity: 7 of 10 evidence sentences identical; 2.0 distinct answers per question; majority vote below greedy`
- **Cause**: Sampling at temperature 0.7, top-p 0.9 is too conservative for FlanT5, which favours very short outputs (mode collapse)
- **Fix**: Raised to temperature 1.0, top-p 0.95 (tuned on calibration questions only): oracle 76.7% -> 83.3%, distinct answers 2.0 -> 2.4
- **Lesson**: Check path diversity before a long run; evidence remains short (~3 words), noted as a limitation

### E03 (2026-10-25)
- **Where**: 03_baselines / step 7 (LXMERT)
- **Error**: `RepositoryNotFoundError: 404 for unc-nlp/frcnn-vg-finetuned`
- **Cause**: Hugging Face removed the Faster R-CNN checkpoint used by the official LXMERT demo to extract image features
- **Fix**: Used the original LXMERT object features for GQA testdev released by the authors (gqa_testdev_obj36), the same features as in the paper
- **Lesson**: Third-party checkpoints can disappear; prefer authors' released artefacts and keep fallbacks

### E04 (2026-10-25)
- **Where**: 04_cwmpvr / steps 6-8 (final evaluation)
- **Error**: `Weights tuned on calibration (val) lowered testdev accuracy (39.6% vs 40.7% majority); calib/eval accuracy surfaces correlated -0.33`
- **Cause**: Distribution shift: val images are 100% Visual Genome (seen in BLIP-2 pretraining), testdev 0%; ITM grounding is much sharper on seen images (AUC 0.603 vs 0.544), so val over-weights grounding
- **Fix**: Reported the pre-specified result unchanged and added 5-fold image-grouped cross-validation on testdev: CW-MPVR 41.2% vs 40.7% majority (p = 0.13), stable weights across folds
- **Lesson**: A calibration split must match the test image source; a model's own confidence can be inflated on its pretraining data
