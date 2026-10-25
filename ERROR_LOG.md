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
