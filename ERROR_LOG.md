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
