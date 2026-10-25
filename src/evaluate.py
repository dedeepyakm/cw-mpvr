"""Evaluation: accuracy, image-clustered bootstrap intervals, paired tests."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _cluster_sums(values: np.ndarray, clusters: np.ndarray):
    codes, uniq = pd.factorize(clusters)
    s = np.bincount(codes, weights=values.astype(float))
    n = np.bincount(codes).astype(float)
    return s, n


def cluster_bootstrap(correct, clusters, n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    """95% CI for accuracy, resampling whole images (questions on one image are not independent)."""
    s, n = _cluster_sums(np.asarray(correct), np.asarray(clusters))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=n_boot)
    acc = counts @ s / (counts @ n)
    return np.quantile(acc, [alpha / 2, 1 - alpha / 2])


def paired_bootstrap(correct_a, correct_b, clusters, n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    """Accuracy difference A - B with an image-clustered bootstrap CI and two-sided p-value."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    sd, n = _cluster_sums(a - b, np.asarray(clusters))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(sd), np.full(len(sd), 1 / len(sd)), size=n_boot)
    diff = counts @ sd / (counts @ n)
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"diff": (a - b).mean(), "ci_lo": lo, "ci_hi": hi, "p_boot": p}


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test on the questions where exactly one method is right."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_A": only_a, "only_B": only_b, "p_mcnemar": p}


def accuracy_table(correct: pd.DataFrame, clusters, seed: int = 42) -> pd.DataFrame:
    """Accuracy (%) with 95% image-clustered bootstrap CI for each column of `correct`."""
    rows = {}
    for m in correct.columns:
        lo, hi = cluster_bootstrap(correct[m].values, clusters, seed=seed)
        rows[m] = {"accuracy": 100 * correct[m].mean(), "ci_low": 100 * lo, "ci_high": 100 * hi}
    return pd.DataFrame(rows).T.round(2)
