"""CW-MPVR: Confidence-Weighted Multi-Path Visual Reasoning, implemented from scratch.

Each path i of a question gets a score from two separate confidence signals:

    s_i = beta * a_i + gamma * g_i
    a_i = mean log-probability of the answer tokens        (answer confidence)
    g_i = log P(match | image, evidence) from BLIP-2 ITM    (grounding confidence)

Weights are a softmax of s_i within the question; the answer with the largest
total weight wins. Special cases of the same rule:
    beta = gamma = 0  -> majority vote (self-consistency)
    gamma = 0         -> answer-only weighting
    beta = 0          -> grounding-only weighting
    beta = gamma      -> one combined (non-decomposed) confidence
"""

import itertools

import numpy as np
import pandas as pd

from .aggregate import softmax_weights, weighted_vote

EPS = 1e-6


def grounding_signal(paths: pd.DataFrame, kind: str = "itm") -> np.ndarray:
    """ITM is a probability -> use its log; ITC is a similarity -> use as is."""
    v = paths[kind].values.astype(float)
    return np.log(np.clip(v, EPS, 1.0)) if kind == "itm" else v


def path_scores(paths, beta, gamma, grounding="itm"):
    return beta * paths["lp_mean"].values + gamma * grounding_signal(paths, grounding)


def cw_mpvr(paths, beta, gamma, grounding="itm") -> pd.DataFrame:
    """Predicted answer and support (winning weight share) for every question."""
    w = softmax_weights(paths, path_scores(paths, beta, gamma, grounding))
    return weighted_vote(paths, w)


def accuracy(pred: pd.Series, gold: pd.Series, sample_weight=None) -> float:
    ok = (pred.reindex(gold.index) == gold).values
    return float(np.average(ok, weights=sample_weight))


def grid_search(paths, gold, betas, gammas, grounding="itm", sample_weight=None) -> pd.DataFrame:
    """Accuracy of every (beta, gamma) pair on a labelled split (calibration only)."""
    rows = []
    for b, g in itertools.product(betas, gammas):
        pred = cw_mpvr(paths, b, g, grounding)["answer"]
        rows.append({"beta": b, "gamma": g, "acc": accuracy(pred, gold, sample_weight)})
    return pd.DataFrame(rows)


def select(grid: pd.DataFrame) -> pd.Series:
    """Highest accuracy; ties go to the simplest setting (smallest beta + gamma)."""
    g = grid.assign(size=grid["beta"] + grid["gamma"])
    return g.sort_values(["acc", "size"], ascending=[False, True]).iloc[0]


def question_features(paths, pred: pd.DataFrame, grounding="itm") -> pd.DataFrame:
    """Per-question confidence features for calibration and abstention (notebook 05)."""
    p = paths.assign(win=paths["answer"].values == pred["answer"].reindex(paths["id"]).values)
    agree = p.groupby("id")["win"].mean()
    sup = p[p["win"]].groupby("id")
    return pd.DataFrame({
        "support": pred["support"],
        "agreement": agree,                                  # share of paths giving the answer
        "answer_conf": sup["lp_mean"].mean(),                # answer confidence of those paths
        "grounding_conf": sup[grounding].mean(),             # grounding confidence of those paths
    })
