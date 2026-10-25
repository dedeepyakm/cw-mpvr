"""Answer aggregation over cached reasoning paths, implemented from scratch.

Every method reads the same cache (one record per question, k paths each)
and differs only in how much weight each path gets in the vote.
"""

import gzip
import json

import numpy as np
import pandas as pd

from .paths import normalize

SIGNALS = ["lp_mean", "itm", "itc", "ev_lp_mean"]


def load_paths(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a .jsonl or .jsonl.gz cache into (questions, paths) tables."""
    opener = gzip.open if path.endswith(".gz") else open
    qs, ps = [], []
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            qs.append({"id": r["id"], "greedy": normalize(r["greedy"]["answer"]),
                       "greedy_lp": r["greedy"]["lp_mean"]})
            for j, p in enumerate(r["paths"]):
                ps.append({"id": r["id"], "path": j, "template": p["template"],
                           "answer": normalize(p["answer"]), **{s: p[s] for s in SIGNALS}})
    return pd.DataFrame(qs), pd.DataFrame(ps)


def weighted_vote(paths: pd.DataFrame, w) -> pd.DataFrame:
    """Pick, per question, the answer with the largest total path weight.

    Ties are broken by more paths, then by the earliest path (deterministic).
    Returns one row per question: answer, support (winning share of the weight).
    """
    d = paths[["id", "answer", "path"]].assign(w=np.asarray(w, dtype=float), n=1)
    agg = (d.groupby(["id", "answer"], sort=False)
             .agg(w=("w", "sum"), n=("n", "sum"), first=("path", "min")).reset_index())
    agg = agg.sort_values(["id", "w", "n", "first"], ascending=[True, False, False, True])
    total = agg.groupby("id")["w"].sum()
    top = agg.drop_duplicates("id").set_index("id")
    top["support"] = top["w"] / total.loc[top.index]
    return top[["answer", "support"]]


def softmax_weights(paths: pd.DataFrame, score) -> np.ndarray:
    """Turn a per-path score into weights that sum to 1 within each question."""
    s = pd.Series(np.asarray(score, dtype=float), index=paths.index)
    s = s - s.groupby(paths["id"]).transform("max")          # numerical stability
    e = np.exp(s)
    return (e / e.groupby(paths["id"]).transform("sum")).values


# ---- baseline weightings --------------------------------------------------
def majority(paths):
    """Self-consistency (Wang et al., 2023): every path counts once."""
    return weighted_vote(paths, np.ones(len(paths)))


def answer_only(paths):
    """Confidence-weighted voting: weight = exp(mean token log-prob),
    i.e. the 'normalised probability' weighting of Wang et al. (2023)."""
    return weighted_vote(paths, np.exp(paths["lp_mean"].values))
