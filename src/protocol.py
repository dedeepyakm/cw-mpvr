"""Evaluation protocol helpers: repeated image-grouped cross-validation and per-type comparisons.

Written from notebooks 06/07. Repeat 0 of `image_folds` reproduces data/cv_folds.csv (notebook 05).
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import cwmpvr as CW
from . import evaluate as E

BETAS = [0, 0.25, 0.5, 1, 2, 4, 8]
GAMMAS = [0, 0.25, 0.5, 1, 2, 4, 8, 16]
VARIANTS = {"Answer-only (CV)": (BETAS, [0]), "Grounding-only (CV)": ([0], GAMMAS), "CW-MPVR (CV)": (BETAS, GAMMAS)}


def image_folds(image_ids, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign every question to a fold so that all questions of one image share a fold."""
    imgs = np.random.default_rng(seed).permutation(sorted(set(image_ids)))
    fold_of = {im: i % n_folds for i, im in enumerate(imgs)}
    return np.array([fold_of[i] for i in image_ids])


def cross_validate(paths, gold: pd.Series, folds, betas, gammas, grounding="itm"):
    """Tune (beta, gamma) on k-1 folds, predict the held-out fold; every prediction is out-of-fold."""
    ids = gold.index
    out, chosen = [], []
    for f in np.unique(folds):
        tr, te = ids[folds != f], ids[folds == f]
        b = CW.select(CW.grid_search(paths[paths.id.isin(tr)], gold.loc[tr], betas, gammas, grounding))
        chosen.append((float(b.beta), float(b.gamma)))
        out.append(CW.cw_mpvr(paths[paths.id.isin(te)], b.beta, b.gamma, grounding))
    return pd.concat(out).reindex(ids), chosen


def repeated_cv(paths, gold: pd.Series, image_ids, repeats: int = 5, n_folds: int = 5, seed: int = 42):
    """R x k-fold CV for every tuned variant. Returns {variant: [(pred_frame, chosen), ...]}."""
    return {name: [cross_validate(paths, gold, image_folds(image_ids, n_folds, seed + r), bs, gs)
                   for r in range(repeats)]
            for name, (bs, gs) in VARIANTS.items()}


def compare(ok_a, ok_b, imgs, seed: int = 42) -> dict:
    """Paired comparison A - B: accuracy difference, image-clustered bootstrap CI, McNemar."""
    ok_a, ok_b, imgs = np.asarray(ok_a), np.asarray(ok_b), np.asarray(imgs)
    bs = E.paired_bootstrap(ok_a, ok_b, imgs, seed=seed)
    mc = E.mcnemar(ok_a, ok_b)
    return {"A − B (pts)": 100 * bs["diff"], "CI low": 100 * bs["ci_lo"], "CI high": 100 * bs["ci_hi"],
            "only A": mc["only_A"], "only B": mc["only_B"], "p McNemar": mc["p_mcnemar"], "p bootstrap": bs["p_boot"]}


def holm(df: pd.DataFrame, cols=("p McNemar", "p bootstrap")) -> pd.DataFrame:
    for c in cols:
        df[c + " (Holm)"] = multipletests(df[c], method="holm")[1]
    return df


def type_masks(meta: pd.DataFrame) -> dict:
    """Question groups promised in the proposal, plus every semantic type."""
    st, se = meta["structural"].values, meta["semantic"].values
    masks = {"all": np.ones(len(meta), bool),
             "relational (semantic: rel)": se == "rel",
             "attribute (semantic: attr)": se == "attr",
             "logical (structural)": st == "logical",
             "comparative (structural: compare)": st == "compare"}
    names = {"obj": "object", "cat": "category", "global": "global"}
    for t, name in names.items():
        if (se == t).any():
            masks[f"{name} (semantic: {t})"] = se == t
    for t in ["query", "verify", "choose"]:
        masks[f"{t} (structural)"] = st == t
    return masks


def type_table(correct: pd.DataFrame, meta: pd.DataFrame, target: str, ref: str, imgs, seed: int = 42):
    """Accuracy per question group, and target-vs-ref gain per group (Holm over groups, excluding 'all')."""
    masks = type_masks(meta)
    acc = pd.DataFrame({g: 100 * correct[m].mean() for g, m in masks.items()}).T.round(2)
    acc.insert(0, "n", [int(m.sum()) for m in masks.values()])
    gains = pd.DataFrame([{"group": g, "n": int(m.sum()),
                           **compare(correct[target].values[m], correct[ref].values[m], np.asarray(imgs)[m], seed)}
                          for g, m in masks.items()])
    rest = gains["group"] != "all"
    for c in ("p McNemar", "p bootstrap"):
        gains.loc[rest, c + " (Holm)"] = multipletests(gains.loc[rest, c], method="holm")[1]
    return acc, gains
