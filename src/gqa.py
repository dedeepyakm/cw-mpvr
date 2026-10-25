"""GQA helpers: loading questions, scene-graph statistics, stratified sampling.

Written from notebooks/01_data_eda.ipynb; imported by later notebooks.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd

HF_REPO = "lmms-lab/GQA"                          # mirror of the official GQA release
CONFIG = {"testdev": "testdev_balanced", "val": "val_balanced"}
TYPE_COLS = ["structural", "semantic"]           # GQA's two question-type labels


def load_questions(split: str) -> pd.DataFrame:
    """Load balanced GQA questions for 'testdev' or 'val' as a tidy table."""
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, f"{CONFIG[split]}_instructions", split=split)
    return to_frame(ds.to_pandas(), split)


def _type(t, key):
    return (t or {}).get(key)


def _n_steps(raw: pd.DataFrame) -> pd.Series:
    """Reasoning steps = number of operations in GQA's semantic program."""
    if "semantic" in raw.columns:
        return raw["semantic"].map(lambda p: 0 if p is None else len(p))
    return raw["semanticStr"].map(lambda s: len(str(s).split("->")) if s else 0)


def to_frame(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Keep only the columns the project needs."""
    df = pd.DataFrame({
        "id": raw["id"].astype(str),
        "imageId": raw["imageId"].astype(str),
        "question": raw["question"],
        "answer": raw["answer"],
        "structural": raw["types"].map(lambda t: _type(t, "structural")),
        "semantic": raw["types"].map(lambda t: _type(t, "semantic")),
        "detailed": raw["types"].map(lambda t: _type(t, "detailed")),
        "n_steps": _n_steps(raw),
    })
    df["q_words"] = df["question"].str.split().str.len()
    df["split"] = split
    return df


def stratified_sample(df: pd.DataFrame, n: int, by=TYPE_COLS, seed: int = 42) -> pd.DataFrame:
    """Proportional stratified sample of n rows.

    Each stratum (e.g. structural x semantic cell) gets n * its share of the data,
    rounded with the largest-remainder rule so the total is exactly n.
    """
    sizes = df.groupby(by).size()
    quota = sizes / sizes.sum() * n
    take = np.floor(quota).astype(int)
    extra = (quota - take).sort_values(ascending=False, kind="mergesort").index[: n - take.sum()]
    take.loc[extra] += 1
    parts = [g.sample(take.loc[k], random_state=seed) for k, g in df.groupby(by) if take.loc[k] > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_scene_graphs(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def scene_graph_stats(sg: dict) -> tuple[pd.DataFrame, Counter]:
    """Per-image object / attribute / relation counts, plus object-name frequencies."""
    rows, names = [], Counter()
    for img, g in sg.items():
        objs = list(g["objects"].values())
        names.update(o["name"] for o in objs)
        rows.append({
            "imageId": img,
            "n_objects": len(objs),
            "n_attributes": sum(len(o["attributes"]) for o in objs),
            "n_relations": sum(len(o["relations"]) for o in objs),
        })
    return pd.DataFrame(rows), names
