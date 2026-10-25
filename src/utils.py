"""Shared helpers: config loading, seeding and paths."""

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "default.yaml") -> dict:
    """Read a YAML file from configs/ and create the data folders it names."""
    with open(REPO_ROOT / "configs" / name) as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "cache"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Fix every random source so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_info() -> str:
    """One-line description of the available GPU."""
    import torch
    if not torch.cuda.is_available():
        return "No GPU: switch Runtime > Change runtime type > GPU"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} | {p.total_memory / 1e9:.1f} GB | torch {torch.__version__}"
