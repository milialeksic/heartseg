"""Small helpers: seeding and git provenance."""

from __future__ import annotations

import random
import subprocess

import numpy as np
import torch
from monai.utils import set_determinism


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch and MONAI for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def git_commit() -> str:
    """Current git commit hash, or 'unknown' outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    """True if there are uncommitted changes (the logged commit hash would then be misleading)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return False