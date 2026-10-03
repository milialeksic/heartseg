"""Deterministic, case-level split generation."""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_splits(
    case_ids: list[str], n_test: int = 4, n_folds: int = 4, seed: int = 42
) -> dict:
    """Return {"test": [...], "folds": [{"train": [...], "val": [...]}, ...]}."""
    ids = sorted(set(case_ids))
    if len(ids) != len(case_ids):
        raise ValueError("Duplicate case ids")
    if n_test + n_folds > len(ids):
        raise ValueError("Not enough cases for requested split")

    rng = random.Random(seed)
    rng.shuffle(ids)

    test = sorted(ids[:n_test])
    rest = ids[n_test:]
    chunks = [rest[i::n_folds] for i in range(n_folds)]

    folds = []
    for k in range(n_folds):
        val = sorted(chunks[k])
        train = sorted(c for j, ch in enumerate(chunks) if j != k for c in ch)
        folds.append({"train": train, "val": val})
    return {"seed": seed, "test": test, "folds": folds}


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits, indent=2))


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())