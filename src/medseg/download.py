"""Download and extract an MSD task using MONAI's built-in mirror.

Usage:
    python -m medseg.download                       # Task02_Heart into ./data
    python -m medseg.download --root data --task Task02_Heart
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monai.apps import DecathlonDataset


def download(root: str | Path = "data", task: str = "Task02_Heart") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # download=True fetches the archive, checks its md5, and extracts it.
    # cache_rate=0 -> do not load any images into memory.
    DecathlonDataset(
        root_dir=str(root),
        task=task,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=0,
    )
    out = root / task
    if not (out / "dataset.json").exists():
        raise FileNotFoundError(f"dataset.json not found in {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--task", default="Task02_Heart")
    args = p.parse_args()
    print(f"Dataset ready at: {download(args.root, args.task)}")


if __name__ == "__main__":
    main()