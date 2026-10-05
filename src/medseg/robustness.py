"""Robustness of the final pipeline to simulated acquisition changes.

Cross-validation cases only (the test set is used). Each fold model is evaluated on its own
validation cases (out-of-fold). The image is perturbed after resampling and before intensity
normalisation, as a different scanner or protocol would change it. Labels are never perturbed.

Perturbations (severity levels in configs/default.yaml, section `robustness`):
    noise   Gaussian noise, std as a fraction of the foreground intensity std
    bias    smooth multiplicative bias field (coil inhomogeneity), polynomial coefficient size
    gamma   non-linear contrast change, log-gamma
    motion  simulated patient motion (ghosting), rotation in degrees and translation in mm
    lowres  thicker slices along axis 2, downsampling factor

Usage (from repo root):
    python -m medseg.robustness
    python -m medseg.robustness "robustness.kinds=[noise,bias]"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchio as tio
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureTyped, NormalizeIntensityd
from omegaconf import OmegaConf

from medseg.data import base_transforms, load_or_create_splits
from medseg.eval import evaluate_ids, load_model
from medseg.utils import git_commit

KINDS = ("noise", "bias", "gamma", "motion", "lowres")


def apply_perturbation(
    img: torch.Tensor, kind: str, severity: float, seed: int = 0, affine=None
) -> torch.Tensor:
    """Perturb a (1, X, Y, Z) image with raw (non-normalised) intensities. Same shape out."""
    if kind == "none":
        return img
    if kind not in KINDS:
        raise ValueError(f"Unknown perturbation {kind!r}; choose from {KINDS}")

    x = img.float()
    if kind == "noise":
        fg = x[x > 0]
        sigma = severity * (fg.std() if fg.numel() > 1 else x.std())
        gen = torch.Generator().manual_seed(seed)
        return x + torch.randn(x.shape, generator=gen) * sigma

    torch.manual_seed(seed)  # TorchIO samples its random parameters from torch's RNG
    aff = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    if kind == "gamma":
        x = x.clamp(min=0)  # gamma needs non-negative intensities
    image = tio.ScalarImage(tensor=x, affine=aff)
    if kind == "bias":
        t = tio.RandomBiasField(coefficients=severity, order=3)
    elif kind == "gamma":
        t = tio.RandomGamma(log_gamma=(severity, severity))
    elif kind == "motion":
        t = tio.RandomMotion(degrees=severity, translation=severity, num_transforms=2)
    else:  # lowres
        t = tio.RandomAnisotropy(axes=(2,), downsampling=(severity, severity))
    return t(image).data.float()


class PerturbImage:
    """Dict transform applying `apply_perturbation` to d["image"], seeded per case."""

    def __init__(self, kind: str, severity: float, seed: int = 0):
        self.kind, self.severity, self.seed = kind, float(severity), int(seed)

    def __call__(self, d: dict) -> dict:
        if self.kind == "none":
            return d
        img = d["image"]
        raw = img.as_tensor() if isinstance(img, MetaTensor) else torch.as_tensor(img)
        affine = img.affine.cpu().numpy() if isinstance(img, MetaTensor) else None
        case_seed = self.seed + sum(map(ord, str(d.get("id", ""))))
        out = apply_perturbation(raw.cpu(), self.kind, self.severity, case_seed, affine)
        d = dict(d)
        d["image"] = MetaTensor(out, meta=img.meta) if isinstance(img, MetaTensor) else out
        return d


def conditions(cfg) -> list[tuple[str, float]]:
    r = cfg.robustness
    out = [("none", 0.0)]
    for kind in r.kinds:
        out += [(str(kind), float(s)) for s in r.levels[kind]]
    return out


def summarize_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: mean Dice, paired change vs the clean image, worst case, mean HD95."""
    clean = df[df["kind"] == "none"].set_index("case_id")
    d = df.copy()
    d["delta_dice"] = d["dice"] - d["case_id"].map(clean["dice"])
    d["delta_hd95_mm"] = d["hd95_mm"] - d["case_id"].map(clean["hd95_mm"])
    return (
        d.groupby(["kind", "severity"], sort=False)
        .agg(
            dice=("dice", "mean"),
            delta_dice=("delta_dice", "mean"),
            worst_dice=("dice", "min"),
            hd95_mm=("hd95_mm", "mean"),
            delta_hd95_mm=("delta_hd95_mm", "mean"),
            n_cases=("dice", "size"),
        )
        .reset_index()
    )


def summary_markdown(s: pd.DataFrame) -> str:
    lines = [
        "| Perturbation | Severity | Dice | Δ Dice vs clean | Worst-case Dice | HD95 (mm) "
        "| Δ HD95 (mm) | Cases |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"| {r['kind']} | {r['severity']:g} | {r['dice']:.3f} | {r['delta_dice']:+.3f} | "
            f"{r['worst_dice']:.3f} | {r['hd95_mm']:.1f} | {r['delta_hd95_mm']:+.1f} | "
            f"{int(r['n_cases'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    base = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.from_cli())
    r = cfg.robustness
    for k in r.kinds:
        if k not in KINDS:
            raise ValueError(f"Unknown perturbation {k!r}; choose from {KINDS}")

    # Final pipeline settings; no per-case images, to keep the run fast.
    cfg.eval.postprocess = str(r.postprocess)
    cfg.eval.save_overlays = False
    cfg.eval.save_pred = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_or_create_splits(
        cfg.data.root, cfg.data.split_file, cfg.data.n_test, cfg.data.n_folds, cfg.seed
    )
    folds = range(int(cfg.data.n_folds))
    models = {f: load_model(cfg, f, device) for f in folds}
    out_dir = Path(r.out_dir)

    frames = []
    for kind, sev in conditions(cfg):
        print(f"\n=== {kind} (severity {sev:g}) ===")
        tf = Compose(
            [
                *base_transforms(cfg.data.spacing),
                PerturbImage(kind, sev, cfg.seed),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        for f in folds:
            ids = splits["folds"][f]["val"]
            df = evaluate_ids(cfg, [models[f]], ids, device, f, "val", out_dir, transforms=tf)
            df["kind"], df["severity"] = kind, sev
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["git_commit"] = git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{cfg.model.name}_robustness.csv"
    df.to_csv(csv, index=False)

    s = summarize_robustness(df)
    table = summary_markdown(s)
    res_dir = Path(r.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / csv.name, index=False)
    (res_dir / "summary.md").write_text(
        "# Robustness (cross-validation cases, out-of-fold, "
        f"post-processing: {r.postprocess})\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)
    print(f"\nPer-case results: {csv}\nSummary: {res_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
