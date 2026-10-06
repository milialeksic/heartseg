"""Per-case evaluation of trained checkpoints.

Usage (from repo root):
        python -m medseg.eval fold=all                       # out-of-fold results, all CV cases
    python -m medseg.eval fold=all eval.postprocess=lcc  # same, keep largest component
    python -m medseg.eval fold=0 split=test              # held-out test set, use once
    python -m medseg.eval fold=all eval.save_pred=true   # also write NIfTI per case
    python -m medseg.eval fold=ensemble split=test eval.postprocess=lcc
        # mean-softmax ensemble of all fold models; test set only, because every CV case
        # was seen in training by the other fold models

Outputs (in cfg.eval.out_dir):
    <tag>.csv                 per-case metrics
    <tag>/overlays/<id>.png   slice overlays (if eval.save_overlays)
    <tag>/preds/<id>_*.nii.gz image, ground truth, prediction (if eval.save_pred)
where <tag> = <model>_fold<fold>_<split>[_lcc].
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_surface_dice,
)
from monai.transforms import AsDiscrete
from omegaconf import OmegaConf
from scipy import ndimage

from medseg.data import get_transforms, list_cases, load_or_create_splits
from medseg.models import build_model
from medseg.utils import git_commit

METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95_mm",
    "assd_mm",
    "nsd",
    "abs_vol_err_ml",
    "rel_vol_err",
    "n_components",
]

_CONN26 = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity for 3D components


def _n_components(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=_CONN26)[1])


def keep_largest_component(pred: torch.Tensor) -> torch.Tensor:
    """Keep only the largest foreground component of a (1, 2, H, W, D) one-hot prediction."""
    if pred.shape[1] != 2:
        raise ValueError("keep_largest_component expects 2 classes (background, foreground)")
    fg = pred[0, 1].cpu().numpy() > 0.5
    labels, n = ndimage.label(fg, structure=_CONN26)
    if n <= 1:
        return pred
    sizes = ndimage.sum(fg, labels, index=range(1, n + 1))
    keep = torch.from_numpy(labels == int(np.argmax(sizes)) + 1).to(pred.dtype)
    return torch.stack([1 - keep, keep]).unsqueeze(0)


def ensemble_probs(logits: list[torch.Tensor]) -> torch.Tensor:
    """Average the class probabilities (softmax over channel dim 1) of several model outputs.

    With a single model this is just its softmax, so argmax is unchanged."""
    if not logits:
        raise ValueError("need at least one model output")
    return torch.stack([torch.softmax(x.float(), dim=1) for x in logits]).mean(dim=0)


def case_metrics(
    pred: torch.Tensor, gt: torch.Tensor, spacing: tuple[float, ...], nsd_tol_mm: float
) -> dict[str, float]:
    """Metrics for one case.

    pred, gt: one-hot tensors of shape (1, C, H, W, D), channel 0 = background.
    Distances are in millimetres (via `spacing`); volumes in millilitres.
    Distance metrics are inf/nan if either mask is empty.
    """
    n_cls = pred.shape[1]
    dice = compute_dice(pred, gt, include_background=False).item()
    hd95 = compute_hausdorff_distance(
        pred, gt, include_background=False, percentile=95, spacing=spacing
    ).item()
    assd = compute_average_surface_distance(
        pred, gt, include_background=False, symmetric=True, spacing=spacing
    ).item()
    # Thresholds are given for all classes; we keep the foreground class (index 1).
    nsd = compute_surface_dice(
        pred, gt, class_thresholds=[nsd_tol_mm] * n_cls, include_background=True, spacing=spacing
    )[0, 1].item()

    p = pred[0, 1] > 0.5
    g = gt[0, 1] > 0.5
    tp = float((p & g).sum())
    ps, gs = float(p.sum()), float(g.sum())
    nan = float("nan")
    voxel_ml = float(np.prod(spacing)) / 1000.0
    vol_p, vol_g = ps * voxel_ml, gs * voxel_ml

    return {
        "dice": dice,
        "iou": tp / (ps + gs - tp) if (ps + gs - tp) > 0 else nan,
        "precision": tp / ps if ps > 0 else nan,
        "recall": tp / gs if gs > 0 else nan,
        "hd95_mm": hd95,
        "assd_mm": assd,
        "nsd": nsd,
        "gt_volume_ml": vol_g,
        "pred_volume_ml": vol_p,
        "abs_vol_err_ml": abs(vol_p - vol_g),
        "rel_vol_err": (vol_p - vol_g) / vol_g if vol_g > 0 else nan,
        "n_components": _n_components(p.cpu().numpy()),
    }


def bootstrap_ci(
    values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over cases (non-finite values are dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, bootstrap CI and number of finite cases for each metric present in df."""
    rows = []
    for m in [m for m in METRICS if m in df.columns]:
        v = df[m].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        lo, hi = bootstrap_ci(v)
        rows.append(
            {
                "metric": m,
                "mean": float(finite.mean()) if len(finite) else float("nan"),
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_finite": int(len(finite)),
                "n_total": int(len(v)),
            }
        )
    return pd.DataFrame(rows)


def _affine(x) -> np.ndarray:
    aff = getattr(x, "affine", None)
    if aff is None:
        return np.eye(4)
    aff = torch.as_tensor(aff).cpu().numpy()
    return aff[0] if aff.ndim == 3 else aff


def _save_nifti(out_dir: Path, case_id: str, image, gt, pred, affine) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image.astype(np.float32), affine), out_dir / f"{case_id}_image.nii.gz")
    nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine), out_dir / f"{case_id}_gt.nii.gz")
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), out_dir / f"{case_id}_pred.nii.gz")


@torch.no_grad()
def evaluate_ids(
    cfg, models: list, ids: list[str], device, fold, split: str, case_dir: Path
) -> pd.DataFrame:
    """Predict and score each case. With several models, their softmax outputs are averaged."""
    cases = list_cases(cfg.data.root)
    items = [{"image": cases[i]["image"], "label": cases[i]["label"], "id": i} for i in ids]
    ds = Dataset(items, get_transforms(cfg.data.spacing, cfg.data.patch_size, train=False))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    n_cls = int(cfg.model.out_channels)
    spacing = tuple(float(s) for s in cfg.data.spacing)
    tol = float(cfg.eval.nsd_tolerance_mm)
    post = str(cfg.eval.get("postprocess", "none"))
    save_pred = bool(cfg.eval.get("save_pred", False))
    save_overlays = bool(cfg.eval.get("save_overlays", False))
    to_onehot_pred = AsDiscrete(argmax=True, to_onehot=n_cls)
    to_onehot_gt = AsDiscrete(to_onehot=n_cls)

    for m in models:
        m.eval()
    rows = []
    for batch in loader:
        cid = batch["id"][0]
        x = batch["image"].to(device)
        y = batch["label"]
        probs = ensemble_probs(
            [sliding_window_inference(x, tuple(cfg.data.patch_size), 4, m) for m in models]
        )
        pred = to_onehot_pred(probs[0]).unsqueeze(0).cpu()
        if post == "lcc":
            pred = keep_largest_component(pred)
        gt = to_onehot_gt(y[0]).unsqueeze(0).cpu()

        row = {"case_id": cid, "fold": fold, "split": split, "postprocess": post}
        row.update(case_metrics(pred, gt, spacing, tol))
        rows.append(row)
        print(
            f"{cid}: dice {row['dice']:.4f}  hd95 {row['hd95_mm']:.2f} mm  "
            f"nsd {row['nsd']:.4f}  components {row['n_components']}"
        )

        if save_pred or save_overlays:
            img_np = x[0, 0].detach().cpu().numpy()
            gt_np = gt[0, 1].numpy() > 0.5
            pred_np = pred[0, 1].numpy() > 0.5
            if save_pred:
                _save_nifti(case_dir / "preds", cid, img_np, gt_np, pred_np, _affine(x))
            if save_overlays:
                from medseg.viz import save_overlay  # matplotlib only needed here

                (case_dir / "overlays").mkdir(parents=True, exist_ok=True)
                save_overlay(
                    img_np,
                    gt_np,
                    pred_np,
                    spacing,
                    case_dir / "overlays" / f"{cid}.png",
                    title=f"{cid} (fold {fold})  Dice {row['dice']:.3f}  "
                    f"HD95 {row['hd95_mm']:.1f} mm  components {row['n_components']}",
                )
    return pd.DataFrame(rows)


def _splits(cfg) -> dict:
    return load_or_create_splits(
        cfg.data.root, cfg.data.split_file, cfg.data.n_test, cfg.data.n_folds, cfg.seed
    )


def load_model(cfg, fold: int, device):
    ckpt = Path("checkpoints") / f"{cfg.model.name}-fold{fold}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Train this fold first.")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.eval()


def evaluate_fold(cfg, fold: int, split: str, device, case_dir: Path) -> pd.DataFrame:
    splits = _splits(cfg)
    ids = splits["test"] if split == "test" else splits["folds"][fold]["val"]
    return evaluate_ids(cfg, [load_model(cfg, fold, device)], ids, device, fold, split, case_dir)


def evaluate_ensemble(cfg, device, case_dir: Path) -> pd.DataFrame:
    """Mean-softmax ensemble of all fold models on the held-out test set."""
    models = [load_model(cfg, f, device) for f in range(int(cfg.data.n_folds))]
    return evaluate_ids(cfg, models, _splits(cfg)["test"], device, "ensemble", "test", case_dir)


def main() -> None:
    base = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.from_cli())
    split = str(cfg.get("split", "val"))
    fold_arg = str(cfg.get("fold", 0))
    post = str(cfg.eval.get("postprocess", "none"))
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if split == "test" and fold_arg == "all":
        raise ValueError("For split=test choose one fold, e.g. fold=0")
    if post not in ("none", "lcc"):
        raise ValueError("eval.postprocess must be 'none' or 'lcc'")
    if fold_arg == "ensemble" and split != "test":
        raise ValueError(
            "fold=ensemble is only valid with split=test: every CV case was seen in training "
            "by the other fold models, so an ensemble cannot be evaluated fairly on CV."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.eval.out_dir)
    tag = f"{cfg.model.name}_fold{fold_arg}_{split}" + ("_lcc" if post == "lcc" else "")
    case_dir = out_dir / tag

    if fold_arg == "ensemble":
        df = evaluate_ensemble(cfg, device, case_dir)
    else:
        folds = list(range(int(cfg.data.n_folds))) if fold_arg == "all" else [int(fold_arg)]
        df = pd.concat(
            [evaluate_fold(cfg, f, split, device, case_dir) for f in folds], ignore_index=True
        )
    df["git_commit"] = git_commit()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.csv"
    df.to_csv(out, index=False)

    print(f"\nSummary ({tag}, n={len(df)} cases):")
    print(summarize(df).round(4).to_string(index=False))
    print(f"\nPer-case results written to {out}")


if __name__ == "__main__":
    main()
