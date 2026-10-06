"""Data loading: case discovery, split handling, MONAI transforms, data loaders."""

from __future__ import annotations

import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)

from medseg.splits import load_splits, make_splits, save_splits


def _case_id(rel_path: str) -> str:
    name = Path(rel_path).name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def list_cases(root: str | Path) -> dict[str, dict[str, str]]:
    """Read the MSD dataset.json and return {case_id: {"image": path, "label": path}}."""
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text())
    cases = {}
    for item in meta["training"]:
        cid = _case_id(item["image"])
        cases[cid] = {
            "image": str((root / item["image"]).resolve()),
            "label": str((root / item["label"]).resolve()),
        }
    return cases


def load_or_create_splits(
    root: str | Path, split_file: str | Path, n_test: int, n_folds: int, seed: int
) -> dict:
    """Load the committed split file, or create it once from the dataset."""
    split_file = Path(split_file)
    if split_file.exists():
        return load_splits(split_file)
    splits = make_splits(list(list_cases(root)), n_test=n_test, n_folds=n_folds, seed=seed)
    save_splits(splits, split_file)
    return splits


def get_transforms(spacing, patch_size, train: bool) -> Compose:
    keys = ["image", "label"]
    common = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(spacing), mode=("bilinear", "nearest")),
        # MRI: z-score over nonzero voxels (no CT-style HU windowing)
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not train:
        return Compose([*common, EnsureTyped(keys=keys)])

    patch = tuple(patch_size)
    aug = [
        SpatialPadd(keys=keys, spatial_size=patch),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandRotate90d(keys=keys, prob=0.3, max_k=3, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]
    return Compose([*common, *aug, EnsureTyped(keys=keys)])


def _items(cases: dict, ids: list[str]) -> list[dict]:
    return [{"image": cases[i]["image"], "label": cases[i]["label"], "id": i} for i in ids]


def get_loaders(cfg, fold: int, cache: bool = True):
    """Return (train_loader, val_loader) for a CV fold. Test set is never used here."""
    d = cfg.data
    cases = list_cases(d.root)
    splits = load_or_create_splits(d.root, d.split_file, d.n_test, d.n_folds, cfg.seed)
    f = splits["folds"][fold]

    ds_cls = CacheDataset if cache else Dataset
    extra = {"cache_rate": 1.0} if cache else {}
    train_ds = ds_cls(
        _items(cases, f["train"]), get_transforms(d.spacing, d.patch_size, True), **extra
    )
    val_ds = ds_cls(
        _items(cases, f["val"]), get_transforms(d.spacing, d.patch_size, False), **extra
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader
