"""Segment new NIfTI images with the final pipeline.

Loads the trained fold models once (mean-softmax ensemble), applies the same preprocessing as
evaluation, keeps the largest connected component, and maps the prediction back to the
original image space: the returned mask has the input's shape and affine, so it overlays
directly on the input scan in any viewer.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from scipy import ndimage

from medseg.eval import ensemble_probs, keep_largest_component, load_model


@dataclass
class SegmentationResult:
    mask: np.ndarray  # uint8, shape of the input image, 1 = left atrium
    affine: np.ndarray  # affine of the input image
    volume_ml: float
    n_components: int
    spacing_mm: tuple[float, float, float]

    def to_nifti_gz(self) -> bytes:
        """The mask as a gzipped NIfTI file, aligned with the input image."""
        return gzip.compress(nib.Nifti1Image(self.mask, self.affine).to_bytes())


class Segmenter:
    def __init__(self, models, spacing, patch_size, postprocess: str = "lcc", device=None):
        if postprocess not in ("none", "lcc"):
            raise ValueError("postprocess must be 'none' or 'lcc'")
        if not models:
            raise ValueError("need at least one model")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [m.to(self.device).eval() for m in models]
        self.patch_size = tuple(int(p) for p in patch_size)
        self.postprocess = postprocess
        self.pre = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=tuple(float(s) for s in spacing), mode="bilinear"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys="image"),
            ]
        )
        # Undo orientation and resampling on the prediction (nearest neighbour for a mask).
        self.invert = Invertd(
            keys="pred", transform=self.pre, orig_keys="image", nearest_interp=True, to_tensor=True
        )

    @classmethod
    def from_config(cls, cfg, device=None) -> Segmenter:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = [load_model(cfg, int(f), device) for f in cfg.serve.folds]
        return cls(
            models, cfg.data.spacing, cfg.data.patch_size, str(cfg.serve.postprocess), device
        )

    @torch.no_grad()
    def predict_file(self, path: str | Path) -> SegmentationResult:
        path = Path(path)
        original = nib.load(str(path))
        d = self.pre({"image": str(path)})

        x = d["image"].unsqueeze(0).to(self.device)
        probs = ensemble_probs(
            [sliding_window_inference(x, self.patch_size, 4, m) for m in self.models]
        )
        fg = (probs[0].argmax(dim=0) == 1).float().cpu()
        onehot = torch.stack([1 - fg, fg]).unsqueeze(0)
        if self.postprocess == "lcc":
            onehot = keep_largest_component(onehot)

        d["pred"] = MetaTensor(onehot[0, 1:2], meta=d["image"].meta)
        inverted = self.invert(d)["pred"]
        mask = (torch.as_tensor(inverted)[0].cpu().numpy() > 0.5).astype(np.uint8)

        expected = tuple(original.shape[:3])
        if mask.shape != expected:
            raise RuntimeError(f"prediction shape {mask.shape} does not match input {expected}")

        zooms = tuple(float(z) for z in original.header.get_zooms()[:3])
        n_components = int(ndimage.label(mask, structure=np.ones((3, 3, 3)))[1])
        return SegmentationResult(
            mask=mask,
            affine=original.affine,
            volume_ml=float(mask.sum()) * float(np.prod(zooms)) / 1000.0,
            n_components=n_components,
            spacing_mm=zooms,
        )
