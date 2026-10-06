import gzip

import nibabel as nib
import numpy as np
import pytest
import torch

from medseg.inference import Segmenter


class Threshold(torch.nn.Module):
    """Stand-in model: foreground where the normalised intensity is positive."""

    def forward(self, x):
        return torch.cat([-x, x], dim=1)


def _write_nifti(tmp_path, shape=(20, 18, 12), zooms=(2.0, 2.0, 3.0)):
    data = np.ones(shape, dtype=np.float32)
    data[5:15, 4:14, 3:9] = 101.0  # bright block = "atrium"
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    path = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path, data, affine


def _segmenter(postprocess="lcc"):
    return Segmenter(
        [Threshold()],
        spacing=(1.25, 1.25, 1.37),
        patch_size=(16, 16, 16),
        postprocess=postprocess,
        device=torch.device("cpu"),
    )


def test_prediction_is_in_original_image_space(tmp_path):
    path, data, affine = _write_nifti(tmp_path)
    r = _segmenter().predict_file(path)
    assert r.mask.shape == data.shape
    assert r.mask.dtype == np.uint8
    assert np.allclose(r.affine, affine)
    assert r.spacing_mm == (2.0, 2.0, 3.0)


def test_prediction_matches_the_bright_block(tmp_path):
    path, _, _ = _write_nifti(tmp_path)
    r = _segmenter().predict_file(path)
    assert r.mask[10, 9, 6] == 1
    assert r.mask[0, 0, 0] == 0
    assert r.n_components == 1
    expected_ml = 10 * 10 * 6 * (2.0 * 2.0 * 3.0) / 1000.0
    assert abs(r.volume_ml - expected_ml) / expected_ml < 0.15


def test_nifti_output_roundtrip(tmp_path):
    path, _, affine = _write_nifti(tmp_path)
    r = _segmenter().predict_file(path)
    img = nib.Nifti1Image.from_bytes(gzip.decompress(r.to_nifti_gz()))
    assert np.array_equal(np.asarray(img.dataobj), r.mask)
    assert np.allclose(img.affine, affine)


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        _segmenter(postprocess="median")
    with pytest.raises(ValueError):
        Segmenter([], spacing=(1, 1, 1), patch_size=(16, 16, 16))
