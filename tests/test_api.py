import gzip

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from medseg.api import create_app
from medseg.inference import SegmentationResult


class FakeSegmenter:
    """Avoids loading checkpoints: reads the upload and returns a fixed small mask."""

    models = [object()]
    device = "cpu"
    postprocess = "lcc"

    def predict_file(self, path):
        nib.load(str(path)).get_fdata()  # fails like the real one on an unreadable file
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        return SegmentationResult(mask, np.eye(4), 0.008, 1, (1.0, 1.0, 1.0))


def _nifti_gz() -> bytes:
    img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    return gzip.compress(img.to_bytes())


client = TestClient(create_app(segmenter=FakeSegmenter()))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["models"] == 1


def test_segment_json():
    r = client.post(
        "/segment?output=json",
        files={"file": ("case.nii.gz", _nifti_gz(), "application/gzip")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["volume_ml"] == 0.01
    assert body["n_components"] == 1
    assert body["shape"] == [4, 4, 4]


def test_segment_returns_nifti_mask():
    r = client.post("/segment", files={"file": ("case.nii.gz", _nifti_gz(), "application/gzip")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
    assert 'filename="case_mask.nii.gz"' in r.headers["content-disposition"]
    assert r.headers["x-components"] == "1"
    mask = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(r.content)).dataobj)
    assert mask.shape == (4, 4, 4)
    assert mask.sum() == 8


def test_rejects_non_nifti():
    r = client.post("/segment", files={"file": ("scan.png", b"not an image", "image/png")})
    assert r.status_code == 415


def test_unreadable_nifti_gives_422():
    r = client.post("/segment", files={"file": ("broken.nii.gz", b"garbage", "application/gzip")})
    assert r.status_code == 422
