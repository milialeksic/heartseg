"""HTTP inference service and web UI for left atrium segmentation.

Run (from repo root, trained checkpoints in checkpoints/):
    uvicorn medseg.api:app --host 127.0.0.1 --port 8000

Web UI:            http://127.0.0.1:8000/
Interactive docs:  http://127.0.0.1:8000/docs

From the command line, e.g. in PowerShell:
    curl.exe -F "file=@data/Task02_Heart/imagesTs/la_001.nii.gz" `
        http://127.0.0.1:8000/segment -o la_001_mask.nii.gz
    curl.exe -F "file=@data/Task02_Heart/imagesTs/la_001.nii.gz" `
        "http://127.0.0.1:8000/segment?output=json"

Outputs of POST /segment (query parameter `output`):
    nifti    the mask as .nii.gz, aligned with the input (default)
    json     volume, components, shape, spacing
    preview  json plus rendered PNG previews and the mask (base64), used by the web UI

Config file: configs/default.yaml, or the path in the HEARTSEG_CONFIG environment variable.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from omegaconf import OmegaConf

from medseg import __version__
from medseg.ui import INDEX_HTML

SUFFIXES = (".nii.gz", ".nii")


def _suffix(filename: str) -> str | None:
    name = filename.lower()
    return next((s for s in SUFFIXES if name.endswith(s)), None)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _preview(path: Path, result) -> dict:
    """Rendered previews of the prediction on the input image, for the web UI."""
    from medseg.viz import render_overview, render_slices  # matplotlib only needed here

    image = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if image.ndim == 4:
        image = image[..., 0]
    return {
        "overview_png": _b64(render_overview(image, result.mask, result.spacing_mm)),
        "slices": [
            {"index": k, "png": _b64(with_seg), "png_raw": _b64(without_seg)}
            for k, with_seg, without_seg in render_slices(image, result.mask, result.spacing_mm)
        ],
        "n_slices_total": int(image.shape[2]),
        "mask_nii_gz": _b64(result.to_nifti_gz()),
    }


def create_app(segmenter=None, max_upload_mb: int = 200) -> FastAPI:
    """Build the app. Pass a segmenter to skip loading checkpoints (used in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.segmenter is None:
            from medseg.inference import Segmenter  # heavy imports only when serving

            cfg = OmegaConf.load(os.environ.get("HEARTSEG_CONFIG", "configs/default.yaml"))
            app.state.segmenter = Segmenter.from_config(cfg)
        yield

    app = FastAPI(
        title="heartseg",
        version=__version__,
        description="Left atrium segmentation of 3D cardiac MRI (research use only).",
        lifespan=lifespan,
    )
    app.state.segmenter = segmenter

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @app.get("/health")
    def health() -> dict:
        s = app.state.segmenter
        return {
            "status": "ok",
            "models": len(s.models),
            "device": str(s.device),
            "postprocess": s.postprocess,
        }

    @app.post("/segment")
    def segment(
        file: Annotated[UploadFile, File(description="3D MRI volume, .nii or .nii.gz")],
        output: Annotated[str, Query(pattern="^(nifti|json|preview)$")] = "nifti",
    ):
        suffix = _suffix(file.filename or "")
        if suffix is None:
            raise HTTPException(415, "Upload a NIfTI file (.nii or .nii.gz).")
        data = file.file.read()
        if len(data) > max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File larger than {max_upload_mb} MB.")
        stem = (file.filename or "image")[: -len(suffix)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"input{suffix}"
            path.write_bytes(data)
            start = time.perf_counter()
            try:
                result = app.state.segmenter.predict_file(path)
            except Exception as e:  # unreadable file, wrong dimensionality, ...
                raise HTTPException(422, f"Could not segment this file: {e}") from e
            elapsed = time.perf_counter() - start
            preview = _preview(path, result) if output == "preview" else None

        summary = {
            "volume_ml": round(result.volume_ml, 2),
            "n_components": result.n_components,
            "shape": list(result.mask.shape),
            "spacing_mm": [round(s, 4) for s in result.spacing_mm],
            "elapsed_s": round(elapsed, 2),
            "filename": f"{stem}_mask.nii.gz",
        }
        if output == "json":
            return JSONResponse(summary)
        if output == "preview":
            seg = app.state.segmenter
            pipeline = {"models": len(seg.models), "postprocess": seg.postprocess}
            return JSONResponse({**summary, **preview, **pipeline})

        return Response(
            content=result.to_nifti_gz(),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{summary["filename"]}"',
                "X-Volume-ml": str(summary["volume_ml"]),
                "X-Components": str(summary["n_components"]),
            },
        )

    return app


app = create_app()
