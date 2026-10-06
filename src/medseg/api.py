"""HTTP inference service for left atrium segmentation.

Run (from repo root, trained checkpoints in checkpoints/):
    uvicorn medseg.api:app --host 127.0.0.1 --port 8000

Then, e.g. in PowerShell:
    curl.exe -F "file=@data/Task02_Heart/imagesTs/la_001.nii.gz" `
        http://127.0.0.1:8000/segment -o la_001_mask.nii.gz
    curl.exe -F "file=@data/Task02_Heart/imagesTs/la_001.nii.gz" `
        "http://127.0.0.1:8000/segment?output=json"

Interactive docs: http://127.0.0.1:8000/docs
Config file: configs/default.yaml, or the path in the HEARTSEG_CONFIG environment variable.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from omegaconf import OmegaConf

from medseg import __version__

SUFFIXES = (".nii.gz", ".nii")


def _suffix(filename: str) -> str | None:
    name = filename.lower()
    return next((s for s in SUFFIXES if name.endswith(s)), None)


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
        output: Annotated[str, Query(pattern="^(nifti|json)$")] = "nifti",
    ):
        suffix = _suffix(file.filename or "")
        if suffix is None:
            raise HTTPException(415, "Upload a NIfTI file (.nii or .nii.gz).")
        data = file.file.read()
        if len(data) > max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File larger than {max_upload_mb} MB.")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"input{suffix}"
            path.write_bytes(data)
            try:
                result = app.state.segmenter.predict_file(path)
            except Exception as e:  # unreadable file, wrong dimensionality, ...
                raise HTTPException(422, f"Could not segment this file: {e}") from e

        summary = {
            "volume_ml": round(result.volume_ml, 2),
            "n_components": result.n_components,
            "shape": list(result.mask.shape),
            "spacing_mm": [round(s, 4) for s in result.spacing_mm],
        }
        if output == "json":
            return JSONResponse(summary)

        stem = (file.filename or "image")[: -len(suffix)]
        return Response(
            content=result.to_nifti_gz(),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}_mask.nii.gz"',
                "X-Volume-ml": str(summary["volume_ml"]),
                "X-Components": str(summary["n_components"]),
            },
        )

    return app


app = create_app()
