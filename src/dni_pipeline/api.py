"""FastAPI backend exposing the DNI extraction pipeline together with the Gradio UI."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import DEFAULT_QWEN_MODEL_PATH
from .logging_service import logging_service
from .workflow import cleanup_resources, process_image
from .ui import create_ui

LOGGER = logging_service.get_logger("dni_pipeline.api")


class DniData(BaseModel):
    """Structured DNI data returned by the API."""

    nombre: Optional[str] = None
    primer_apellido: Optional[str] = None
    segundo_apellido: Optional[str] = None
    dni: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    fecha_validez: Optional[str] = None
    sexo: Optional[str] = None
    nacionalidad: Optional[str] = None


app = FastAPI(
    title="DNI Extraction API",
    description="Extract structured data from Spanish DNI images using docTR + Qwen3-VL-8B.",
    version="1.0.0",
    openapi_url="/openapi.json",
)


def _safe_unlink(path: Path) -> None:
    """Best-effort removal of temporary files."""
    try:
        if path.exists():
            path.unlink()
            LOGGER.debug("Removed temporary file %s", path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to remove temporary file %s: %s", path, exc)


async def _save_upload_to_disk(upload: UploadFile) -> Path:
    """Persist an uploaded file to disk and return the path."""
    suffix = Path(upload.filename or "").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = Path(tmp.name)
            LOGGER.debug("Stored upload %s at %s", upload.filename, tmp_path)
    finally:
        upload.file.close()
    return tmp_path


async def _run_blocking(func, *args, **kwargs):
    """Execute a blocking callable in the default thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


@app.post("/api/v1/extract", response_model=DniData)
async def extract_dni(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr_max_side: int = 1600,
    vlm_size: int = 512,
    max_new_tokens: int = 256,
    model_path: str = str(DEFAULT_QWEN_MODEL_PATH),
) -> DniData:
    """Run the pipeline for a single uploaded DNI image."""
    LOGGER.info("Received extraction request for file %s", file.filename)
    try:
        tmp_path = await _save_upload_to_disk(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    background_tasks.add_task(_safe_unlink, tmp_path)

    try:
        record = await _run_blocking(
            process_image,
            image_path=tmp_path,
            ocr_max_side=ocr_max_side,
            vlm_size=vlm_size,
            max_new_tokens=max_new_tokens,
            model_path=model_path,
        )
    except Exception as exc:
        LOGGER.exception("Pipeline execution failed for %s: %s", tmp_path, exc)
        raise HTTPException(status_code=500, detail="Extraction failed") from exc

    return DniData(**record)


@app.post("/api/v1/cleanup")
async def cleanup() -> dict[str, str]:
    """Unload heavy models and release GPU memory."""
    await _run_blocking(cleanup_resources)
    return {"status": "resources cleaned"}


# Gradio UI mounted at root for interactive usage.
LOGGER.info("Mounting Gradio UI at '/'")
ui = create_ui()
app.mount("/", ui.app)
