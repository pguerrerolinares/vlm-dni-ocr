"""Gradio interface that wraps the DNI extraction workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr
import httpx
from gradio_client import utils as gradio_client_utils

from . import DEFAULT_QWEN_MODEL_PATH
from .logging_service import logging_service

LOGGER = logging_service.get_logger("dni_pipeline.ui")

# Work around Gradio 4.44.0 bug when converting JSON schema with boolean
# ``additionalProperties`` into Python type hints (fixed upstream in 4.44.1).
_ORIG_JSON_SCHEMA_CONVERTER = gradio_client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Dict[str, Any]" if schema else "Dict[str, None]"
    return _ORIG_JSON_SCHEMA_CONVERTER(schema, defs)


gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


RELATIVE_API_PATH = "/api/v1/extract"
DEFAULT_API_ENDPOINT = f"http://127.0.0.1:8000{RELATIVE_API_PATH}"
HTTP_TIMEOUT = 600.0


def _resolve_endpoint(api_endpoint: str, request: "gr.Request | None") -> str:
    """Return an absolute endpoint, inferring base URL from the current request when needed."""
    endpoint = (api_endpoint or "").strip()
    if not endpoint:
        endpoint = DEFAULT_API_ENDPOINT

    if endpoint.startswith(("http://", "https://")):
        return endpoint

    normalized_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    # Handle relative paths such as "/api/v1/extract" by borrowing the current request origin.
    if request is not None:
        try:
            return str(request.url.replace(path=normalized_path, query="", fragment=""))
        except Exception:  # pragma: no cover - fallback guard
            LOGGER.debug("Failed to build endpoint from request; falling back to default.")

    # Fall back to the default endpoint base when the request context is missing.
    default_url = httpx.URL(DEFAULT_API_ENDPOINT)
    return str(default_url.copy_with(path=normalized_path, query=None, fragment=None))


def extract_dni_ui(
    image_path: str,
    ocr_max_side: int = 1600,
    vlm_size: int = 512,
    max_new_tokens: int = 256,
    model_path: str = str(DEFAULT_QWEN_MODEL_PATH),
    api_endpoint: str = RELATIVE_API_PATH,
    request: "gr.Request | None" = None,
) -> str:
    """Callback used by Gradio to call the FastAPI extraction endpoint."""
    if not image_path:
        LOGGER.warning("No image path provided to Gradio callback")
        return "Error: no se recibió ninguna imagen."

    path = Path(image_path)
    if not path.exists():
        LOGGER.warning("File provided by Gradio no longer exists: %s", path)
        return "Error: el archivo temporal ya no está disponible."

    endpoint = _resolve_endpoint(api_endpoint, request)
    LOGGER.info("Gradio UI invoking API endpoint %s for %s", endpoint, path.name)
    try:
        with path.open("rb") as file_handle:
            files = {"file": (path.name, file_handle, "image/jpeg")}
            params = {
                "ocr_max_side": ocr_max_side,
                "vlm_size": vlm_size,
                "max_new_tokens": max_new_tokens,
                "model_path": model_path,
            }
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                response = client.post(endpoint, files=files, params=params)
        if response.status_code != 200:
            LOGGER.error(
                "API call failed (%s): %s", response.status_code, response.text[:200]
            )
            return f"Error {response.status_code}: {response.text}"
        return response.text
    except httpx.HTTPError as exc:
        LOGGER.exception("HTTP request to %s failed: %s", endpoint, exc)
        return (
            "Error: no se pudo contactar con el endpoint de extracción.\n"
            f"Detalles: {exc}"
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected error invoking API from Gradio: %s", exc)
        return f"Error inesperado: {exc}"


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="DNI Extraction Pipeline") as demo:
        gr.Markdown("# DNI Extraction Pipeline")
        gr.Markdown(
            "Sube una imagen de un DNI español para extraer un JSON estructurado. "
            "La extracción se realiza en local combinando docTR (OCR) y Qwen3-VL-8B."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Imagen del DNI")
                with gr.Accordion("Parámetros avanzados", open=False):
                    ocr_max_side = gr.Slider(
                        minimum=800,
                        maximum=2400,
                        value=1600,
                        step=100,
                        label="Tamaño máximo OCR",
                    )
                    vlm_size = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=32,
                        label="Tamaño entrada VLM",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=256,
                        step=16,
                        label="Max new tokens",
                    )
                    model_path = gr.Textbox(
                        value=str(DEFAULT_QWEN_MODEL_PATH),
                        label="Ruta/Repo del modelo Qwen",
                    )
                    api_endpoint = gr.Textbox(
                        value=RELATIVE_API_PATH,
                        label="Endpoint API (FastAPI)",
                    )

                submit_btn = gr.Button("Extraer datos", variant="primary")

            with gr.Column():
                json_output = gr.Textbox(label="Resultado JSON", lines=14)

        submit_btn.click(
            fn=extract_dni_ui,
            inputs=[
                image_input,
                ocr_max_side,
                vlm_size,
                max_new_tokens,
                model_path,
                api_endpoint,
            ],
            outputs=json_output,
            queue=False,
        )

        gr.Markdown(
            "**Nota:** La primera ejecución puede tardar más debido a la carga de modelos "
            "en memoria. Reutilizando el proceso se aprovechan los modelos ya inicializados."
        )

    # Gradio 4.44.0 expects Blocks to expose ``max_file_size`` when handling uploads.
    if not hasattr(demo, "max_file_size"):
        demo.max_file_size = None

    return demo
