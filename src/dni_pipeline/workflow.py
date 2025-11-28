"""High-level orchestration helpers for processing DNI images."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .image_preprocessing import prepare_images
from .logging_service import logging_service
from .ocr_doctr import build_ocr_block, run_doctr_ocr, sort_ocr_items, unload_model as unload_ocr
from .postprocess import normalize_and_validate, parse_model_output
from .vlm_qwen import run_qwen_vlm, unload_model as unload_vlm

LOGGER = logging_service.get_logger("dni_pipeline.workflow")
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def iter_image_paths(path: Path) -> Iterable[Path]:
    """Yield image files from a directory, sorted for reproducibility."""
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            yield item


def process_image(
    image_path: Path,
    *,
    ocr_max_side: int,
    vlm_size: int,
    max_new_tokens: int,
    model_path: str,
) -> Dict[str, Optional[str]]:
    """Process a single DNI image end-to-end and return the normalised record."""
    LOGGER.info("Processing %s", image_path)
    try:
        with logging_service.log_stage(
            "prepare_images",
            logger=LOGGER,
            extra={"image": image_path.name},
        ):
            ocr_image, vlm_image, cached_ocr_items = prepare_images(
                image_path, ocr_max_side, vlm_size
            )
            LOGGER.debug(
                "Image variants ready: OCR size %sx%s, VLM size %sx%s",
                *ocr_image.size,
                *vlm_image.size,
            )
    except Exception as exc:
        LOGGER.error("Failed to load image %s: %s", image_path, exc)
        return normalize_and_validate(None)

    ocr_block = "[OCR]\n[/OCR]"
    try:
        with logging_service.log_stage(
            "docTR_OCR",
            logger=LOGGER,
            extra={"image": image_path.name},
        ):
            if cached_ocr_items is not None:
                LOGGER.debug(
                    "Reusing cached OCR tokens from orientation scoring: %s items",
                    len(cached_ocr_items),
                )
                ocr_items = cached_ocr_items
            else:
                ocr_items = run_doctr_ocr(ocr_image)
            ordered_items = sort_ocr_items(ocr_items)
            LOGGER.debug("Sorted %s OCR tokens", len(ordered_items))
            ocr_block = build_ocr_block(ordered_items)
            LOGGER.debug("OCR block content:\n%s", ocr_block)
    except Exception as exc:
        LOGGER.warning("docTR OCR failed for %s: %s", image_path, exc)

    raw_model_output = ""
    try:
        with logging_service.log_stage(
            "qwen_inference",
            logger=LOGGER,
            extra={"image": image_path.name, "max_new_tokens": max_new_tokens},
        ):
            raw_model_output = run_qwen_vlm(
                vlm_image,
                ocr_block,
                max_new_tokens=max_new_tokens,
                model_id=model_path,
            )
            LOGGER.debug("Raw model output: %s", raw_model_output)
    except Exception as exc:
        LOGGER.error("Qwen inference failed for %s: %s", image_path, exc)
        return normalize_and_validate(None)

    with logging_service.log_stage(
        "postprocess",
        logger=LOGGER,
        extra={"image": image_path.name},
    ):
        parsed_data = parse_model_output(raw_model_output)
        if "raw_output" in parsed_data:
            LOGGER.warning(
                "Model output for %s was not valid JSON. Raw output snippet: %s",
                image_path,
                parsed_data["raw_output"],
            )
        return normalize_and_validate(parsed_data)


def save_json(record: Dict[str, Optional[str]], output_path: Path) -> None:
    """Persist the record to disk as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def process_directory(input_dir: Path) -> List[Path]:
    """Return a list of image paths detected in the provided directory."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    return list(iter_image_paths(input_dir))


def cleanup_resources() -> None:
    """Force release of loaded models and GPU memory."""
    LOGGER.info("Cleaning up pipeline resources...")
    unload_ocr()
    unload_vlm()
