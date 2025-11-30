"""High-level orchestration helpers for processing DNI images."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .image_preprocessing import prepare_images, preprocess_for_ocr
from .logging_service import logging_service
from .ocr_doctr import (
    OcrTextLine,
    build_ocr_lines,
    render_ocr_block,
    run_doctr_ocr,
    sort_ocr_items,
    unload_model as unload_ocr,
)
from .postprocess import (
    build_cleaner_prompt,
    finalize_cleaner_result,
    parse_cleaner_output,
    parse_model_output,
)
from .vlm_qwen import (
    run_qwen_cleaner,
    run_qwen_vlm,
    unload_model as unload_vlm,
)

LOGGER = logging_service.get_logger("dni_pipeline.workflow")
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LOW_FOCUS_REJECT_THRESHOLD = 25.0


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
    enable_card_crop: bool = True,
    processed_image_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process a single DNI image end-to-end and return the normalised record."""
    LOGGER.info("Processing %s", image_path)
    try:
        with logging_service.log_stage(
            "prepare_images",
            logger=LOGGER,
            extra={"image": image_path.name},
        ):
            prepared = prepare_images(
                image_path, ocr_max_side, vlm_size, enable_card_crop=enable_card_crop
            )
            ocr_image = prepared.ocr_image
            vlm_image = prepared.vlm_image
            cached_ocr_items = prepared.ocr_items
            LOGGER.debug(
                "Preprocess metadata for %s: %s",
                image_path.name,
                prepared.metadata,
            )
            LOGGER.debug(
                "Image variants ready: OCR size %sx%s, VLM size %sx%s",
                *ocr_image.size,
                *vlm_image.size,
            )
            if processed_image_dir is not None:
                processed_image_dir.mkdir(parents=True, exist_ok=True)
                processed_path = processed_image_dir / f"{image_path.stem}_processed.png"
                vlm_image.save(processed_path)
                LOGGER.info("Saved processed VLM image for %s at %s", image_path.name, processed_path)
                if prepared.vlm_zoom_image is not None:
                    zoom_path = processed_image_dir / f"{image_path.stem}_vlm_zoom.png"
                    prepared.vlm_zoom_image.save(zoom_path)
                    LOGGER.info("Saved VLM zoom image for %s at %s", image_path.name, zoom_path)
    except Exception as exc:
        LOGGER.error("Failed to load image %s: %s", image_path, exc)
        return {"raw_output": f"error: {exc}"}

    quality = prepared.metadata.get("quality", {})
    warnings = prepared.metadata.get("warnings", [])
    focus_value = float(quality.get("focus", 0.0))
    if "low_focus" in warnings or focus_value < LOW_FOCUS_REJECT_THRESHOLD:
        LOGGER.warning(
            "Focus %.2f too low for %s.",
            focus_value,
            image_path.name,
        )

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
            ocr_lines: List[OcrTextLine] = build_ocr_lines(ordered_items)
            LOGGER.debug("Grouped OCR tokens into %s lines", len(ocr_lines))        
            ocr_block = render_ocr_block(ocr_lines)
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
        return {"raw_output": f"error: {exc}"}

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
            parsed_data["trace"] = {
                "raw_ocr": ocr_block,
                "vlm_raw_output": raw_model_output,
            }
            return parsed_data

        cleaner_prompt = build_cleaner_prompt(ocr_block, parsed_data)
        LOGGER.debug("Cleaner prompt generated (%s chars)", len(cleaner_prompt))

        cleaner_raw_output = ""
        cleaner_data: Dict[str, Any] = {"raw_output": ""}
        cleaner_max_tokens = min(2048, max(512, max_new_tokens * 4))

        for attempt in range(2):
            prompt_to_use = cleaner_prompt
            if attempt == 1:
                prompt_to_use = (
                    cleaner_prompt
                    + "\n\nThe previous response was truncated. "
                    "Output the complete JSON again from the start, with no explanations."
                )
            try:
                cleaner_raw_output = run_qwen_cleaner(
                    clean_prompt=prompt_to_use,
                    max_new_tokens=cleaner_max_tokens,
                    model_id=model_path,
                )
                LOGGER.debug("Cleaner raw output (attempt %s): %s", attempt + 1, cleaner_raw_output)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Cleaner LLM failed for %s on attempt %s: %s", image_path, attempt + 1, exc)
                cleaner_raw_output = f"cleaner_error: {exc}"
                break

            cleaner_data = parse_cleaner_output(cleaner_raw_output)
            if "raw_output" not in cleaner_data:
                break
            LOGGER.warning(
                "Cleaner output for %s attempt %s was not valid JSON; retrying...",
                image_path.name,
                attempt + 1,
            )

        final_data = finalize_cleaner_result(cleaner_data)
        trace = {
            "raw_ocr": ocr_block,
            "vlm_raw_output": raw_model_output,
            "cleaner_prompt": cleaner_prompt,
            "cleaner_raw_output": cleaner_raw_output,
        }
        if isinstance(final_data, dict):
            final_data["trace"] = trace
            manual_review = final_data.get("manual_review")
            if isinstance(manual_review, list) and manual_review:
                LOGGER.warning(
                    "Manual review required for %s: %s",
                    image_path.name,
                    ", ".join(manual_review),
                )
        return final_data


def save_json(record: Dict[str, Any], output_path: Path) -> None:
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
