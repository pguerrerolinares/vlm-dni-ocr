"""Image loading and preprocessing utilities for the DNI pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps

from .logging_service import logging_service
from .ocr_doctr import OcrItem, run_doctr_ocr

LOGGER = logging_service.get_logger(__name__)
ORIENTATION_ANGLES = (0, 90, 180, 270)
ORIENTATION_CONFIDENCE_FOR_SCORE = 0.45
ORIENTATION_EARLY_EXIT_TOKENS = 25


def _score_orientation_items(ocr_items: Sequence[OcrItem]) -> Tuple[float, int]:
    """Return aggregate confidence score and count of useful tokens for an orientation."""
    filtered = [
        item for item in ocr_items if item.confidence >= ORIENTATION_CONFIDENCE_FOR_SCORE
    ]
    total_conf = sum(item.confidence for item in filtered)
    score = total_conf + 0.1 * len(filtered)
    return score, len(filtered)


def _ordered_orientation_angles(aspect_ratio: float) -> Tuple[int, int, int, int]:
    """Prioritise orientations based on aspect ratio to minimise unnecessary OCR passes."""
    if aspect_ratio == 0:
        return ORIENTATION_ANGLES
    if 0.95 <= aspect_ratio <= 1.05:
        return ORIENTATION_ANGLES
    if aspect_ratio > 1:
        return (0, 180, 90, 270)
    return (90, 270, 0, 180)


def load_image(path: Path | str) -> Image.Image:
    """Load an image from disk, fix EXIF orientation, and convert to RGB."""
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {img_path}")
    LOGGER.debug("Loading image from %s", img_path)
    with Image.open(img_path) as img:
        rgb_image = ImageOps.exif_transpose(img).convert("RGB")
    LOGGER.debug("Loaded image size: %sx%s", *rgb_image.size)
    return rgb_image


def preprocess_for_ocr(image: Image.Image, max_side: int = 1600) -> Image.Image:
    """Prepare the image for docTR OCR: limit resolution and enhance contrast slightly."""
    ocr_image = image.copy()
    width, height = ocr_image.size
    LOGGER.debug("Original image size before OCR preprocessing: %sx%s", width, height)
    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        LOGGER.info(
            "Resizing image for OCR: original %sx%s -> %sx%s (max_side=%s)",
            width,
            height,
            new_size[0],
            new_size[1],
            max_side,
        )
        ocr_image = ocr_image.resize(new_size, Image.Resampling.BILINEAR)
    enhancer = ImageEnhance.Contrast(ocr_image)
    ocr_image = enhancer.enhance(1.1)
    LOGGER.debug("OCR image ready with size: %sx%s", *ocr_image.size)
    return ocr_image


def preprocess_for_vlm(image: Image.Image, target_size: int = 512) -> Image.Image:
    """Prepare the image for the VLM: maintain aspect ratio, pad to square canvas."""
    vlm_image = image.copy()
    LOGGER.debug("Preparing VLM image with target size %s", target_size)
    vlm_image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - vlm_image.width) // 2
    offset_y = (target_size - vlm_image.height) // 2
    canvas.paste(vlm_image, (offset_x, offset_y))
    LOGGER.info(
        "Prepared VLM image: content size %sx%s padded to %sx%s",
        vlm_image.width,
        vlm_image.height,
        canvas.width,
        canvas.height,
    )
    return canvas


def auto_orient_with_ocr(
    image: Image.Image,
    ocr_max_side: int = 1600,
) -> Tuple[Image.Image, Optional[Image.Image], Optional[List[OcrItem]]]:
    """
    Try the different right-angle rotations and pick the one that yields the richest OCR.

    The rotations are prioritised with a light aspect ratio heuristic, but every candidate
    is scored via docTR until a confident orientation is found. The preprocessed image and
    OCR tokens from the winning orientation are cached so the caller can reuse them.
    """
    width, height = image.size
    if width == 0 or height == 0:
        LOGGER.warning("Image has invalid dimensions; skipping auto-orientation")
        return image, None, None

    aspect_ratio = width / height
    candidate_angles = _ordered_orientation_angles(aspect_ratio)
    LOGGER.info(
        "Evaluating orientations %s based on aspect ratio %.2f",
        candidate_angles,
        aspect_ratio,
    )
    best_image = image
    best_score = -1.0
    best_angle = 0
    best_preprocessed: Optional[Image.Image] = None
    best_items: Optional[List[OcrItem]] = None
    best_token_count = 0

    for idx, angle in enumerate(candidate_angles):
        if idx > 0 and best_items is not None and best_token_count >= ORIENTATION_EARLY_EXIT_TOKENS:
            LOGGER.info(
                "Already have %d high-confidence tokens; skipping remaining orientations",
                best_token_count,
            )
            break

        candidate = image if angle == 0 else image.rotate(angle, expand=True)
        LOGGER.debug("Scoring orientation %d degrees (size=%sx%s)", angle, *candidate.size)
        prepped = preprocess_for_ocr(candidate, max_side=ocr_max_side)
        try:
            items = run_doctr_ocr(prepped)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("docTR OCR failed during orientation %d°: %s", angle, exc)
            continue

        score, useful_tokens = _score_orientation_items(items)
        LOGGER.debug(
            "Orientation %d° scored %.2f with %d tokens >= %.2f confidence",
            angle,
            score,
            useful_tokens,
            ORIENTATION_CONFIDENCE_FOR_SCORE,
        )

        if score > best_score:
            best_score = score
            best_image = candidate
            best_angle = angle
            best_preprocessed = prepped
            best_items = items
            best_token_count = useful_tokens

        if useful_tokens >= ORIENTATION_EARLY_EXIT_TOKENS:
            LOGGER.info(
                "Orientation %d° reached %d high-confidence tokens; early exit",
                angle,
                useful_tokens,
            )
            break

    if best_preprocessed is None:
        LOGGER.warning("Falling back to heuristic orientation; docTR scoring failed.")
        return best_image, None, None

    LOGGER.info(
        "Selected orientation %d° with %.0f useful tokens (size=%sx%s)",
        best_angle,
        best_token_count,
        *best_image.size,
    )
    return best_image, best_preprocessed, best_items


def prepare_images(
    path: Path | str,
    ocr_max_side: int = 1600,
    vlm_target_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Optional[List[OcrItem]]]:
    """Convenience helper that loads an image and returns the OCR and VLM variants."""
    LOGGER.info(
        "Preparing images for OCR and VLM (ocr_max_side=%s, vlm_target_size=%s): %s",
        ocr_max_side,
        vlm_target_size,
        path,
    )
    base_image = load_image(path)
    oriented_image, cached_preprocessed, cached_items = auto_orient_with_ocr(
        base_image, ocr_max_side
    )

    if cached_preprocessed is not None:
        ocr_image = cached_preprocessed
    else:
        ocr_image = preprocess_for_ocr(oriented_image, ocr_max_side)
    vlm_image = preprocess_for_vlm(oriented_image, vlm_target_size)
    LOGGER.debug(
        "Prepared variants - OCR size: %sx%s, VLM canvas size: %sx%s",
        *ocr_image.size,
        *vlm_image.size,
    )
    return ocr_image, vlm_image, cached_items
