"""Image loading and preprocessing utilities for the DNI pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageStat

from .logging_service import logging_service
from .ocr_doctr import OcrItem, run_doctr_ocr

LOGGER = logging_service.get_logger(__name__)
ORIENTATION_ANGLES = (0, 90, 180, 270)
ORIENTATION_TOKEN_CONFIDENCE_THRESHOLD = 0.3
ORIENTATION_LOW_CONFIDENCE_TOKEN_COUNT = 10
DOCUMENT_LUMINANCE_THRESHOLD = 235
DOCUMENT_MIN_COVERAGE_RATIO = 0.15
DOCUMENT_AGGRESSIVE_CROP_THRESHOLD = 0.5
DOCUMENT_COMPONENT_MIN_RATIO = 0.08
OCR_CONTRAST_FACTOR = 1.15
DOC_WARP_TARGET_SIZE = (1400, 900)
DOCUMENT_MIN_CONTOUR_AREA_RATIO = 0.08
DOCUMENT_MIN_CONTOUR_AREA_FALLBACK = 0.04
DOCUMENT_WARP_MIN_RATIO = 0.1
FOCUS_WARNING_THRESHOLD = 80.0
MEAN_LUMINANCE_MIN = 60.0
MEAN_LUMINANCE_MAX = 210.0
VLM_ZOOM_MAX_COVERAGE = 0.98
@dataclass
class AutoOrientationResult:
    """Container describing the best orientation after OCR-based scoring."""

    image: Image.Image
    preprocessed_image: Optional[Image.Image]
    ocr_items: Optional[List[OcrItem]]
    angle: int
    confidence: str
    token_counts: Dict[int, int]


@dataclass
class PreparedImages:
    """Bundle with all preprocessed variants and their metadata."""

    ocr_image: Image.Image
    vlm_image: Image.Image
    vlm_zoom_image: Optional[Image.Image]
    ocr_items: Optional[List[OcrItem]]
    metadata: Dict[str, Any]


@dataclass
class DocumentNormalizationResult:
    """Result of document detection and normalization."""

    image: Image.Image
    zoom_image: Image.Image
    quad: Optional[List[List[float]]]
    perspective_applied: bool
    zoom_factor: float
    quality: Dict[str, float]


def _count_useful_tokens(ocr_items: Sequence[OcrItem]) -> int:
    """Count tokens above the confidence threshold used for orientation scoring."""
    return sum(
        1
        for item in ocr_items
        if item.confidence >= ORIENTATION_TOKEN_CONFIDENCE_THRESHOLD
    )


def _orientation_confidence(token_counts: Dict[int, int]) -> str:
    """Return 'low' if every orientation yielded very few tokens."""
    max_tokens = max(token_counts.values(), default=0)
    return "low" if max_tokens < ORIENTATION_LOW_CONFIDENCE_TOKEN_COUNT else "high"


def _ordered_orientation_angles(aspect_ratio: float) -> Tuple[int, int, int, int]:
    """Prioritise orientations based on aspect ratio to minimise unnecessary OCR passes."""
    if aspect_ratio == 0:
        return ORIENTATION_ANGLES
    if 0.95 <= aspect_ratio <= 1.05:
        return ORIENTATION_ANGLES
    if aspect_ratio > 1:
        return (0, 180, 90, 270)
    return (90, 270, 0, 180)


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL image in RGB to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _cv_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _resize_canvas(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize while preserving aspect ratio and pad to the requested canvas size."""
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        return image.copy()
    resized = image.copy()
    scale = min(target_w / resized.width, target_h / resized.height)
    if scale <= 0:
        return image.copy()
    new_size = (max(1, int(resized.width * scale)), max(1, int(resized.height * scale)))
    resized = resized.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    offset_x = (target_w - new_size[0]) // 2
    offset_y = (target_h - new_size[1]) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas




def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return a consistently ordered set of quad points (tl, tr, br, bl)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _document_detection_variants(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Return different preprocessing variants to help document contour detection."""
    variants: List[Tuple[str, Image.Image]] = [("original", image)]
    try:
        contrast_boost = ImageEnhance.Contrast(image).enhance(1.35)
        variants.append(("contrast", contrast_boost))
    except Exception:  # pragma: no cover - defensive guard
        pass
    try:
        inverted = ImageOps.invert(image)
        variants.append(("inverted", inverted))
    except Exception:  # pragma: no cover - defensive guard
        pass
    try:
        clahe_variant = _enhance_document_image(image)
        variants.append(("clahe", clahe_variant))
    except Exception:  # pragma: no cover - defensive guard
        pass
    return variants


def _detect_polygon_on_variant(image: Image.Image) -> Optional[np.ndarray]:
    """Detect the largest quadrilateral contour on a specific image variant."""
    cv_image = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_area = image.width * image.height
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        min_ratio = (
            DOCUMENT_MIN_CONTOUR_AREA_RATIO
            if area >= image_area * DOCUMENT_MIN_CONTOUR_AREA_RATIO
            else DOCUMENT_MIN_CONTOUR_AREA_FALLBACK
        )
        if area < image_area * min_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            LOGGER.debug("Detected document contour with area %.2f", area)
            return approx.reshape(4, 2).astype("float32")
    if sorted_contours:
        contour = sorted_contours[0]
        area = cv2.contourArea(contour)
        if area >= image_area * DOCUMENT_MIN_CONTOUR_AREA_FALLBACK:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            LOGGER.debug("Using min-area rectangle fallback with area %.2f", area)
            return box.astype("float32")
    return None


def _detect_document_polygon(image: Image.Image) -> Optional[np.ndarray]:
    """Detect the largest quadrilateral contour resembling the DNI."""
    for variant_name, variant in _document_detection_variants(image):
        polygon = _detect_polygon_on_variant(variant)
        if polygon is not None:
            if variant_name != "original":
                LOGGER.debug("Document contour detected using %s variant", variant_name)
            return polygon
    return None


def _warp_document(
    image: Image.Image,
    quad: np.ndarray,
    target_size: Tuple[int, int] = DOC_WARP_TARGET_SIZE,
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Apply a perspective transform to map the document to a canonical rectangle."""
    try:
        rect = _order_quad_points(quad)
        width, height = target_size
        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, destination)
        warped = cv2.warpPerspective(
            _pil_to_cv(image),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return _cv_to_pil(warped), matrix
    except cv2.error as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Perspective transform failed: %s", exc)
        return None


def _apply_unsharp_mask(
    image: Image.Image,
    radius: float = 1.2,
    amount: float = 0.6,
) -> Image.Image:
    """Apply a light unsharp mask to improve local contrast before warping."""
    cv_image = _pil_to_cv(image)
    sigma = max(0.1, float(radius))
    blurred = cv2.GaussianBlur(cv_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(cv_image, 1.0 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype("uint8")
    return _cv_to_pil(sharpened)


def _crop_polygon_with_padding(
    image: Image.Image,
    quad: np.ndarray,
    padding: float = 0.06,
) -> Tuple[Image.Image, np.ndarray]:
    """Crop the detected document with padding and return the adjusted quad."""
    width, height = image.size
    x_min = float(np.min(quad[:, 0]))
    y_min = float(np.min(quad[:, 1]))
    x_max = float(np.max(quad[:, 0]))
    y_max = float(np.max(quad[:, 1]))
    pad_x = max(2.0, (x_max - x_min) * padding)
    pad_y = max(2.0, (y_max - y_min) * padding)
    x0 = int(max(0, math.floor(x_min - pad_x)))
    y0 = int(max(0, math.floor(y_min - pad_y)))
    x1 = int(min(width, math.ceil(x_max + pad_x)))
    y1 = int(min(height, math.ceil(y_max + pad_y)))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return image.copy(), quad.copy()
    cropped = image.crop((x0, y0, x1, y1))
    adjusted = quad.copy()
    adjusted[:, 0] -= x0
    adjusted[:, 1] -= y0
    return cropped, adjusted


def _enhance_document_image(image: Image.Image) -> Image.Image:
    """Apply CLAHE and mild denoise to improve OCR readability after warping."""
    cv_image = _pil_to_cv(image)
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    return _cv_to_pil(denoised)


def _focus_metric(image: Image.Image) -> float:
    """Return the variance of the Laplacian as a focus metric."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_document_view(
    image: Image.Image,
    target_size: Tuple[int, int] = DOC_WARP_TARGET_SIZE,
) -> DocumentNormalizationResult:
    """Detect, rectify, and enhance the DNI prior to OCR/VLM processing."""
    polygon = _detect_document_polygon(image)
    polygon_area_ratio = 0.0
    if polygon is not None:
        image_area = image.width * image.height
        if image_area > 0:
            polygon_area_ratio = float(
                cv2.contourArea(polygon.astype("float32"))
            ) / float(image_area)
        if polygon_area_ratio < DOCUMENT_WARP_MIN_RATIO:
            LOGGER.debug(
                "Detected polygon area ratio %.3f below threshold; skipping warp",
                polygon_area_ratio,
            )
            polygon = None
    perspective_applied = False
    quad_serialised: Optional[List[List[float]]] = None
    working = image
    zoom_factor = 1.0
    bbox_for_crop: Optional[Tuple[int, int, int, int]] = None
    if polygon is not None:
        polygon_serialised = polygon.astype("float32").tolist()
        warp_source = image
        adjusted_quad = polygon.copy()
        try:
            warp_source, adjusted_quad = _crop_polygon_with_padding(image, polygon)
        except Exception:  # pragma: no cover - defensive fallback
            warp_source = image
            adjusted_quad = polygon.copy()
        warp_source = _apply_unsharp_mask(warp_source)
        warp_result = _warp_document(warp_source, adjusted_quad, target_size)
        if warp_result is not None:
            warped, matrix = warp_result
            working = warped
            perspective_applied = True
            quad_serialised = polygon_serialised
    if not perspective_applied:
        bbox_for_crop, luminance_coverage = _document_bbox_and_coverage(image)
        if luminance_coverage < DOCUMENT_AGGRESSIVE_CROP_THRESHOLD:
            component_bbox = _largest_component_bbox(image)
            if component_bbox is not None:
                LOGGER.debug(
                    "Using connected-component bbox fallback due to low coverage %.3f",
                    luminance_coverage,
                )
                bbox_for_crop = component_bbox
        if bbox_for_crop is not None:
            x0, y0, x1, y1 = bbox_for_crop
            margin_x = int((x1 - x0) * 0.1)
            margin_y = int((y1 - y0) * 0.1)
            crop_box = (
                max(0, x0 - margin_x),
                max(0, y0 - margin_y),
                min(image.width, x1 + margin_x),
                min(image.height, y1 + margin_y),
            )
            working = image.crop(crop_box)
        working = _apply_unsharp_mask(working)
        working = _resize_canvas(working, target_size)
    enhanced = _enhance_document_image(working)
    base_focus = _focus_metric(enhanced)
    zoom_factor = 1.0
    if perspective_applied:
        zoom_factor = 1.6
    elif bbox_for_crop is not None:
        zoom_factor = 1.2
    if base_focus < FOCUS_WARNING_THRESHOLD:
        focus_ratio = FOCUS_WARNING_THRESHOLD / max(base_focus, 1.0)
        zoom_factor = max(zoom_factor, min(2.5, zoom_factor * math.sqrt(focus_ratio)))
    zoom_image = enhanced
    if zoom_factor > 1.0:
        zoom_width = max(1, int(enhanced.width * zoom_factor))
        zoom_height = max(1, int(enhanced.height * zoom_factor))
        zoom_image = enhanced.resize((zoom_width, zoom_height), Image.Resampling.LANCZOS)

    focus_value = _focus_metric(zoom_image)
    luminance = float(ImageStat.Stat(enhanced.convert("L")).mean[0])
    quality = {
        "focus": round(focus_value, 2),
        "mean_luminance": round(luminance, 2),
    }
    return DocumentNormalizationResult(
        image=enhanced,
        zoom_image=zoom_image,
        quad=quad_serialised,
        perspective_applied=perspective_applied,
        zoom_factor=zoom_factor,
        quality=quality,
    )


def _document_bbox_and_coverage(
    image: Image.Image,
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """Approximate the document bounding box and coverage using a luminance mask."""
    width, height = image.size
    if width == 0 or height == 0:
        return None, 0.0
    gray = image.convert("L")
    mask = gray.point(
        lambda value: 0 if value > DOCUMENT_LUMINANCE_THRESHOLD else 255
    )
    bbox = mask.getbbox()
    total_pixels = width * height
    pixel_sum = ImageStat.Stat(mask).sum[0]
    non_white_pixels = pixel_sum / 255.0
    coverage_ratio = max(0.0, min(1.0, non_white_pixels / total_pixels))
    return bbox, coverage_ratio


def _largest_component_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of the largest connected component as a fallback crop."""
    cv_image = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Ensure the document appears white (255) so contours outline it.
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.width * image.height
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * DOCUMENT_COMPONENT_MIN_RATIO:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if area > best_area:
            best_area = area
            best_bbox = (x, y, x + w, y + h)
    return best_bbox


def _format_bbox(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[List[int]]:
    """Return the bbox as a JSON-friendly list."""
    if bbox is None:
        return None
    return [int(coord) for coord in bbox]


def _size_dict(image: Image.Image) -> Dict[str, int]:
    """Represent an image size as a dict to ease downstream serialisation."""
    return {"width": image.width, "height": image.height}


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
    ocr_image = enhancer.enhance(OCR_CONTRAST_FACTOR)
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
) -> AutoOrientationResult:
    """
    Try the different right-angle rotations and pick the one that yields the richest OCR.

    The rotations are prioritised with a light aspect ratio heuristic, but every candidate
    is scored via docTR until a confident orientation is found. The preprocessed image and
    OCR tokens from the winning orientation are cached so the caller can reuse them.
    """
    width, height = image.size
    if width == 0 or height == 0:
        LOGGER.warning("Image has invalid dimensions; skipping auto-orientation")
        return AutoOrientationResult(
            image=image,
            preprocessed_image=None,
            ocr_items=None,
            angle=0,
            confidence="low",
            token_counts={angle: 0 for angle in ORIENTATION_ANGLES},
        )

    aspect_ratio = width / height
    candidate_angles = _ordered_orientation_angles(aspect_ratio)
    LOGGER.info(
        "Evaluating orientations %s based on aspect ratio %.2f",
        candidate_angles,
        aspect_ratio,
    )
    best_image = image
    best_angle = 0
    best_preprocessed: Optional[Image.Image] = None
    best_items: Optional[List[OcrItem]] = None
    best_token_count = 0
    token_counts: Dict[int, int] = {angle: 0 for angle in ORIENTATION_ANGLES}

    evaluated_angles = 0
    for angle in candidate_angles:
        candidate = image if angle == 0 else image.rotate(angle, expand=True)
        LOGGER.debug("Scoring orientation %d degrees (size=%sx%s)", angle, *candidate.size)
        prepped = preprocess_for_ocr(candidate, max_side=ocr_max_side)
        try:
            items = run_doctr_ocr(prepped)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("docTR OCR failed during orientation %d°: %s", angle, exc)
            continue

        useful_tokens = _count_useful_tokens(items)
        token_counts[angle] = useful_tokens
        LOGGER.debug(
            "Orientation %d° yielded %d tokens >= %.2f confidence",
            angle,
            useful_tokens,
            ORIENTATION_TOKEN_CONFIDENCE_THRESHOLD,
        )

        if useful_tokens > best_token_count:
            best_image = candidate
            best_angle = angle
            best_preprocessed = prepped
            best_items = items
            best_token_count = useful_tokens

        evaluated_angles += 1

    if best_preprocessed is None:
        LOGGER.warning("Falling back to heuristic orientation; docTR scoring failed.")
        orientation_confidence = _orientation_confidence(token_counts)
        return AutoOrientationResult(
            image=best_image,
            preprocessed_image=None,
            ocr_items=None,
            angle=best_angle,
            confidence=orientation_confidence,
            token_counts=token_counts,
        )

    orientation_confidence = _orientation_confidence(token_counts)
    LOGGER.info(
        "Selected orientation %d° with %d useful tokens (confidence=%s)",
        best_angle,
        best_token_count,
        orientation_confidence,
    )
    return AutoOrientationResult(
        image=best_image,
        preprocessed_image=best_preprocessed,
        ocr_items=best_items,
        angle=best_angle,
        confidence=orientation_confidence,
        token_counts=token_counts,
    )


def prepare_images(
    path: Path | str,
    ocr_max_side: int = 1600,
    vlm_target_size: int = 512,
    enable_card_crop: bool = False,
) -> PreparedImages:
    """Load the image, correct its orientation, and generate the OCR/VLM variants."""
    LOGGER.info(
        "Preparing images for OCR and VLM (ocr_max_side=%s, vlm_target_size=%s): %s",
        ocr_max_side,
        vlm_target_size,
        path,
    )
    base_image = load_image(path)
    normalization = normalize_document_view(base_image)
    normalized_image = normalization.image
    orientation_source = normalization.zoom_image
    _, normalized_coverage = _document_bbox_and_coverage(normalized_image)
    adjusted_zoom_factor = normalization.zoom_factor
    if normalized_coverage is not None and normalized_coverage >= VLM_ZOOM_MAX_COVERAGE:
        adjusted_zoom_factor = min(adjusted_zoom_factor, 1.2)

    effective_ocr_max_side = int(ocr_max_side * max(1.0, adjusted_zoom_factor))
    effective_ocr_max_side = max(ocr_max_side, min(effective_ocr_max_side, 2400))
    orientation = auto_orient_with_ocr(orientation_source, effective_ocr_max_side)
    oriented_zoom_image = orientation.image
    oriented_vlm_source = normalized_image
    if orientation.angle:
        oriented_vlm_source = oriented_vlm_source.rotate(orientation.angle, expand=True)

    ocr_image = (
        orientation.preprocessed_image
        if orientation.preprocessed_image is not None
        else preprocess_for_ocr(oriented_zoom_image, effective_ocr_max_side)
    )
    vlm_image = preprocess_for_vlm(oriented_vlm_source, vlm_target_size)
    document_bbox, coverage_ratio = _document_bbox_and_coverage(oriented_vlm_source)
    warnings: List[str] = []
    if not normalization.perspective_applied and document_bbox is None:
        warnings.append("document_not_detected")
    elif coverage_ratio < DOCUMENT_MIN_COVERAGE_RATIO:
        warnings.append("document_area_small")
    focus_value = normalization.quality.get("focus", 0.0)
    mean_luminance = normalization.quality.get("mean_luminance", 0.0)
    if focus_value < FOCUS_WARNING_THRESHOLD:
        warnings.append("low_focus")
    if not (MEAN_LUMINANCE_MIN <= mean_luminance <= MEAN_LUMINANCE_MAX):
        warnings.append("poor_lighting")

    vlm_zoom_image = None
    if (
        enable_card_crop
        and document_bbox is not None
        and coverage_ratio < VLM_ZOOM_MAX_COVERAGE
    ):
        x0, y0, x1, y1 = document_bbox
        if x1 - x0 > 0 and y1 - y0 > 0:
            zoom_crop = oriented_vlm_source.crop((x0, y0, x1, y1))
            if zoom_crop.width > 0 and zoom_crop.height > 0:
                vlm_zoom_image = preprocess_for_vlm(zoom_crop, vlm_target_size)

    metadata: Dict[str, Any] = {
        "source_path": str(Path(path)),
        "input_size": _size_dict(base_image),
        "orientation": {
            "angle": orientation.angle,
            "confidence": orientation.confidence,
            "token_counts": {str(angle): count for angle, count in orientation.token_counts.items()},
        },
        "document": {
            "coverage_ratio": round(coverage_ratio, 4),
            "bbox": _format_bbox(document_bbox),
            "quad": normalization.quad,
            "perspective_correction": normalization.perspective_applied,
            "zoom_factor": normalization.zoom_factor,
        },
        "quality": normalization.quality,
        "warnings": warnings,
        "views": {
            "ocr": {
                "size": _size_dict(ocr_image),
                "max_side": effective_ocr_max_side,
            },
            "vlm": {
                "size": _size_dict(vlm_image),
                "target_size": vlm_target_size,
            },
            "vlm_zoom": _size_dict(vlm_zoom_image) if vlm_zoom_image else None,
        },
        "params": {
            "ocr_max_side": ocr_max_side,
            "ocr_effective_max_side": effective_ocr_max_side,
            "vlm_target_size": vlm_target_size,
            "enable_card_crop": enable_card_crop,
        },
    }
    LOGGER.debug("Preprocessing metadata: %s", metadata)
    return PreparedImages(
        ocr_image=ocr_image,
        vlm_image=vlm_image,
        vlm_zoom_image=vlm_zoom_image,
        ocr_items=orientation.ocr_items,
        metadata=metadata,
    )
