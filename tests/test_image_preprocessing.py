"""Unit tests for image preprocessing helpers."""
from __future__ import annotations

from PIL import Image, ImageDraw

from dni_pipeline.core.preprocessing import (
    _document_bbox_and_coverage,
    DOC_WARP_TARGET_SIZE,
    normalize_document_view,
    preprocess_for_ocr,
    preprocess_for_vlm,
)


def test_preprocess_for_ocr_limits_long_side_and_keeps_aspect_ratio() -> None:
    """The OCR view must be resized so its longest side matches the configured max."""
    image = Image.new("RGB", (4000, 1000), (200, 200, 200))
    processed = preprocess_for_ocr(image, max_side=1000)
    assert processed.size == (1000, 250)


def test_preprocess_for_vlm_returns_square_canvas_with_padding() -> None:
    """The VLM view should be padded to a square canvas with white borders."""
    image = Image.new("RGB", (100, 200), (10, 20, 30))
    canvas = preprocess_for_vlm(image, target_size=256)
    assert canvas.size == (256, 256)
    assert canvas.getpixel((0, 0)) == (255, 255, 255)
    assert canvas.getpixel((128, 128)) == (10, 20, 30)


def test_document_bbox_and_coverage_detects_dark_region() -> None:
    """The luminance-based heuristic should find the document footprint."""
    image = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 150, 150], fill=(20, 20, 20))
    bbox, coverage = _document_bbox_and_coverage(image)
    assert bbox is not None
    left, top, right, bottom = bbox
    assert left <= 50 <= right
    assert top <= 50 <= bottom
    assert 0.2 <= coverage <= 0.3


def test_normalize_document_view_detects_and_warps_document() -> None:
    """The normalization step should warp a skewed DNI into the canonical canvas."""
    base = Image.new("RGB", (800, 600), (30, 100, 150))
    draw = ImageDraw.Draw(base)
    polygon = [(100, 120), (640, 90), (700, 520), (130, 540)]
    draw.polygon(polygon, fill=(240, 240, 240))
    result = normalize_document_view(base)
    assert result.perspective_applied is True
    assert result.quad is not None
    assert result.image.size == DOC_WARP_TARGET_SIZE
    assert result.quality["focus"] >= 0.0
