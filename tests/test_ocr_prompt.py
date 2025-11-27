"""Unit tests for OCR prompt block construction."""
from __future__ import annotations

from dni_pipeline.ocr_doctr import OcrItem, build_ocr_block


def test_build_ocr_block_filters_low_confidence_tokens() -> None:
    """Only tokens above the threshold must be included."""
    items = [
        OcrItem(text="HELLO", bbox=(0.0, 0.0, 0.1, 0.1), confidence=0.95),
        OcrItem(text="noise", bbox=(0.2, 0.2, 0.3, 0.3), confidence=0.2),
    ]
    ocr_block = build_ocr_block(items, min_confidence=0.7)
    assert "[OCR]" in ocr_block and "[/OCR]" in ocr_block
    assert "HELLO" in ocr_block
    assert "noise" not in ocr_block
