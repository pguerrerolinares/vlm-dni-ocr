"""docTR OCR helpers used by the DNI pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

from .logging_service import logging_service

LOGGER = logging_service.get_logger(__name__)


@dataclass(frozen=True)
class OcrItem:
    """Single OCR token with its bounding box and confidence."""

    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float


_OCR_MODEL = None


def load_doctr_model(det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn"):
    """Lazy-load the docTR OCR predictor and move it to the best available device."""
    global _OCR_MODEL
    if _OCR_MODEL is None:
        LOGGER.info("Loading docTR model (det=%s, reco=%s)", det_arch, reco_arch)
        os.environ.setdefault("DOCTR_MULTIPROCESSING_DISABLE", "TRUE")
        predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor.to(device)
        predictor.eval()
        _OCR_MODEL = predictor
        LOGGER.info("docTR model ready on %s", device)
    else:
        LOGGER.debug("Reusing cached docTR model")
    return _OCR_MODEL


def unload_model() -> None:
    """Free memory by releasing the cached docTR model."""
    global _OCR_MODEL
    if _OCR_MODEL is not None:
        LOGGER.info("Unloading docTR model...")
        del _OCR_MODEL
        _OCR_MODEL = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        LOGGER.info("docTR model unloaded")


def run_doctr_ocr(image: Image.Image) -> List[OcrItem]:
    """Run docTR OCR on a preprocessed image and return a flat list of OCR tokens."""
    LOGGER.info("Running docTR OCR on image size %sx%s", *image.size)
    model = load_doctr_model()
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    doc = DocumentFile.from_images([buffer.getvalue()])
    result = model(doc)
    items: List[OcrItem] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geometry = word.geometry
                    x_min, y_min = geometry[0]
                    x_max, y_max = geometry[1]
                    items.append(
                        OcrItem(
                            text=word.value,
                            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
                            confidence=float(word.confidence),
                        )
                    )
    LOGGER.info("docTR OCR extracted %s tokens", len(items))
    return items


def sort_ocr_items(ocr_items: Sequence[OcrItem]) -> List[OcrItem]:
    """Sort OCR tokens in reading order (top-to-bottom, then left-to-right)."""
    sorted_items = sorted(
        ocr_items,
        key=lambda item: (round(item.bbox[1], 3), item.bbox[0]),
    )
    LOGGER.debug("Sorted %s OCR tokens into reading order", len(sorted_items))
    return sorted_items


def build_ocr_block(
    ocr_items: Iterable[OcrItem],
    min_confidence: float = 0.6,
) -> str:
    """Build the `[OCR]` text block for prompt injection."""
    lines = []
    for item in ocr_items:
        if item.confidence < min_confidence:
            LOGGER.debug(
                "Skipping OCR token below confidence threshold %.2f: '%s' (%.2f)",
                min_confidence,
                item.text,
                item.confidence,
            )
            continue
        text = item.text.strip()
        if not text:
            continue
        lines.append(text)
    block = "\n".join(lines)
    block_text = f"[OCR]\n{block}\n[/OCR]" if block else "[OCR]\n[/OCR]"
    LOGGER.info("Built OCR block with %s lines (threshold %.2f)", len(lines), min_confidence)
    return block_text

