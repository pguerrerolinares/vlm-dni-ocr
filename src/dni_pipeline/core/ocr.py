"""docTR OCR helpers used by the DNI pipeline."""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

from ..logging_service import logging_service

LOGGER = logging_service.get_logger(__name__)


@dataclass(frozen=True)
class OcrItem:
    """Single OCR token with its bounding box and confidence."""

    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float


@dataclass
class OcrTextLine:
    """OCR tokens grouped into natural reading lines."""

    y: float
    tokens: List[str]
    norm_tokens: List[str]
    x_centers: List[float]

    @property
    def text(self) -> str:
        """Return the original text representation for the line."""
        return " ".join(self.tokens).strip()

    @property
    def normalized_text(self) -> str:
        """Return the accent-less uppercase representation to ease matching."""
        joined = " ".join(token for token in self.norm_tokens if token)
        return joined.strip()

    @property
    def x_mean(self) -> float:
        """Average X position of the line (0-1 range)."""
        if not self.x_centers:
            return 0.5
        return float(sum(self.x_centers) / len(self.x_centers))


DEFAULT_OCR_MIN_CONFIDENCE = 0.3
LINE_Y_TOLERANCE = 0.012


def _strip_accents(text: str) -> str:
    """Remove accents while keeping ASCII-friendly characters."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _normalize_token(text: str) -> str:
    """Return an uppercase, accent-less token suited for keyword matching."""
    cleaned = text.strip()
    cleaned = cleaned.replace("\u00BA", "")
    cleaned = cleaned.replace("\u00B0", "")
    cleaned = cleaned.strip(":")
    cleaned = cleaned.strip(".,;")
    cleaned = cleaned.strip()
    cleaned = _strip_accents(cleaned)
    return cleaned.upper()



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
    min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
    y_tolerance: float = LINE_Y_TOLERANCE,
) -> str:
    """Build the `[OCR]` text block for prompt injection."""
    lines = build_ocr_lines(ocr_items, min_confidence=min_confidence, y_tolerance=y_tolerance)
    block_text = render_ocr_block(lines)
    LOGGER.info("Built OCR block with %s lines (threshold %.2f)", len(lines), min_confidence)
    return block_text


def build_ocr_lines(
    ocr_items: Iterable[OcrItem],
    min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
    y_tolerance: float = LINE_Y_TOLERANCE,
) -> List[OcrTextLine]:
    """Group tokens into text lines after applying the confidence filter."""
    items = list(ocr_items)
    lines: List[OcrTextLine] = []
    kept_tokens = 0
    for item in items:
        if item.confidence < min_confidence:
            continue
        text = item.text.strip()
        if not text:
            continue
        normalized = _normalize_token(text)
        if not normalized and not text:
            continue
        kept_tokens += 1
        y_coord = item.bbox[1]
        x_center = (item.bbox[0] + item.bbox[2]) / 2.0
        if lines and abs(y_coord - lines[-1].y) <= y_tolerance:
            lines[-1].tokens.append(text)
            lines[-1].norm_tokens.append(normalized)
            lines[-1].x_centers.append(x_center)
        else:
            lines.append(
                OcrTextLine(
                    y=y_coord,
                    tokens=[text],
                    norm_tokens=[normalized],
                    x_centers=[x_center],
                )
            )
    LOGGER.debug(
        "OCR tokens before filtering: %d - kept %d tokens >= %.2f confidence",
        len(items),
        kept_tokens,
        min_confidence,
    )
    return lines


def render_ocr_block(ocr_lines: Sequence[OcrTextLine]) -> str:
    """Render the `[OCR]` block from grouped lines."""
    line_texts = [line.text for line in ocr_lines if line.text]
    if not line_texts:
        return "[OCR]\n[/OCR]"
    content = "\n".join(line_texts)
    return f"[OCR]\n{content}\n[/OCR]"
