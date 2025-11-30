"""Central configuration primitives for the DNI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import DEFAULT_QWEN_MODEL_PATH


@dataclass(slots=True)
class PipelineSettings:
    """Grouped configuration used by the pipeline orchestration layer."""

    ocr_max_side: int = 1600
    vlm_size: int = 512
    max_new_tokens: int = 128
    model_path: str = str(DEFAULT_QWEN_MODEL_PATH)
    enable_card_crop: bool = True
    processed_image_dir: Optional[Path] = None

    def copy_with(self, **overrides) -> "PipelineSettings":
        """Return a shallow copy overriding any provided fields."""
        data = {
            "ocr_max_side": self.ocr_max_side,
            "vlm_size": self.vlm_size,
            "max_new_tokens": self.max_new_tokens,
            "model_path": self.model_path,
            "enable_card_crop": self.enable_card_crop,
            "processed_image_dir": self.processed_image_dir,
        }
        data.update(overrides)
        return PipelineSettings(**data)


__all__ = ["PipelineSettings"]
