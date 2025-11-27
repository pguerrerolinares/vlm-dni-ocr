"""Core package for the DNI OCR + VLM pipeline."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_QWEN_SEARCH_PATHS = [
    PROJECT_ROOT / "Qwen3-VL-8B-Instruct",
    DEFAULT_MODELS_DIR / "Qwen3-VL-8B-Instruct",
]
_ENV_QWEN_PATH = os.getenv("DNI_PIPELINE_QWEN_MODEL_PATH")
if _ENV_QWEN_PATH:
    env_path = Path(_ENV_QWEN_PATH).expanduser()
    DEFAULT_QWEN_SEARCH_PATHS.insert(0, env_path)
else:
    env_path = None

DEFAULT_QWEN_MODEL_PATH = next(
    (path for path in DEFAULT_QWEN_SEARCH_PATHS if path.exists()),
    env_path if env_path is not None else PROJECT_ROOT / "Qwen3-VL-8B-Instruct",
)

__version__ = "0.1.0"

__all__ = [
    "PACKAGE_ROOT",
    "SRC_ROOT",
    "PROJECT_ROOT",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_QWEN_SEARCH_PATHS",
    "DEFAULT_QWEN_MODEL_PATH",
    "__version__",
]
