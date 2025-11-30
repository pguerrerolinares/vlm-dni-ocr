"""Interface to Qwen3-VL-8B-Instruct for multimodal extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import DEFAULT_QWEN_SEARCH_PATHS
from .logging_service import logging_service

LOGGER = logging_service.get_logger(__name__)

_QWEN_PROCESSOR = None
_QWEN_MODEL = None
_QWEN_CACHE_KEY: str | None = None


def _resolve_model_source(model_id: str) -> str:
    """Return a Hugging Face repo id or an absolute path if local files exist."""
    path_candidate = Path(model_id).expanduser()
    if path_candidate.exists():
        LOGGER.debug("Resolved local model path: %s", path_candidate)
        return str(path_candidate.resolve())
    for candidate in DEFAULT_QWEN_SEARCH_PATHS:
        if candidate.exists():
            LOGGER.debug(
                "Falling back to detected local Qwen model path: %s", candidate
            )
            return str(candidate.resolve())
    if "/" not in model_id and not Path(model_id).suffix:
        LOGGER.warning(
            "Model path '%s' not found locally; ensure the Qwen weights are available "
            "offline or provide a Hugging Face repo id.",
            model_id,
        )
    LOGGER.debug("Using Hugging Face repo id for model: %s", model_id)
    return model_id


def load_qwen_model(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> Tuple[AutoProcessor, Qwen3VLForConditionalGeneration]:
    """Lazy-load the Qwen processor and model with sensible defaults."""
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_CACHE_KEY
    model_source = _resolve_model_source(model_id)
    if _QWEN_MODEL is None or _QWEN_PROCESSOR is None or _QWEN_CACHE_KEY != model_source:
        LOGGER.info("Loading Qwen model from %s", model_source)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        # Qwen model relies on custom modeling code, hence trust_remote_code=True here.
        try:
            processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
            )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_source,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
        except OSError as exc:  # pragma: no cover - defensive logging for offline usage
            available_paths = ", ".join(str(p) for p in DEFAULT_QWEN_SEARCH_PATHS)
            raise RuntimeError(
                "Failed to load Qwen3-VL-8B-Instruct from "
                f"'{model_source}'. Ensure the model weights are available locally "
                f"(checked paths: {available_paths}) or provide a reachable Hugging "
                "Face repo id via --model-path."
            ) from exc
        model.to(device)
        model.eval()
        _QWEN_MODEL = model
        _QWEN_PROCESSOR = processor
        _QWEN_CACHE_KEY = model_source
        LOGGER.info("Qwen model ready on %s with dtype %s", device, torch_dtype)
    else:
        LOGGER.debug("Reusing cached Qwen model for %s", model_source)
    return _QWEN_PROCESSOR, _QWEN_MODEL


def unload_model() -> None:
    """Free memory by releasing the cached Qwen model and processor."""
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_CACHE_KEY
    if _QWEN_MODEL is not None:
        LOGGER.info("Unloading Qwen model...")
        del _QWEN_MODEL
        del _QWEN_PROCESSOR
        _QWEN_MODEL = None
        _QWEN_PROCESSOR = None
        _QWEN_CACHE_KEY = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        LOGGER.info("Qwen model unloaded")


def build_prompt_text(ocr_block: str) -> str:
    """Create the textual instructions passed alongside the image."""
    return (
        "Act as an expert data extraction assistant for Spanish National Identity Cards (DNI).\n"
        "Carefully examine the provided image first. You also receive a raw OCR transcript with high recall but noisy text.\n"
        "Follow these rules:\n"
        "1. Always prioritise what you read in the image over the OCR hints when they contradict each other.\n"
        "2. Never invent data. If a field cannot be confirmed with high confidence, output null.\n"
        "3. Return exactly one valid JSON object with the following keys and no extra commentary:\n"
        '{\n'
        '  "nombre": null,\n'
        '  "primer_apellido": null,\n'
        '  "segundo_apellido": null,\n'
        '  "dni": null,\n'
        '  "fecha_nacimiento": null,\n'
        '  "fecha_validez": null,\n'
        '  "sexo": null,\n'
        '  "nacionalidad": null\n'
        '}\n'
        "4. Dates must use DD/MM/AAAA. If unsure about a date, return null for that field.\n"
        "5. The DNI must be the 8-digit number plus the control letter. If incomplete or uncertain, return null.\n"
        "6. If a name or surname is ambiguous or partly missing, use null instead of guessing.\n"
        "Respond with JSON only.\n\n"
        "Here is the OCR transcript (may contain errors):\n"
        f"{ocr_block}"
    )


def build_conversation(prompt_text: str) -> List[Dict]:
    """Build the structured conversation for the chat template."""
    LOGGER.debug("Building conversation payload for Qwen input")
    return [
        {
            "role": "system",
            "content": "You extract structured data from Spanish DNI images with high reliability.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def run_qwen_vlm(
    vlm_image: Image.Image,
    ocr_block: str,
    max_new_tokens: int = 256,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> str:
    """Run Qwen3-VL-8B on the image plus OCR context and return the raw model output."""
    LOGGER.info("Running Qwen VLM inference (max_new_tokens=%s)", max_new_tokens)
    processor, model = load_qwen_model(model_id)
    prompt_text = build_prompt_text(ocr_block)
    LOGGER.debug("Prompt text length: %s characters", len(prompt_text))
    conversation = build_conversation(prompt_text)

    chat_template = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    LOGGER.debug("Chat template prepared with length %s characters", len(chat_template))
    inputs = processor(
        text=[chat_template],
        images=[vlm_image],
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
        LOGGER.debug("Moved inputs to CUDA device %s", model.device)

    prompt_length = inputs["input_ids"].shape[1]
    LOGGER.debug("Prompt token length: %s", prompt_length)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )[0].strip()
    LOGGER.info("Qwen generation completed, output length %s characters", len(generated_text))
    return generated_text


def run_qwen_cleaner(
    clean_prompt: str,
    max_new_tokens: int = 512,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> str:
    """Run Qwen3-VL-8B in text-only mode to clean and validate extracted fields."""
    LOGGER.info("Running Qwen cleaner (max_new_tokens=%s)", max_new_tokens)
    processor, model = load_qwen_model(model_id)
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a meticulous cleaner and validator for Spanish DNI data. "
                "Always follow the provided rules and return valid JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": clean_prompt},
            ],
        },
    ]
    chat_template = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    LOGGER.debug("Cleaner chat template length: %s characters", len(chat_template))
    processor_kwargs = {
        "text": [chat_template],
        "return_tensors": "pt",
    }
    inputs = processor(**processor_kwargs)
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
        LOGGER.debug("Moved cleaner inputs to CUDA device %s", model.device)
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )[0].strip()
    LOGGER.info("Cleaner generation completed, output length %s characters", len(generated_text))
    return generated_text
