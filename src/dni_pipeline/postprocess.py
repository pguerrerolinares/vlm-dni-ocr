"""Postprocessing helpers: parse model output and normalise fields."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

from .logging_service import logging_service

TARGET_FIELDS = [
    "nombre",
    "primer_apellido",
    "segundo_apellido",
    "dni",
    "fecha_nacimiento",
    "fecha_validez",
    "sexo",
    "nacionalidad",
]

_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

LOGGER = logging_service.get_logger(__name__)


def parse_model_output(text: str) -> Dict[str, Any]:
    """
    Parse the raw model output into a JSON-compatible dictionary.

    Returns a dictionary with extracted fields. When parsing fails, the dict contains a
    single ``raw_output`` entry with the unmodified text so callers can log/inspect it.
    """
    if not text:
        LOGGER.warning("Model output was empty")
        return {"raw_output": ""}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            LOGGER.info("Parsed model output as JSON on first attempt")
            return data
    except json.JSONDecodeError:
        LOGGER.debug("Direct JSON parsing failed; attempting to locate JSON fragment")
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                LOGGER.info("Parsed JSON from extracted fragment")
                return data
        except json.JSONDecodeError:
            LOGGER.debug("Fragment-based JSON parsing failed")
            pass
    LOGGER.error("Unable to parse model output as JSON")
    return {"raw_output": text}


def normalize_and_validate(data: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Normalise the extracted fields, enforcing null for uncertain values."""
    record: Dict[str, Optional[str]] = {field: None for field in TARGET_FIELDS}
    if not data:
        LOGGER.warning("No data provided for normalisation; returning empty record")
        return record

    record["nombre"] = _normalise_name(data.get("nombre"))
    record["primer_apellido"] = _normalise_name(data.get("primer_apellido"))
    record["segundo_apellido"] = _normalise_name(data.get("segundo_apellido"))
    record["dni"] = _normalise_dni(data.get("dni"))
    record["fecha_nacimiento"] = _normalise_date(data.get("fecha_nacimiento"))
    record["fecha_validez"] = _normalise_date(data.get("fecha_validez"))
    record["sexo"] = _normalise_sex(data.get("sexo"))
    record["nacionalidad"] = _normalise_nationality(data.get("nacionalidad"))
    LOGGER.info("Normalised record: %s", record)
    return record


def _to_clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    elif not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalise_name(value: Any) -> Optional[str]:
    cleaned = _to_clean_string(value)
    if not cleaned:
        return None
    return cleaned


def _normalise_dni(value: Any) -> Optional[str]:
    cleaned = _to_clean_string(value)
    if not cleaned:
        LOGGER.debug("DNI normalisation: empty input -> null")
        return None
    simplified = re.sub(r"[^0-9A-Za-z]", "", cleaned).upper()
    if len(simplified) == 8 and simplified.isdigit():
        letter = _DNI_LETTERS[int(simplified) % 23]
        return f"{simplified}{letter}"
    if len(simplified) == 9 and simplified[:-1].isdigit() and simplified[-1].isalpha():
        expected_letter = _DNI_LETTERS[int(simplified[:-1]) % 23]
        if simplified[-1] == expected_letter:
            return simplified
        LOGGER.debug("DNI normalisation: checksum mismatch for %s", simplified)
    return None


def _normalise_date(value: Any) -> Optional[str]:
    cleaned = _to_clean_string(value)
    if not cleaned:
        LOGGER.debug("Date normalisation: empty input -> null")
        return None
    digits = re.sub(r"[^0-9]", "", cleaned)
    if len(digits) == 8:
        day, month, year = digits[:2], digits[2:4], digits[4:]
    else:
        parts = [p for p in re.split(r"[^\d]", cleaned) if p]
        if len(parts) != 3:
            LOGGER.debug("Date normalisation: unexpected parts for value %s", cleaned)
            return None
        day, month, year = parts
    if len(year) == 2:
        LOGGER.debug("Date normalisation: refusing two-digit year for value %s", cleaned)
        return None
    try:
        dt = datetime(year=int(year), month=int(month), day=int(day))
    except ValueError:
        LOGGER.debug("Date normalisation: invalid calendar date for value %s", cleaned)
        return None
    return dt.strftime("%d/%m/%Y")


def _normalise_sex(value: Any) -> Optional[str]:
    cleaned = _to_clean_string(value)
    if not cleaned:
        LOGGER.debug("Sex normalisation: empty input -> null")
        return None
    upper = cleaned.upper()
    if upper in {"M", "F"}:
        return upper
    if upper in {"H", "VARON", "HOMBRE", "MALE"}:
        return "M"
    if upper in {"MUJER", "FEMALE", "FEMENINO"}:
        return "F"
    if len(upper) == 1:
        return upper
    return None


def _normalise_nationality(value: Any) -> Optional[str]:
    cleaned = _to_clean_string(value)
    if not cleaned:
        LOGGER.debug("Nationality normalisation: empty input -> null")
        return None
    return cleaned.upper()
