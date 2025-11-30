"""Postprocessing helpers for cleaner/validator orchestration."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..logging_service import logging_service

LOGGER = logging_service.get_logger(__name__)
EXPECTED_FIELDS = (
    "nombre",
    "primer_apellido",
    "segundo_apellido",
    "dni",
    "fecha_nacimiento",
    "fecha_validez",
    "sexo",
    "nacionalidad",
)
CRITICAL_FIELDS = {"dni", "fecha_validez", "nacionalidad"}
DATE_FIELDS = {"fecha_nacimiento", "fecha_validez"}


def parse_model_output(text: str) -> Dict[str, Any]:
    """Parse the raw VLM output into a JSON-compatible dict, tolerating loose formats."""
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
    LOGGER.error("Unable to parse model output as JSON")
    return {"raw_output": text}


def build_cleaner_prompt(ocr_block: str, vlm_record: Dict[str, Any]) -> str:
    """Create the textual instructions for the cleaner/validator LLM."""
    vlm_json = json.dumps(vlm_record, ensure_ascii=False, indent=2)
    rules = """
You must strictly apply these validation rules:
1. DNI must match ^\\d{8}[A-Z]$ and the checksum letter uses the canonical sequence TRWAGMYFPDXBNJZSQVHLCKE. If the checksum fails, keep the best guess, set confidence "low" and add an error message such as "checksum mismatch".
2. Dates accept formats DD/MM/AAAA, DD-MM-AAAA, AAAA/MM/DD, AAAA-MM-DD, or "DD MM AAAA". Convert every valid date to ISO `AAAA-MM-DD`. Reject impossible dates (day 1-31, month 1-12, year 1900-2100) by keeping the raw string with confidence "low" plus an error message.
3. Nacionalidad allowed values: ESP, ESPAÑOL, ESPAÑOLA. Map all synonyms to "ESP". Anything else must stay as-is with confidence "low".
4. Sexo allowed values: map M/VARON/H to "M", F/MUJER to "F". Unknown values stay as-is with confidence "low".
5. General cleaning: remove duplicated spaces, stray punctuation, invisible characters, but never invent data.
6. Cross-check every final value with RAW_OCR; if a value is absent from RAW_OCR, lower the confidence unless you have explicit evidence from VLM_PARSED.
7. Always flag manual review for any critical field (DNI, fecha_validez, nacionalidad) that ends with confidence "low".
"""
    schema = """
Your response must be **valid JSON** with this schema (no comments, no markdown):
{
  "fields": {
    "nombre": {"value": null, "confidence": "missing", "errors": []},
    "primer_apellido": {"value": null, "confidence": "missing", "errors": []},
    "segundo_apellido": {"value": null, "confidence": "missing", "errors": []},
    "dni": {"value": null, "confidence": "missing", "errors": []},
    "fecha_nacimiento": {"value": null, "confidence": "missing", "errors": []},
    "fecha_validez": {"value": null, "confidence": "missing", "errors": []},
    "sexo": {"value": null, "confidence": "missing", "errors": []},
    "nacionalidad": {"value": null, "confidence": "missing", "errors": []}
  },
  "manual_review": [],
  "notes": ""
}
Use only the confidence levels: "high", "medium", "low", "missing".
"""
    return (
        "You are a meticulous cleaner + validator for Spanish DNI extractions.\n"
        f"{rules.strip()}\n\n"
        f"{schema.strip()}\n\n"
        "[RAW_OCR]\n"
        f"{ocr_block.strip()}\n"
        "[/RAW_OCR]\n\n"
        "[VLM_PARSED_JSON]\n"
        f"{vlm_json}\n"
        "[/VLM_PARSED_JSON]\n"
        "Return only the JSON object."
    )


def parse_cleaner_output(text: str) -> Dict[str, Any]:
    """Parse the cleaner LLM output."""
    return parse_model_output(text)


def finalize_cleaner_result(cleaner_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all fields exist and compute manual review flags for critical low-confidence entries."""
    if "raw_output" in cleaner_data:
        return cleaner_data

    result_fields: Dict[str, Dict[str, Any]] = {}
    manual_review: List[str] = list(cleaner_data.get("manual_review", []))
    fields_section = cleaner_data.get("fields", {})

    for field in EXPECTED_FIELDS:
        field_info = fields_section.get(field, {})
        value = field_info.get("value")
        confidence = str(field_info.get("confidence", "missing")).lower()
        if confidence not in {"high", "medium", "low", "missing"}:
            confidence = "low"
        errors = field_info.get("errors") or []
        entry = {
            "value": value,
            "confidence": confidence,
            "errors": errors,
        }
        if field == "dni":
            _adjust_dni_entry(entry)
        elif field in DATE_FIELDS and entry["value"]:
            iso_value = _ensure_iso_date(str(entry["value"]))
            if iso_value:
                entry["value"] = iso_value
        result_fields[field] = entry
        if entry["confidence"] == "low" and field in CRITICAL_FIELDS and field not in manual_review:
            manual_review.append(field)

    manual_review = sorted(set(manual_review))
    validations = _build_validations(result_fields)
    return {
        "fields": result_fields,
        "validations": validations,
        "manual_review": manual_review,
        "notes": cleaner_data.get("notes"),
    }


def _adjust_dni_entry(entry: Dict[str, Any]) -> None:
    """Ensure the DNI value contains 8 digits + letter and record reconstruction metadata."""
    value = entry.get("value")
    if not value:
        return
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(value).upper())
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if len(digits) < 8:
        return
    raw_digits = digits[:8]
    entry["raw_dni_digits"] = raw_digits
    expected_letter = _dni_letter(raw_digits)
    explicit_letter = next((ch for ch in cleaned[8:] if ch.isalpha()), None)

    if explicit_letter:
        entry["value"] = raw_digits + explicit_letter
        if explicit_letter != expected_letter:
            entry.setdefault("errors", []).append("checksum mismatch")
            entry.setdefault("meta", {})["reconstructed_letter"] = True
            entry["value"] = raw_digits + expected_letter
            entry["confidence"] = "low"
    else:
        entry["value"] = raw_digits + expected_letter
        entry.setdefault("meta", {})["reconstructed_letter"] = True
        if entry["confidence"] == "high":
            entry["confidence"] = "medium"

    if entry["confidence"] == "low" and "reconstructed_letter" not in entry.get("errors", []):
        entry.setdefault("errors", []).append("reconstructed_letter")


def _dni_letter(number: str) -> str:
    """Return the DNI checksum letter."""
    try:
        index = int(number) % 23
    except ValueError:
        return ""
    letters = "TRWAGMYFPDXBNJZSQVHLCKE"
    return letters[index]


def _build_validations(fields: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """Compute validation booleans for downstream consumers."""
    validations = {
        "dni_checksum_ok": _dni_checksum_ok(fields.get("dni")),
        "fecha_nacimiento_reasonable": _date_reasonable(fields.get("fecha_nacimiento")),
        "fecha_validez_reasonable": _date_reasonable(fields.get("fecha_validez")),
    }
    return validations


def _dni_checksum_ok(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry or not entry.get("value"):
        return False
    value = str(entry["value"]).upper()
    if not re.fullmatch(r"\d{8}[A-Z]", value):
        return False
    digits = value[:8]
    letter = value[-1]
    return _dni_letter(digits) == letter


def _date_reasonable(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry or not entry.get("value"):
        return False
    iso = _ensure_iso_date(str(entry["value"]))
    return iso is not None


def _ensure_iso_date(value: str) -> Optional[str]:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value if _valid_iso_date(value) else None

    for pattern in (r"(\d{2})/(\d{2})/(\d{4})", r"(\d{2})-(\d{2})-(\d{4})", r"(\d{2})\s+(\d{2})\s+(\d{4})"):
        m = re.fullmatch(pattern, value)
        if m:
            day, month, year = m.groups()
            iso = f"{year}-{month}-{day}"
            return iso if _valid_iso_date(iso) else None
    return None


def _valid_iso_date(value: str) -> bool:
    try:
        year, month, day = (int(part) for part in value.split("-"))
        if not (1900 <= year <= 2100):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
    except ValueError:
        return False
    return True
