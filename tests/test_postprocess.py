"""Unit tests for postprocess helpers."""
from __future__ import annotations

from dni_pipeline.core.postprocess import (
    build_cleaner_prompt,
    finalize_cleaner_result,
    parse_cleaner_output,
    parse_model_output,
)


def test_parse_model_output_handles_json_fragment() -> None:
    """The parser should extract JSON embedded in text."""
    text = "Answer:\n{\"dni\":\"12345678Z\"}\nThanks"
    parsed = parse_model_output(text)
    assert parsed["dni"] == "12345678Z"


def test_build_cleaner_prompt_contains_required_sections() -> None:
    """Prompt must include OCR, VLM JSON and checksum rules."""
    prompt = build_cleaner_prompt("[OCR]\n123\n[/OCR]", {"dni": "123"})
    assert "[RAW_OCR]" in prompt
    assert "[VLM_PARSED_JSON]" in prompt
    assert "TRWAGMYFPDX" in prompt


def test_parse_cleaner_output_preserves_raw_text_when_invalid() -> None:
    """Cleaner parser should fall back to raw_output when JSON is broken."""
    parsed = parse_cleaner_output("not json at all")
    assert "raw_output" in parsed


def test_finalize_cleaner_result_enforces_manual_review() -> None:
    """Critical fields with low confidence must trigger manual review."""
    cleaner_data = {
        "fields": {
            "dni": {"value": "12345678Z", "confidence": "low", "errors": ["checksum mismatch"]},
            "nacionalidad": {"value": "ESP", "confidence": "high", "errors": []},
            "fecha_validez": {"value": "07/07/2026", "confidence": "high", "errors": []},
        },
        "manual_review": [],
        "notes": "sample",
    }
    result = finalize_cleaner_result(cleaner_data)
    assert "dni" in result["manual_review"]
    assert result["fields"]["nombre"]["confidence"] == "missing"
    assert result["validations"]["dni_checksum_ok"] is True
    assert result["fields"]["fecha_validez"]["value"] == "2026-07-07"


def test_finalize_cleaner_result_reconstructs_missing_letter() -> None:
    """When the DNI lacks its control letter, it should be reconstructed with downgraded confidence."""
    cleaner_data = {
        "fields": {
            "dni": {"value": "79139242", "confidence": "high", "errors": []},
        },
        "manual_review": [],
        "notes": None,
    }
    result = finalize_cleaner_result(cleaner_data)
    dni_entry = result["fields"]["dni"]
    assert dni_entry["value"] == "79139242Z"
    assert dni_entry["confidence"] == "medium"
    assert dni_entry["raw_dni_digits"] == "79139242"
    assert dni_entry["meta"]["reconstructed_letter"] is True
    assert result["validations"]["dni_checksum_ok"] is True


def test_finalize_cleaner_result_handles_checksum_mismatch() -> None:
    """Mismatched letters should be replaced and confidence forced to low."""
    cleaner_data = {
        "fields": {
            "dni": {"value": "79139242A", "confidence": "high", "errors": []},
        },
        "manual_review": [],
    }
    result = finalize_cleaner_result(cleaner_data)
    dni_entry = result["fields"]["dni"]
    assert dni_entry["value"] == "79139242Z"
    assert dni_entry["confidence"] == "low"
    assert "checksum mismatch" in dni_entry["errors"]
    assert result["validations"]["dni_checksum_ok"] is True


def test_finalize_cleaner_result_converts_dates_to_iso() -> None:
    """Date fields must end in ISO format when valid."""
    cleaner_data = {
        "fields": {
            "fecha_nacimiento": {"value": "08/12/1996", "confidence": "high", "errors": []},
            "fecha_validez": {"value": "2026-07-07", "confidence": "high", "errors": []},
        },
    }
    result = finalize_cleaner_result(cleaner_data)
    assert result["fields"]["fecha_nacimiento"]["value"] == "1996-12-08"
    assert result["fields"]["fecha_validez"]["value"] == "2026-07-07"
    assert result["validations"]["fecha_nacimiento_reasonable"] is True
    assert result["validations"]["fecha_validez_reasonable"] is True
