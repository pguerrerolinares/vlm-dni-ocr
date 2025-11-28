"""Unit tests for postprocessing helpers."""
from __future__ import annotations

from dni_pipeline.postprocess import normalize_and_validate


def test_normalize_and_validate_populates_expected_fields() -> None:
    """Happy path: valid inputs are preserved and formatted."""
    data = {
        "nombre": "Laura",
        "primer_apellido": "Garcia",
        "segundo_apellido": "Lopez",
        "dni": "12345678Z",
        "fecha_nacimiento": "01-02-1990",
        "fecha_validez": "15042030",
        "sexo": "female",
        "nacionalidad": "Esp",
    }
    record = normalize_and_validate(data)
    assert record["nombre"] == "Laura"
    assert record["primer_apellido"] == "Garcia"
    assert record["segundo_apellido"] == "Lopez"
    assert record["dni"] == "12345678Z"
    assert record["fecha_nacimiento"] == "01/02/1990"
    assert record["fecha_validez"] == "15/04/2030"
    assert record["sexo"] == "F"
    assert record["nacionalidad"] == "ESP"


def test_normalize_and_validate_handles_missing_and_invalid_values() -> None:
    """Null/invalid values should be normalised to None."""
    data = {
        "dni": "ABC",
        "fecha_nacimiento": "32/13/2020",
        "sexo": "",
    }
    record = normalize_and_validate(data)
    assert all(record[field] is None for field in record)
