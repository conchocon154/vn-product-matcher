import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vnmatch.normalize import (
    extract_dimensions,
    extract_material,
    normalize,
    normalize_unit,
    strip_diacritics,
)


def test_strip_diacritics_keeps_d_distinct():
    assert strip_diacritics("Đinh rút") == "Dinh rut"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Bu lông lục giác ngoài 10x50x17 inox 304", "bu long luc giac ngoai 10x50x17 inox 304"),
        ("BL LGN 10*50 i304", "bu long luc giac ngoai 10x50 inox 304"),
        ("bu long  10 X 50", "bu long 10x50"),
    ],
)
def test_normalize_canonical_forms(raw, expected):
    assert normalize(raw) == expected


def test_x_inside_inox_is_not_a_multiplication_sign():
    # Regression: `inox 304` once collapsed to `inox304`, breaking material match.
    assert normalize("inox 304") == "inox 304"


def test_metric_thread_is_one_token():
    assert normalize("bu long M 8") == "bu long m8"


def test_extract_dimensions():
    assert extract_dimensions("Vít 10x50x17 inox") == ["10x50x17"]
    assert extract_dimensions("Bu lông M8 inox") == ["m8"]
    assert extract_dimensions("Sơn chống rỉ") == []


def test_extract_material_prefers_the_specific_grade():
    assert extract_material("bu long inox 304") == "inox 304"
    assert extract_material("pat sat xi") == "sat xi"
    assert extract_material("son chong ri") is None


def test_normalize_unit_aliases():
    assert normalize_unit("Cái") == "cai"
    assert normalize_unit("Kg") == "kg"
