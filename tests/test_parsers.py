import math

import pandas as pd
import pytest

from datamate.core import parsers as P


@pytest.mark.parametrize("raw, want", [
    ("55.000đ", 55000), ("25k", 25000), ("1,01tr", 1010000), ("1,500.0000", 1500), (22000, 22000),
    ("1tr090", 1090000), ("1k5", 1500), ("12.500 VND", 12500),
])
def test_parse_money(raw, want):
    assert P.parse_money(raw) == pytest.approx(want)


def test_parse_money_blank_and_series():
    assert math.isnan(P.parse_money(None))
    assert math.isnan(P.parse_money("abc"))
    s = P.parse_money(pd.Series(["10k", None, 5]))
    assert s.dtype == float and s[0] == 10000 and math.isnan(s[1]) and s[2] == 5


def test_parse_qty_unit():
    assert P.parse_qty_unit("5 chai") == (5.0, "chai")
    assert P.parse_qty_unit(3) == (3.0, None)
    df = P.parse_qty_unit(pd.Series(["2 hộp", "1,5 kg"]))
    assert list(df.columns) == ["qty", "unit"] and df.qty.tolist() == [2.0, 1.5] and df.unit.tolist() == ["hộp", "kg"]


@pytest.mark.parametrize("raw, want", [
    ("15/03/2025", "2025-03-15"), ("2025-03-15", "2025-03-15"), ("1 thg 5 2025", "2025-05-01"),
])
def test_parse_date(raw, want):
    assert pd.Timestamp(P.parse_date(raw)) == pd.Timestamp(want)


def test_parse_date_default_year():
    assert pd.Timestamp(P.parse_date("1/5", default_year=2025)) == pd.Timestamp("2025-05-01")


def test_text_normalisation():
    assert P.norm_text("  Cà  Phê SỮA ") == "cà phê sữa"
    assert P.fold_text("Đường Cát Trắng") == "duong cat trang"
    assert P.norm_text(None) is None and P.fold_text(float("nan")) is None and P.norm_text("<NA>") is None


@pytest.mark.parametrize("label, want", [
    ("Đã hủy đơn", "cancelled"), ("không giao được", "cancelled"), ("Khách trả hàng", "returned"),
    ("chưa giao", "delivering"), ("Giao thành công", "completed"), ("ok", None), (None, None),
])
def test_status_hint(label, want):
    assert P.status_hint(label) == want


def test_map_status_mapping_and_default():
    assert P.map_status("ok", {"OK": "completed"}) == "completed"
    assert P.map_status("???") is None
    assert P.map_status(pd.Series(["Đã hủy", "ok"]), {"ok": "completed"}).tolist() == ["cancelled", "completed"]


def test_year_from_text():
    assert P.year_from_text("Thang 02 2025") == 2025 and P.year_from_text("T7") is None
