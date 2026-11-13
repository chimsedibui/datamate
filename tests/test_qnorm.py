import pytest

from datamate.qa.qnorm import normalize


@pytest.mark.parametrize("raw, want", [
    ("Dthu th7/26 bnhiu?", "doanh thu tháng 7/2026 bao nhiêu?"),
    ("T10/2025 bán được bao nhiêu đơn", "tháng 10/2025 bán được bao nhiêu đơn"),
    ("doanh thu 8-2025", "doanh thu tháng 8/2025"),
    ("doanh thu q2/2026", "doanh thu quý 2/2026"),
    ("doanh thu tháng ni", "doanh thu tháng này"),
    ("hôm ni bán mấy đơn", "hôm nay bán mấy đơn"),
    ("sl sp bán hqua", "số lượng sản phẩm bán hôm qua"),
])
def test_normalize(raw, want):
    assert normalize(raw) == want


@pytest.mark.parametrize("q", [
    "Doanh thu tháng 7/2026 là bao nhiêu?",      # already plain
    "Doanh thu ngày 15/08/2025",                 # full date, not a month
    "Doanh thu quý 2/2026",                      # '2/2026' belongs to 'quý'
    "Giá 13/2025",                               # not a valid month
    "",
])
def test_normalize_leaves_plain_text(q):
    assert normalize(q) == q


@pytest.mark.parametrize("q", ["Dthu th7/26 bnhiu z?", "kh nào mua nhiều nhứt thg 5-2026", "8/26 bán dc bn đh"])
def test_normalize_is_idempotent(q):
    once = normalize(q)
    assert normalize(once) == once
