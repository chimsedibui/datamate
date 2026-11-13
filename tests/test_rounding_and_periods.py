import pandas as pd

from datamate.core.parsers import round_half_up
from datamate.qa.agent import SC_VARIANTS, normalize_answer
from datamate.qa.checks import periods_outside


def test_round_half_up_matches_rulebook():
    assert round_half_up(1234.5) == 1235          # Python round() gives 1234 (half to even)
    assert round_half_up(2.5) == 3
    assert round_half_up(2.675, 2) == 2.68        # round(2.675, 2) == 2.67 on the binary float
    assert round_half_up(-2.5) == -3
    assert round_half_up(12.344, 2) == 12.34


def test_money_rounds_half_up_from_the_raw_value():
    assert normalize_answer({"type": "number", "answer": 1358008.5}) == {"type": "number", "answer": 1358009}
    assert normalize_answer({"type": "number", "answer": 1234.495}) == {"type": "number", "answer": 1234}
    assert normalize_answer({"type": "number", "answer": 12.345}) == {"type": "number", "answer": 12.35}
    assert normalize_answer({"type": "percent", "answer": 33.335}) == {"type": "percent", "answer": 33.34}
    assert normalize_answer({"type": "list", "answer": [1500.5, "a"]}) == {"type": "list", "answer": [1501, "a"]}


def test_periods_outside_data():
    t = {"orders": pd.DataFrame({"order_date": pd.to_datetime(["2025-02-01", "2026-09-14"])})}
    assert periods_outside([("tháng 1/2025", "2025-01-01", "2025-01-31")], t)
    assert periods_outside([("x", "2026-10-01", "2026-10-31")], t)
    assert not periods_outside([("x", "2025-01-15", "2025-02-15")], t)      # partly inside -> answerable
    assert not periods_outside([("a", "2025-01-01", "2025-01-31"), ("b", "2025-03-01", "2025-03-31")], t)
    assert not periods_outside([], t) and not periods_outside([("x", "2025-01-01", "2025-01-31")], {})


def test_variant_zero_is_the_plain_prompt():
    assert SC_VARIANTS[0] == "" and len(set(SC_VARIANTS)) == len(SC_VARIANTS) >= 3
