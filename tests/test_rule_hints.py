"""Rulebook row retrieval and README conventions shown to the QA model."""
import pandas as pd

from datamate.qa import helpers, qparse
from datamate.qa.agent import readme_conventions

RULEBOOK = """# Rulebook
## 2. Thời gian
| Luật | Mặc định |
|---|---|
| Tháng này | từ ngày 1 đến hôm nay |
## 3. Xếp hạng
| Luật | Mặc định |
|---|---|
| Khách sộp nhất | khách có **giá trị trung bình một đơn** cao nhất, từ 3 đơn trở lên |
| Bán chạy | theo **số lượng** |
## 6. Theo ngành
### Sửa xe
| Luật | Mặc định |
|---|---|
| Bảo hành | số phiếu bảo hành chia cho số phiếu có thu tiền |
### Cửa hàng điện thoại
| Luật | Mặc định |
|---|---|
| Bảo hành | số lượt bảo hành chia cho số máy bán |
"""


def test_parse_and_industry():
    rules = qparse.parse_rulebook(RULEBOOK)
    assert [r["key"] for r in rules] == ["Tháng này", "Khách sộp nhất", "Bán chạy", "Bảo hành", "Bảo hành"]
    assert qparse.industry_of(rules, "Motorcycle repair. Sổ sửa xe chủ tiệm tự ghi") == "Sửa xe"
    assert qparse.industry_of(rules, "Điện thoại di động") == "Cửa hàng điện thoại"
    assert qparse.industry_of(rules, "tạp hoá") is None


def test_relevant_rules():
    rules = qparse.parse_rulebook(RULEBOOK)
    got = qparse.relevant_rules("Quý 3 tỉ lệ bảo hành bao nhiêu %?", rules, "Sửa xe")
    assert [(r["industry"], r["key"]) for r in got] == [("Sửa xe", "Bảo hành")]   # never the phone shop's rule
    got = qparse.relevant_rules("thang nay khach nao sop nhat", rules, None)
    assert [r["key"] for r in got] == ["Khách sộp nhất"]                            # time rows are not restated
    assert qparse.relevant_rules("doanh thu tháng này", rules, None) == []


def test_readme_conventions():
    txt = "# Dữ liệu x\n\nSổ sửa xe.\n\n## File\n- `a.xlsx`\n\n## Quy ước\n- Hoa hồng tính trên tổng Tiền."
    out = readme_conventions(txt)
    assert "Hoa hồng" in out and "Sổ sửa xe" in out and "a.xlsx" not in out


def test_due_dates():
    assert helpers.due_dates("2026-01-31", 31, 2) == [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-03-31")]
    assert helpers.due_dates("2025-12-10", 5, 1) == [pd.Timestamp("2026-01-05")]


def test_qty_in_shop_without_units():
    t = {"products": pd.DataFrame({"product_id": ["P1", "P2"], "name": ["Máy A", "Ốp B"], "base_unit": [None, None]}),
         "order_lines": pd.DataFrame({"product_id": ["P1", "P2"], "qty": [2, 3], "qty_base": [2, 3],
                                      "unit": [None, None]})}
    assert helpers.qty_in(t, t["order_lines"], "chiếc").tolist() == [2.0, 3.0]
    # one product with a recorded unit: the others are not counted in it (rulebook "bán được bao nhiêu <đơn vị>")
    t["products"]["base_unit"] = ["chiếc", "cái"]
    assert helpers.qty_in(t, t["order_lines"], "chiếc").fillna(-1).tolist() == [2.0, -1.0]
