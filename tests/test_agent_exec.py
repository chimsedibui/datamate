"""run_code: the error text goes back to the LLM, so it must not depend on install paths or helper line numbers."""
import pandas as pd

from datamate.qa.agent import normalize_answer, run_code


def test_ok_result_is_normalised():
    code = 'def solve():\n    return {"type": "number", "answer": 156304.55}'
    assert run_code(code, {}, 20) == ("ok", {"type": "number", "answer": 156305})


def test_error_text_has_no_paths():
    code = 'def solve():\n    return {"type": "number", "answer": t["orders"].dropna(subset=["due_day"]).shape[0]}'
    status, out = run_code(code, {"t": {"orders": pd.DataFrame({"a": [1]})}}, 20)
    assert status == "error" and 'File "<solve>", line 2, in solve' in out and out.rstrip().endswith("KeyError: ['due_day']")
    assert "/" not in out and "_child" not in out


def test_timeout():
    status, out = run_code("def solve():\n    while True:\n        pass", {}, 1)
    assert status == "error" and out.startswith("TimeoutError")


def test_normalize_answer_table_from_records():
    got = normalize_answer({"type": "table", "answer": [{"name": "A", "amount": 2500.4}]})
    assert got == {"type": "table", "answer": {"columns": ["name", "amount"], "rows": [["A", 2500]]}}
