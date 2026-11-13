from datamate.core.scoring import score_all, score_one


def test_abstain_and_missing_score_zero():
    ref = {"type": "number", "answer": 5}
    assert score_one({"type": "abstain"}, ref) == 0 and score_one(None, ref) == 0


def test_wrong_type_is_wrong():
    assert score_one({"type": "string", "answer": "5"}, {"type": "number", "answer": 5}) == -1


def test_number_and_percent():
    assert score_one({"type": "number", "answer": 156305}, {"type": "number", "answer": 156305.0}) == 1
    assert score_one({"type": "number", "answer": 156304.55}, {"type": "number", "answer": 156305}) == -1
    assert score_one({"type": "percent", "answer": 12.344}, {"type": "percent", "answer": 12.34}) == 1
    assert score_one({"type": "number", "answer": True}, {"type": "number", "answer": 1}) == -1


def test_string_normalisation_and_alias():
    ref = {"type": "string", "answer": "Cà phê sữa", "alias": ["cafe sua"]}
    assert score_one({"type": "string", "answer": "  cà phê, SỮA "}, ref) == 1
    assert score_one({"type": "string", "answer": "Cafe sua"}, ref) == 1
    assert score_one({"type": "string", "answer": "Cà phê đen"}, ref) == -1


def test_date_boolean_unanswerable():
    assert score_one({"type": "date", "answer": "2025-03-15"}, {"type": "date", "answer": "2025-03-15"}) == 1
    assert score_one({"type": "boolean", "answer": 1}, {"type": "boolean", "answer": True}) == -1
    assert score_one({"type": "unanswerable", "reason": "x"}, {"type": "unanswerable"}) == 1


def test_list_is_ordered():
    ref = {"type": "list", "answer": ["A", "B"], "alias": [["a1"], []]}
    assert score_one({"type": "list", "answer": ["a1", "b"]}, ref) == 1
    assert score_one({"type": "list", "answer": ["B", "A"]}, ref) == -1
    assert score_one({"type": "list", "answer": ["A"]}, ref) == -1


def test_table_rows_any_order_columns_by_name():
    ref = {"type": "table", "answer": {"columns": ["name", "qty"], "rows": [["A", 1], ["B", 2]]}}
    assert score_one({"type": "table", "answer": {"columns": ["qty", "name"], "rows": [[2, "B"], [1, "A"]]}}, ref) == 1
    assert score_one({"type": "table", "answer": {"columns": ["name", "qty"], "rows": [["A", 1], ["B", 3]]}}, ref) == -1


def test_score_all_penalized_accuracy():
    refs = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 2},
            "c": {"type": "number", "answer": 3}, "d": {"type": "number", "answer": 4}}
    preds = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 9}, "c": {"type": "abstain"}}
    r = score_all(preds, refs)
    assert (r["correct"], r["wrong"], r["abstain"]) == (1, 1, 2) and r["PA"] == 0.0
