"""T3: 评估集加载/校验测试（用 tmp_path 造小 JSON，无 LLM）。"""

import json

import pytest

from georesearcher.capabilities.evaluation.dataset import EvalSetError, load_eval_set


def _write(tmp_path, data):
    p = tmp_path / "eval_set.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_valid_set(tmp_path):
    data = [
        {"id": "q001", "question": "Q1?", "expected_paper_ids": ["p1"]},
        {"id": "q002", "question": "Q2?", "expected_paper_ids": ["p2", "p3"], "tags": ["方法"]},
    ]
    cases = load_eval_set(_write(tmp_path, data))
    assert len(cases) == 2
    assert cases[0].id == "q001"
    assert cases[1].tags == ["方法"]


def test_empty_expected_paper_ids_raises(tmp_path):
    data = [{"id": "q001", "question": "Q1?", "expected_paper_ids": []}]
    with pytest.raises(EvalSetError, match="expected_paper_ids 为空"):
        load_eval_set(_write(tmp_path, data))


def test_empty_question_raises(tmp_path):
    data = [{"id": "q001", "question": "   ", "expected_paper_ids": ["p1"]}]
    with pytest.raises(EvalSetError, match="question 为空"):
        load_eval_set(_write(tmp_path, data))


def test_duplicate_id_raises(tmp_path):
    data = [
        {"id": "q001", "question": "Q1?", "expected_paper_ids": ["p1"]},
        {"id": "q001", "question": "Q2?", "expected_paper_ids": ["p2"]},
    ]
    with pytest.raises(EvalSetError, match="id 重复"):
        load_eval_set(_write(tmp_path, data))


def test_non_array_top_level_raises(tmp_path):
    p = tmp_path / "eval_set.json"
    p.write_text(json.dumps({"id": "q001"}), encoding="utf-8")
    with pytest.raises(EvalSetError, match="顶层必须是数组"):
        load_eval_set(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(EvalSetError, match="不存在"):
        load_eval_set(tmp_path / "nope.json")


def test_bad_json_raises(tmp_path):
    p = tmp_path / "eval_set.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvalSetError, match="JSON 解析失败"):
        load_eval_set(p)


def test_sqlite_missing_paper_warns(tmp_path):
    data = [{"id": "q001", "question": "Q1?", "expected_paper_ids": ["p1", "ghost"]}]

    class _FakeStore:
        def get_paper(self, pid):
            return object() if pid == "p1" else None

    with pytest.warns(UserWarning, match="ghost"):
        cases = load_eval_set(_write(tmp_path, data), sqlite_store=_FakeStore())
    assert len(cases) == 1
