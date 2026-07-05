"""scripts/gen_eval_set.py 纯函数单测（无 LLM、无 SQLite）。

脚本不在 package 里，用 importlib 按路径加载。只测纯函数：
_theme_clusters（主题簇分组）、_parse_qa_json（含多篇 relevant_paper_idx）、
_resolve_group_gt（idx → paper id）。
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_eval_set.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_eval_set", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ges = _load_module()


def _paper(pid, tags):
    return {"id": pid, "title": f"title-{pid}", "tags": tags}


# ---------- _theme_clusters ----------

def test_clusters_only_use_theme_tags():
    # 纯地理 tag 的论文不成簇；主题 tag 才分组
    papers = [
        _paper("p1", ["教育不平等/学校隔离", "欧洲"]),
        _paper("p2", ["教育不平等/学校隔离", "北美"]),
        _paper("p3", ["空间分析/空间自相关"]),
        _paper("g1", ["欧洲"]),  # 纯地理，不该成簇
        _paper("g2", ["北美"]),
    ]
    clusters = ges._theme_clusters(papers, group_size=3, group_count=5, seed=1)
    all_ids = {p["id"] for c in clusters for p in c}
    # 纯地理论文不出现在任何簇里
    assert "g1" not in all_ids and "g2" not in all_ids
    # 学校隔离簇应含 p1,p2（该 tag 下唯二）
    seg = [c for c in clusters if {"p1", "p2"} <= {p["id"] for p in c}]
    assert seg, "学校隔离主题应组成一个含 p1,p2 的簇"


def test_clusters_size_between_2_and_group_size():
    papers = [_paper(f"p{i}", ["教育不平等/学校隔离"]) for i in range(6)]
    clusters = ges._theme_clusters(papers, group_size=3, group_count=3, seed=7)
    assert clusters, "应能组出簇"
    for c in clusters:
        assert 2 <= len(c) <= 3


def test_clusters_skip_tag_with_single_paper():
    # 某主题 tag 只有 1 篇 → 不成簇
    papers = [
        _paper("solo", ["研究方法/量化方法"]),
        _paper("a", ["空间分析/空间自相关"]),
        _paper("b", ["空间分析/空间自相关"]),
    ]
    clusters = ges._theme_clusters(papers, group_size=3, group_count=5, seed=3)
    all_ids = {p["id"] for c in clusters for p in c}
    assert "solo" not in all_ids


def test_clusters_empty_when_no_theme_tags():
    papers = [_paper("g1", ["欧洲"]), _paper("g2", ["北美"])]
    assert ges._theme_clusters(papers, group_size=3, group_count=5, seed=1) == []


def test_clusters_respects_group_count():
    papers = [_paper(f"p{i}", ["教育不平等/学校隔离"]) for i in range(10)]
    clusters = ges._theme_clusters(papers, group_size=2, group_count=2, seed=1)
    assert len(clusters) <= 2


# ---------- _parse_qa_json ----------

def test_parse_single_qa():
    raw = '[{"question": "Q1?", "reference_answer": "A1"}]'
    out = ges._parse_qa_json(raw)
    assert out == [{"question": "Q1?", "reference_answer": "A1"}]


def test_parse_with_relevant_idx():
    raw = '[{"question": "Q?", "reference_answer": "A", "relevant_paper_idx": [1, 3]}]'
    out = ges._parse_qa_json(raw)
    assert out[0]["relevant_paper_idx"] == [1, 3]


def test_parse_strips_code_fence():
    raw = '```json\n[{"question": "Q?", "reference_answer": "A"}]\n```'
    out = ges._parse_qa_json(raw)
    assert out[0]["question"] == "Q?"


def test_parse_dedups_and_filters_idx():
    raw = '[{"question": "Q?", "relevant_paper_idx": [2, 2, "x", 0, -1, 3]}]'
    out = ges._parse_qa_json(raw)
    assert out[0]["relevant_paper_idx"] == [2, 3]


def test_parse_malformed_returns_empty():
    assert ges._parse_qa_json("not json at all") == []
    assert ges._parse_qa_json("") == []
    assert ges._parse_qa_json('{"question": "x"}') == []  # 非数组


def test_parse_skips_items_without_question():
    raw = '[{"reference_answer": "A"}, {"question": "Q?"}]'
    out = ges._parse_qa_json(raw)
    assert len(out) == 1 and out[0]["question"] == "Q?"


# ---------- _resolve_group_gt ----------

def test_resolve_gt_uses_idx_subset():
    group = [_paper("a", []), _paper("b", []), _paper("c", [])]
    # 选第 1、3 篇 → [a, c]
    assert ges._resolve_group_gt(group, [1, 3]) == ["a", "c"]


def test_resolve_gt_none_falls_back_to_all():
    group = [_paper("a", []), _paper("b", [])]
    assert ges._resolve_group_gt(group, None) == ["a", "b"]


def test_resolve_gt_less_than_two_falls_back_to_all():
    # idx 只指 1 篇 → 不足 2，退回整组
    group = [_paper("a", []), _paper("b", []), _paper("c", [])]
    assert ges._resolve_group_gt(group, [2]) == ["a", "b", "c"]


def test_resolve_gt_ignores_out_of_range_idx():
    group = [_paper("a", []), _paper("b", [])]
    # idx 5 越界被忽略；剩 [1,2] → [a,b]
    assert ges._resolve_group_gt(group, [1, 2, 5]) == ["a", "b"]
