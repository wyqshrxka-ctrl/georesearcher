import textwrap

from georesearcher.config import Config, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.models.llm.provider == "deepseek"
    assert cfg.models.llm.base_url == "https://api.deepseek.com"
    assert cfg.models.embedding.provider == "local"
    assert cfg.storage.vector_store.backend in {"chroma", "milvus"}
    assert cfg.retrieval.top_k > 0


def test_evaluation_section_loads():
    cfg = load_config()
    assert cfg.evaluation.top_k > 0
    assert 0.0 < cfg.evaluation.diagnosis.hit_rate_low <= 1.0
    assert 0.0 < cfg.evaluation.diagnosis.faithfulness_low <= 1.0


def test_evaluation_backward_compatible(tmp_path):
    """不写 evaluation 段也能加载（向后兼容，M0/M1 config 不破）。"""
    minimal = textwrap.dedent(
        """
        models:
          llm: {provider: deepseek, model: deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
          judge: {provider: deepseek, model: deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
          embedding: {provider: local, model: BAAI/bge-m3}
        """
    )
    raw = __import__("yaml").safe_load(minimal)
    cfg = Config.model_validate(raw)
    # 未提供 evaluation 段 → 用默认值
    assert cfg.evaluation.eval_set_path.endswith("eval_set.json")
    assert cfg.evaluation.eval_generation is True
    assert cfg.evaluation.diagnosis.hit_rate_low == 0.6


def test_search_section_loads():
    """M3 search section 在 config.yaml 中存在且正确加载。"""
    cfg = load_config()
    assert cfg.search.provider == "openalex"
    assert cfg.search.rate_limit_per_sec > 0
    assert cfg.search.limit_default > 0


def test_search_backward_compatible(tmp_path):
    """不写 search 段也能加载（向后兼容）。"""
    minimal = textwrap.dedent(
        """
        models:
          llm: {provider: deepseek, model: deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
          judge: {provider: deepseek, model: deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
          embedding: {provider: local, model: BAAI/bge-m3}
        """
    )
    raw = __import__("yaml").safe_load(minimal)
    cfg = Config.model_validate(raw)
    # 未提供 search 段 → 用默认值
    assert cfg.search.provider == "openalex"
    assert cfg.search.rate_limit_per_sec == 3.0
    assert cfg.search.limit_default == 10
