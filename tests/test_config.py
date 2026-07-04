from georesearcher.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.models.llm.provider == "deepseek"
    assert cfg.models.llm.base_url == "https://api.deepseek.com"
    assert cfg.models.embedding.provider == "local"
    assert cfg.storage.vector_store.backend in {"chroma", "milvus"}
    assert cfg.retrieval.top_k > 0
