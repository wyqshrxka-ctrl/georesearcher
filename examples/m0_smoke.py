"""M0 冒烟示例：不需要 API key / 重依赖，验证配置+存储层可跑通。

运行：uv run python examples/m0_smoke.py
"""
from georesearcher.config import load_config
from georesearcher.storage import get_sqlite_store, get_vector_store
from georesearcher.types import Paper


def main():
    cfg = load_config()
    print(f"[config] LLM={cfg.models.llm.model}  VS={cfg.storage.vector_store.backend}")

    store = get_sqlite_store(cfg)
    store.add_paper(Paper(id="demo1", title="Urban vitality via POI", year=2024, venue="Nature Cities"))
    print(f"[sqlite] papers count = {store.count_papers()}")
    store.close()

    vs = get_vector_store(cfg)
    print(f"[vector] backend = {type(vs).__name__}")
    print("M0 smoke OK.")


if __name__ == "__main__":
    main()
