"""集中配置加载（config.yaml + .env）。ADR-07：切后端/模型只改配置。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


class ModelCfg(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    temperature: float = 0.3
    device: str = "cpu"
    api_key_env: str | None = None


class VectorStoreCfg(BaseModel):
    backend: str = "chroma"
    persist_dir: str = "./data/chroma"
    collection: str = "papers"


class SqliteCfg(BaseModel):
    path: str = "./data/georesearcher.db"


class FilesCfg(BaseModel):
    pdf_dir: str = "./data/pdfs"
    figure_dir: str = "./data/figures"
    report_dir: str = "./data/reports"


class StorageCfg(BaseModel):
    vector_store: VectorStoreCfg = VectorStoreCfg()
    sqlite: SqliteCfg = SqliteCfg()
    files: FilesCfg = FilesCfg()


class ModelsCfg(BaseModel):
    llm: ModelCfg
    judge: ModelCfg
    embedding: ModelCfg


class RetrievalCfg(BaseModel):
    top_k: int = 5
    use_hyde: bool = True
    use_multi_query: bool = True
    use_reranker: bool = True
    use_bm25: bool = True
    use_cross_encoder: bool = True
    rrf_k: int = 60
    hybrid_candidates: int = 40
    rerank_candidates: int = 20


class DiagnosisCfg(BaseModel):
    """评估诊断阈值（经验值，非绝对；可在 config.yaml 覆盖）。"""

    hit_rate_low: float = 0.6  # 检索 hit_rate 低于此 → 判"检索层弱"
    faithfulness_low: float = 0.7  # 生成 faithfulness 低于此 → 判"生成层弱"


class EvaluationCfg(BaseModel):
    eval_set_path: str = "./tests/eval_set/eval_set.json"
    top_k: int = 5
    eval_generation: bool = True
    run_ragas: bool = False
    report_dir: str = "./data/reports"
    diagnosis: DiagnosisCfg = DiagnosisCfg()


class SearchCfg(BaseModel):
    """M3 检索配置（plan-m3 §4）。"""

    provider: str = "openalex"  # A1：默认唯一实现
    mailto: str | None = None  # A2 polite pool
    rate_limit_per_sec: float = 3.0  # A2 客户端限速
    timeout: float = 20.0
    max_retries: int = 2
    limit_default: int = 10


class Config(BaseModel):
    models: ModelsCfg
    storage: StorageCfg = StorageCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    evaluation: EvaluationCfg = EvaluationCfg()  # 带默认，向后兼容
    search: SearchCfg = SearchCfg()  # M3 检索，向后兼容

    def api_key(self, model_cfg: ModelCfg) -> str | None:
        """从环境变量读取密钥（绝不硬编码，绝不入库）。"""
        if not model_cfg.api_key_env:
            return None
        return os.environ.get(model_cfg.api_key_env)


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """加载配置（带缓存）。首次调用时载入 .env。"""
    load_dotenv(_PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
