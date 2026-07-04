"""本地 Embedding 接口（bge-m3）。ADR-05：本地、免费、离线，不做 API 抽象。

执行者：sentence-transformers 是 [rag] 可选依赖，M0 不强制安装。
未安装时 get_embedder() 仍可构造，只有真正 embed 时才加载模型。
"""
from __future__ import annotations

from ..config import Config, ModelCfg, load_config


class Embedder:
    """本地句向量编码器（延迟加载模型）。"""

    def __init__(self, cfg: ModelCfg):
        self._cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "需要安装 embedding 依赖：uv sync --extra rag"
                ) from e
            self._model = SentenceTransformer(self._cfg.model, device=self._cfg.device)
        return self._model

    @property
    def model_name(self) -> str:
        return self._cfg.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder(cfg: Config | None = None) -> Embedder:
    cfg = cfg or load_config()
    return Embedder(cfg.models.embedding)
