"""④ 模型层：LLM / judge / embedding 的接口抽象（可切）。"""
from .llm import LLMClient, get_llm, get_judge
from .embedding import Embedder, get_embedder

__all__ = ["LLMClient", "get_llm", "get_judge", "Embedder", "get_embedder"]
