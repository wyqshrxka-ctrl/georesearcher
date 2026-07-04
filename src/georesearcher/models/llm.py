"""LLM 接口（DeepSeek，OpenAI 兼容）。judge 复用同一接口，只是配置不同。

执行者：DeepSeek 用 OpenAI 兼容端点，不要找 deepseek 专用 SDK（design §12.2）。
密钥从环境变量读取，绝不硬编码。
"""
from __future__ import annotations

from ..config import Config, ModelCfg, load_config


class LLMClient:
    """对 OpenAI 兼容 Chat API 的薄封装。"""

    def __init__(self, cfg: ModelCfg, api_key: str | None):
        self._cfg = cfg
        self._api_key = api_key
        self._client = None  # 延迟初始化，避免无 key 时导入即报错

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    f"未找到 API key，请在 .env 设置 {self._cfg.api_key_env}"
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._cfg.base_url)
        return self._client

    @property
    def model(self) -> str:
        return self._cfg.model

    def chat(self, messages: list[dict], **kwargs) -> str:
        """发送对话消息，返回文本。"""
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=kwargs.pop("temperature", self._cfg.temperature),
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str, **kwargs) -> str:
        """便捷方法：单条 user prompt。"""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


def get_llm(cfg: Config | None = None) -> LLMClient:
    cfg = cfg or load_config()
    return LLMClient(cfg.models.llm, cfg.api_key(cfg.models.llm))


def get_judge(cfg: Config | None = None) -> LLMClient:
    """评估用 judge，可在 config.yaml 里配置为不同模型（ADR-06）。"""
    cfg = cfg or load_config()
    return LLMClient(cfg.models.judge, cfg.api_key(cfg.models.judge))
