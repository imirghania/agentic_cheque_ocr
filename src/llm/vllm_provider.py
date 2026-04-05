from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from logger import logger
from .base import LlmProvider

if TYPE_CHECKING:
    from config.settings import Settings


class VllmProvider(LlmProvider):
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        temperature: float = 0,
    ) -> None:
        logger.info("Initializing vLLM: model=%s, base_url=%s, temperature=%.1f",
                     model, base_url, temperature)
        self._llm = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=temperature,
        )

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "VllmProvider":
        model = overrides.pop("model", settings.vllm_model)
        base_url = overrides.pop("base_url", settings.vllm_base_url)
        logger.debug("VllmProvider.from_settings: model=%s, base_url=%s",
                     model, base_url)
        return cls(model=model, base_url=base_url, **overrides)

    def extract_json(self, prompt: str, schema: dict) -> dict:
        logger.debug("vLLM extract_json: invoking LLM")
        t0 = time.time()
        response = self._llm.invoke(prompt)
        elapsed = time.time() - t0
        logger.debug("vLLM response received in %.2fs", elapsed)
        content = response.content
        if isinstance(content, dict):
            return content
        return json.loads(content)
