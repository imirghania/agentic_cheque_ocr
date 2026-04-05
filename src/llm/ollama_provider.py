from __future__ import annotations

import json as _json
import logging
import time
from typing import TYPE_CHECKING

from langchain_ollama import ChatOllama

from logger import logger
from .base import LlmProvider

if TYPE_CHECKING:
    from config.settings import Settings


class OllamaProvider(LlmProvider):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0,
    ) -> None:
        logger.info("Initializing Ollama LLM: model=%s, base_url=%s, temperature=%.1f",
                     model, base_url, temperature)
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "OllamaProvider":
        model = overrides.pop("model", settings.ollama_model)
        base_url = overrides.pop("base_url", settings.ollama_base_url)
        logger.debug("OllamaProvider.from_settings: model=%s, base_url=%s",
                     model, base_url)
        return cls(model=model, base_url=base_url, **overrides)

    def extract_json(self, prompt: str, schema: dict) -> dict:
        logger.debug("Ollama extract_json: invoking LLM")
        t0 = time.time()
        response = self._llm.invoke(prompt)
        elapsed = time.time() - t0
        logger.debug("Ollama response received in %.2fs", elapsed)
        content = response.content
        if isinstance(content, dict):
            return content
        return _json.loads(content)
