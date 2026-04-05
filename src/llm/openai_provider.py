from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI # type: ignore

from logger import logger
from .base import LlmProvider

if TYPE_CHECKING:
    from config.settings import Settings


class OpenAiProvider(LlmProvider):
    def __init__(self, model: str = "gpt-4o", 
                api_key: str | None = None, 
                temperature: float = 0) -> None:
        kwargs: dict = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        logger.info("Initializing OpenAI LLM: model=%s, temperature=%.1f",
                     model, temperature)
        self._llm = ChatOpenAI(**kwargs)

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "OpenAiProvider":
        model = overrides.pop("model", settings.openai_model)
        api_key = overrides.pop("api_key", settings.openai_api_key)
        key_info = "set" if api_key else "not set (using env var)"
        logger.debug("OpenAiProvider.from_settings: model=%s, api_key=%s",
                     model, key_info)
        return cls(model=model, api_key=api_key, **overrides)

    def extract_json(self, prompt: str, schema: dict) -> dict:
        logger.debug("OpenAI extract_json: invoking with structured output")
        structured_llm = self._llm.with_structured_output(schema)
        t0 = time.time()
        result = structured_llm.invoke(prompt)
        elapsed = time.time() - t0
        logger.debug("OpenAI response received in %.2fs", elapsed)
        if isinstance(result, dict):
            return result
        return result.model_dump()
