from __future__ import annotations

from typing import TYPE_CHECKING

from src.llm.base import LlmProvider
from src.llm.openai_provider import OpenAiProvider   # noqa: F401
from src.llm.ollama_provider import OllamaProvider  # noqa: F401
from src.llm.vllm_provider import VllmProvider  # noqa: F401

if TYPE_CHECKING:
    from config.settings import Settings


def get_llm_provider(name: str, settings: Settings | None = None, **overrides) -> LlmProvider:
    normalized = name.lower().removesuffix("provider").removesuffix("llm")
    if normalized not in LlmProvider._registry:
        raise ValueError(f"Unknown LLM provider: {name}. Available: {list(LlmProvider._registry.keys())}")
    cls = LlmProvider._registry[normalized]
    if settings is not None:
        return cls.from_settings(settings, **overrides)
    return cls(**overrides)


def list_llm_providers() -> list[str]:
    return list(LlmProvider._registry.keys())