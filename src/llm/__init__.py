from __future__ import annotations

from src.llm.base import LlmProvider
from src.llm.openai_provider import OpenAiProvider   # noqa: F401
from src.llm.ollama_provider import OllamaProvider  # noqa: F401
from src.llm.vllm_provider import VllmProvider  # noqa: F401


def get_llm_provider(name: str, **kwargs) -> LlmProvider:
    normalized = name.lower().removesuffix("provider").removesuffix("llm")
    if normalized not in LlmProvider._registry:
        raise ValueError(f"Unknown LLM provider: {name}. Available: {list(LlmProvider._registry.keys())}")
    return LlmProvider._registry[normalized](**kwargs)


def list_llm_providers() -> list[str]:
    return list(LlmProvider._registry.keys())