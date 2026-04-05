from __future__ import annotations

from abc import ABC, abstractmethod


class LlmProvider(ABC):
    _registry: dict[str, type[LlmProvider]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower().removesuffix("provider").removesuffix("llm")
        LlmProvider._registry[name] = cls

    @abstractmethod
    def extract_json(self, prompt: str, schema: dict) -> dict:
        ...
