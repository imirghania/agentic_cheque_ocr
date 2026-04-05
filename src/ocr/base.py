from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None
    blocks: list[dict] | None = None


class OcrProvider(ABC):
    _registry: dict[str, type[OcrProvider]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower().removesuffix("ocr")
        OcrProvider._registry[name] = cls

    @abstractmethod
    def extract(self, image: Path | str) -> OcrResult:
        ...
