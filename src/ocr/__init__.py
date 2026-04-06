from __future__ import annotations

from typing import TYPE_CHECKING

from src.ocr.base import OcrProvider
from src.ocr.tesseract import TesseractOcr  # noqa: F401
from src.ocr.easyocr import EasyOcr  # noqa: F401
from src.ocr.glm_ocr import GlmOcr  # noqa: F401

if TYPE_CHECKING:
    from config.settings import Settings


def get_ocr_provider(name: str, 
                    settings: Settings | None = None, 
                    **overrides) -> OcrProvider:
    normalized = name.lower().removesuffix("ocr")
    if normalized not in OcrProvider._registry:
        raise ValueError(f"Unknown OCR provider: {name}. Available: {list(OcrProvider._registry.keys())}")
    cls = OcrProvider._registry[normalized]
    if settings is not None:
        return cls.from_settings(settings, **overrides)
    return cls(**overrides)


def list_ocr_providers() -> list[str]:
    return list(OcrProvider._registry.keys())
