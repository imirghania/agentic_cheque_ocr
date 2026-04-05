from __future__ import annotations

from src.ocr.base import OcrProvider
from src.ocr.tesseract import TesseractOcr  # noqa: F401
from src.ocr.easyocr import EasyOcr  # noqa: F401


def get_ocr_provider(name: str, **kwargs) -> OcrProvider:
    normalized = name.lower().removesuffix("ocr")
    if normalized not in OcrProvider._registry:
        raise ValueError(f"Unknown OCR provider: {name}. Available: {list(OcrProvider._registry.keys())}")
    return OcrProvider._registry[normalized](**kwargs)


def list_ocr_providers() -> list[str]:
    return list(OcrProvider._registry.keys())
