from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict):
    image_path: str
    user_prompt: str
    requested_keys: list[str]
    ocr_text: str
    ocr_confidence: float | None
    extracted_data: dict
    error: str | None
