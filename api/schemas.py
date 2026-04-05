from __future__ import annotations

from pydantic import BaseModel


class ExtractResponse(BaseModel):
    extracted_data: dict
    ocr_confidence: float | None
    error: str | None
    markdown: str | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    ocr_provider: str
    llm_provider: str


class FieldsResponse(BaseModel):
    fields: list[str]


class ProvidersResponse(BaseModel):
    ocr_providers: list[str]
