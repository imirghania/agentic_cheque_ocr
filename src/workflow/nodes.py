from __future__ import annotations

from pathlib import Path

from src.llm.base import LlmProvider
from src.ocr.base import OcrProvider
from src.workflow.prompt import (
    ALLOWED_KEYS,
    build_dynamic_schema,
    build_extraction_prompt,
    build_resolve_keys_prompt,
)
from src.workflow.state import GraphState


def resolve_keys_node(state: GraphState, llm_provider: LlmProvider) -> GraphState:
    user_prompt = state["user_prompt"]

    if not user_prompt:
        return {**state, "error": "No user prompt provided"}

    prompt = build_resolve_keys_prompt(list(ALLOWED_KEYS))

    resolve_schema = {
        "type": "object",
        "title": "ResolveKeys",
        "properties": {
            "requested_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of keys to extract from the cheque",
            },
        },
        "required": ["requested_keys"],
        "additionalProperties": False,
    }

    try:
        result = llm_provider.extract_json(prompt, resolve_schema)
        requested_keys = result.get("requested_keys", [])
        if not requested_keys:
            requested_keys = list(ALLOWED_KEYS)
        invalid = set(requested_keys) - ALLOWED_KEYS
        if invalid:
            return {**state, "error": f"LLM resolved to invalid keys: {invalid}"}
        return {**state, "requested_keys": requested_keys}
    except Exception as e:
        return {**state, "requested_keys": list(ALLOWED_KEYS)}


def ocr_node(state: GraphState, ocr_provider: OcrProvider) -> GraphState:
    image_path = state["image_path"]

    if not Path(image_path).exists():
        return {**state, "error": f"Image not found: {image_path}"}

    result = ocr_provider.extract(image_path)

    return {
        **state,
        "ocr_text": result.text,
        "ocr_confidence": result.confidence,
    }


def llm_node(state: GraphState, llm_provider: LlmProvider) -> GraphState:
    ocr_text = state["ocr_text"]
    requested_keys = state["requested_keys"]

    if not ocr_text:
        return {**state, "error": "No OCR text available"}

    schema = build_dynamic_schema(requested_keys)
    prompt = build_extraction_prompt(ocr_text, requested_keys)

    try:
        result = llm_provider.extract_json(prompt, schema)
        return {**state, "extracted_data": result}
    except Exception as e:
        return {**state, "error": f"LLM extraction failed: {e}"}
