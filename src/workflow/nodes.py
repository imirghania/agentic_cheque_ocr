from __future__ import annotations

import logging
from pathlib import Path

from logger import logger
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
        logger.warning("No user prompt provided, cannot resolve keys")
        return {"error": "No user prompt provided"}

    logger.info("Resolving requested keys from user prompt")
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
            logger.warning("LLM returned empty key list, falling back to all keys")
            requested_keys = list(ALLOWED_KEYS)
        invalid = set(requested_keys) - ALLOWED_KEYS
        if invalid:
            logger.error("LLM resolved to invalid keys: %s", invalid)
            return {**state, "error": f"LLM resolved to invalid keys: {invalid}"}
        logger.info("Resolved %d keys: %s", len(requested_keys), requested_keys)
        return {"requested_keys": requested_keys}
    except Exception as e:
        logger.error("Failed to resolve keys, falling back to all keys: %s", e)
        return {"requested_keys": list(ALLOWED_KEYS)}


def ocr_node(state: GraphState, ocr_provider: OcrProvider) -> GraphState:
    image_path = state["image_path"]

    if not Path(image_path).exists():
        logger.error("Image file not found: %s", image_path)
        return {"error": f"Image not found: {image_path}"}

    ocr_name = type(ocr_provider).__name__
    logger.info("Running OCR extraction with %s on %s", ocr_name, image_path)

    result = ocr_provider.extract(image_path)

    logger.info(
        "OCR completed: %d text blocks, confidence=%.1f%%, text_length=%d",
        len(result.blocks or []),
        result.confidence or 0,
        len(result.text),
    )

    return {
        "ocr_text": result.text,
        "ocr_confidence": result.confidence,
    }


def llm_node(state: GraphState, llm_provider: LlmProvider) -> GraphState:
    ocr_text = state["ocr_text"]
    requested_keys = state["requested_keys"]

    if not ocr_text:
        logger.error("No OCR text available for LLM extraction")
        return {**state, "error": "No OCR text available"}

    llm_name = type(llm_provider).__name__
    logger.info(
        "Running LLM extraction with %s for %d keys",
        llm_name,
        len(requested_keys),
    )

    schema = build_dynamic_schema(requested_keys)
    prompt = build_extraction_prompt(ocr_text, requested_keys)

    try:
        result = llm_provider.extract_json(prompt, schema)
        extracted_fields = [k for k, v in result.items() if v is not None]
        logger.info(
            "LLM extraction completed: %d/%d fields populated (%s)",
            len(extracted_fields),
            len(requested_keys),
            extracted_fields,
        )
        return {"extracted_data": result}
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return {"error": f"LLM extraction failed: {e}"}
