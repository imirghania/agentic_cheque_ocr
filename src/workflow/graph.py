from __future__ import annotations

import time
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from logger import logger
from src.llm.base import LlmProvider
from src.ocr.base import OcrProvider
from src.workflow.nodes import ocr_node, llm_node, resolve_keys_node
from src.workflow.state import GraphState


class ChequeReaderGraph:
    def __init__(self, ocr_provider: OcrProvider, llm_provider: LlmProvider) -> None:
        self._ocr = ocr_provider
        self._llm = llm_provider
        ocr_name = type(ocr_provider).__name__
        llm_name = type(llm_provider).__name__
        logger.info("ChequeReaderGraph initialized with OCR=%s, LLM=%s", ocr_name, llm_name)
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        
        builder.add_node("resolve_keys", lambda s: resolve_keys_node(s, self._llm))
        builder.add_node("ocr", lambda s: ocr_node(s, self._ocr))
        builder.add_node("llm", lambda s: llm_node(s, self._llm))
        
        builder.add_edge(START, "resolve_keys")
        builder.add_edge("resolve_keys", "ocr")
        builder.add_edge("ocr", "llm")
        builder.add_edge("llm", END)
        
        logger.debug("Pipeline graph built: START -> resolve_keys -> ocr -> llm -> END")
        return builder.compile()

    def run(
        self,
        image_path: str | Path,
        user_prompt: str,
    ) -> dict:
        image_path_str = str(image_path)
        logger.info("Starting cheque extraction: image=%s, prompt=%r", image_path_str, user_prompt)
        start_time = time.time()

        initial_state: GraphState = {
            "image_path": image_path_str,
            "user_prompt": user_prompt,
            "requested_keys": [],
            "ocr_text": "",
            "ocr_confidence": None,
            "extracted_data": {},
            "error": None,
        }

        result = self._graph.invoke(initial_state)

        elapsed = time.time() - start_time
        error = result.get("error")
        if error:
            logger.error("Extraction failed after %.2fs: %s", elapsed, error)
        else:
            logger.info("Extraction completed successfully in %.2fs", elapsed)

        return {
            "extracted_data": result.get("extracted_data", {}),
            "ocr_confidence": result.get("ocr_confidence"),
            "error": result.get("error"),
        }
