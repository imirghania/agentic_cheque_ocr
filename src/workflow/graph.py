from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from src.llm.base import LlmProvider
from src.ocr.base import OcrProvider
from src.workflow.nodes import ocr_node, llm_node, resolve_keys_node
from src.workflow.state import GraphState


class ChequeReaderGraph:
    def __init__(self, ocr_provider: OcrProvider, llm_provider: LlmProvider) -> None:
        self._ocr = ocr_provider
        self._llm = llm_provider
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
        
        return builder.compile()

    def run(
        self,
        image_path: str | Path,
        user_prompt: str,
    ) -> dict:
        initial_state: GraphState = {
            "image_path": str(image_path),
            "user_prompt": user_prompt,
            "requested_keys": [],
            "ocr_text": "",
            "ocr_confidence": None,
            "extracted_data": {},
            "error": None,
        }

        result = self._graph.invoke(initial_state)

        return {
            "extracted_data": result.get("extracted_data", {}),
            "ocr_confidence": result.get("ocr_confidence"),
            "error": result.get("error"),
        }
