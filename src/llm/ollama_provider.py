from __future__ import annotations

from langchain_ollama import ChatOllama

from .base import LlmProvider


class OllamaProvider(LlmProvider):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0,
    ) -> None:
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )

    def extract_json(self, prompt: str, schema: dict) -> dict:
        import json as _json

        response = self._llm.invoke(prompt)
        content = response.content
        if isinstance(content, dict):
            return content
        return _json.loads(content)
