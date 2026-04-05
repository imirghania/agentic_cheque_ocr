from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from .base import LlmProvider


class VllmProvider(LlmProvider):
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        temperature: float = 0,
    ) -> None:
        self._llm = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=temperature,
        )

    def extract_json(self, prompt: str, schema: dict) -> dict:
        response = self._llm.invoke(prompt)
        content = response.content
        if isinstance(content, dict):
            return content
        return json.loads(content)
