from __future__ import annotations

from langchain_openai import ChatOpenAI # type: ignore

from .base import LlmProvider


class OpenAiProvider(LlmProvider):
    def __init__(self, model: str = "gpt-4o", 
                api_key: str | None = None, 
                temperature: float = 0) -> None:
        kwargs: dict = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        self._llm = ChatOpenAI(**kwargs)

    def extract_json(self, prompt: str, schema: dict) -> dict:
        structured_llm = self._llm.with_structured_output(schema)
        result = structured_llm.invoke(prompt)
        if isinstance(result, dict):
            return result
        return result.model_dump()
