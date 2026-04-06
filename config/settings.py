from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ocr_provider: str = "easyocr"
    use_gpu: bool = False
    tesseract_lang: str = "eng"
    glm_ocr_model: str = "zai-org/GLM-OCR"

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    server_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_prefix="")


settings = Settings()
