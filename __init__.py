from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from config.settings import settings
from src.llm import get_llm_provider
from src.ocr import get_ocr_provider
from src.workflow import ChequeReaderGraph


def extract_cheque(
    image_path: str | Path,
    user_prompt: str = "Extract all available fields from this cheque.",
    ocr_provider: str = "easyocr",
    llm_provider: str = "openai",
    **overrides,
) -> dict:
    load_dotenv()

    ocr = get_ocr_provider(ocr_provider, settings, **overrides)
    llm = get_llm_provider(llm_provider, settings, **overrides)

    graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)

    return graph.run(image_path=image_path, user_prompt=user_prompt)
