from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

from src.ocr import get_ocr_provider
from src.llm import get_llm_provider
from src.workflow import ChequeReaderGraph


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract structured data from bank cheque images")
    parser.add_argument("image", type=str, help="Path to the cheque image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract all available fields from this cheque.",
        help="Natural language description of what to extract (default: extract all fields)",
    )
    parser.add_argument(
        "--ocr-provider",
        type=str,
        default=None,
        help="OCR provider to use (default: from .env or 'easyocr')",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider to use (default: from .env or 'openai')",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    import os

    ocr_provider_name = args.ocr_provider or os.getenv("OCR_PROVIDER", "easyocr")
    llm_provider_name = args.llm_provider or os.getenv("LLM_PROVIDER", "openai")

    ocr_kwargs = {}
    if ocr_provider_name == "easyocr":
        ocr_kwargs["gpu"] = os.getenv("EASYOCR_GPU", "false").lower() == "true"
    elif ocr_provider_name == "tesseract":
        ocr_kwargs["lang"] = os.getenv("TESSERACT_LANG", "eng")

    llm_kwargs = {}
    if llm_provider_name == "openai":
        llm_kwargs["model"] = os.getenv("OPENAI_MODEL", "gpt-4o")
        if os.getenv("OPENAI_API_KEY"):
            llm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    elif llm_provider_name == "ollama":
        llm_kwargs["model"] = os.getenv("OLLAMA_MODEL", "llama3.1")
        llm_kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif llm_provider_name == "vllm":
        llm_kwargs["model"] = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        llm_kwargs["base_url"] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    ocr = get_ocr_provider(ocr_provider_name, **ocr_kwargs)
    llm = get_llm_provider(llm_provider_name, **llm_kwargs)

    graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)

    result = graph.run(image_path=args.image, user_prompt=args.prompt)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
