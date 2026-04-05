from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

from cli_client import call_server
from config.settings import settings
from logger import setup_logging, logger
from src.llm import get_llm_provider
from src.ocr import get_ocr_provider
from src.workflow import ChequeReaderGraph


def main() -> None:
    load_dotenv()
    setup_logging()

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
    parser.add_argument(
        "--server",
        action="store_true",
        help="Use the running API server instead of loading models locally",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        logger.error("Image not found: %s", args.image)
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.server:
        logger.info("Using server mode: %s", settings.server_url)
        result = call_server(
            image_path=args.image,
            prompt=args.prompt,
            output_format=args.output_format,
            server_url=settings.server_url,
        )
        if args.output_format == "markdown" and result.get("markdown"):
            print(result["markdown"])
        else:
            indent = 2 if args.pretty else None
            print(json.dumps(result, indent=indent, default=str))
        return

    ocr_provider_name = args.ocr_provider or settings.ocr_provider
    llm_provider_name = args.llm_provider or settings.llm_provider

    logger.info("Selected providers: OCR=%s, LLM=%s", ocr_provider_name, llm_provider_name)

    ocr = get_ocr_provider(ocr_provider_name, settings)
    llm = get_llm_provider(llm_provider_name, settings)

    graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)

    result = graph.run(image_path=args.image, user_prompt=args.prompt)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
