# Cheque Reader

A modular bank cheque image reader that uses OCR to extract text while preserving spatial layout, then feeds it into an LLM to extract structured data as JSON.

## Architecture

```
Image + User Prompt -> Key Resolution -> OCR -> LLM Extraction -> JSON Output
```

### Components

- **OCR Layer** (`ocr/`): Pluggable OCR providers with a common `OcrProvider` interface
  - `TesseractOcr` - Tesseract-based OCR with block-level spatial data
  - `EasyOcr` - EasyOCR-based OCR with bounding box data
- **LLM Layer** (`llm/`): Pluggable LLM providers with a common `LlmProvider` interface
  - `OpenAiLlm` - OpenAI (GPT-4o) via LangChain
  - `OllamaLlm` - Local Ollama models
  - `VllmLlm` - vLLM self-hosted models via OpenAI-compatible API
- **Workflow** (`workflow/`): LangGraph graph that orchestrates resolve_keys -> OCR -> LLM extraction
- **Schema** (`schema/`): Pydantic model defining the cheque data structure

### Supported Fields

| Key | Description |
|---|---|
| `bank_name` | Name of the bank |
| `bank_branch` | Bank branch name or location |
| `bank_info` | Bank contact details (phone, fax, website, email, post_box) |
| `cheque_number` | Cheque number |
| `date` | Date on the cheque |
| `payee` | Name of the payee |
| `amount_in_words` | Amount written in words |
| `amount_in_numbers` | Amount in numeric form |
| `payer` | Name of the account holder |
| `account_number` | Payer's account number |
| `micr_code` | MICR code at the bottom of the cheque |
| `sort_code` | Bank sort code |

## Setup

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (Recommended)

```bash
uv sync
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and provider preferences
```

## Usage

### CLI

```bash
# Extract all fields
python main.py cheque.png --pretty

# Extract specific fields using natural language
python main.py cheque.png --prompt "Get the payee name, amount, and date" --pretty

# Use a different OCR provider
python main.py cheque.png --ocr-provider tesseract --pretty

# Use Ollama instead of OpenAI
python main.py cheque.png --llm-provider ollama --pretty
```

### Python API

```python
from src.workflow import ChequeReaderGraph
from src.ocr import get_ocr_provider
from src.llm import get_llm_provider

ocr = get_ocr_provider("easyocr")
llm = get_llm_provider("openai")

graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)

result = graph.run(
    image_path="cheque.png",
    user_prompt="Extract the bank name, cheque number, and amount",
)

print(result["extracted_data"])
```

Or use the convenience function:

```python
from extract_cheque import extract_cheque

result = extract_cheque(
    image_path="cheque.png",
    user_prompt="Get the payee, amount in numbers, and date",
    ocr_provider="easyocr",
    llm_provider="openai",
)
```

## Adding a New OCR Provider

1. Create a new file in `src/ocr/` (e.g., `src/ocr/my_ocr.py`)
2. Inherit from `OcrProvider` in `src/ocr/base.py`
3. Implement the `extract(self, image: Path | str) -> OcrResult` method
4. Import the class in `src/ocr/__init__.py` (auto-registration via `__init_subclass__`):

```python
from src.ocr.my_ocr import MyOcr  # noqa: F401
```

The provider will be automatically registered. The name is derived from the class name by removing `"Ocr"` (e.g., `MyOcr` → `my`).

## Adding a New LLM Provider

1. Create a new file in `src/llm/` (e.g., `src/llm/my_llm.py`)
2. Inherit from `LlmProvider` in `src/llm/base.py`
3. Implement `extract_json(self, prompt: str, schema: dict) -> dict`
4. Import the class in `src/llm/__init__.py` (auto-registration via `__init_subclass__`):

```python
from src.llm.my_llm import MyLlm  # noqa: F401
```

The provider will be automatically registered. The name is derived from the class name by removing `"Provider"` and `"Llm"` (e.g., `MyLlm` → `my`).
