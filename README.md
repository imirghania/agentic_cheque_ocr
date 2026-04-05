# Cheque Reader

A modular bank cheque image reader that uses OCR to extract text while preserving spatial layout, then feeds it into an LLM to extract structured data as JSON.

## Architecture

```
Image + User Prompt -> Key Resolution -> OCR -> LLM Extraction -> JSON Output
```

### Components

- **OCR Layer** (`src/ocr/`): Pluggable OCR providers with a common `OcrProvider` interface
  - `EasyOcr` - EasyOCR-based OCR with bounding box data
  - `TesseractOcr` - Tesseract-based OCR with block-level spatial data
- **LLM Layer** (`src/llm/`): Pluggable LLM providers with a common `LlmProvider` interface
  - `OpenAiProvider` - OpenAI (GPT-4o) via LangChain
  - `OllamaProvider` - Local Ollama models
  - `VllmProvider` - vLLM self-hosted models via OpenAI-compatible API
- **Workflow** (`src/workflow/`): LangGraph graph that orchestrates resolve_keys -> OCR -> LLM extraction
- **Schema** (`src/schema/`): Pydantic model defining the cheque data structure

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

```bash
uv sync
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and provider preferences
```

## Usage

### 1. CLI (Local Mode)

Runs OCR and LLM models directly on your machine. Requires GPU for best OCR performance.

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

CLI options:
| Flag | Description |
|---|---|
| `--ocr-provider` | OCR engine: `easyocr` (default) or `tesseract` |
| `--llm-provider` | LLM engine: `openai` (default), `ollama`, or `vllm` |
| `--prompt` | Natural language description of fields to extract |
| `--pretty` | Pretty-print JSON output |
| `--output-format` | `json` (default) or `markdown` |

### 2. CLI (Server Mode)

Sends the image to a running API server for processing. Useful when the server has dedicated GPU resources.

```bash
# Start the server first (see API section below)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Then use CLI in server mode
python main.py cheque.png --server --pretty
```

The `--server` flag reads `server_url` from `.env` (default: `http://localhost:8000`).

### 3. FastAPI Server

Run a persistent API server with models pre-loaded. The server caches OCR and LLM instances on startup, avoiding repeated model loading.

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health and loaded providers |
| `GET` | `/api/v1/cheque/available-fields` | List of extractable fields |
| `GET` | `/api/v1/cheque/available-ocr-providers` | List of available OCR engines |
| `POST` | `/api/v1/cheque/extract` | Upload a cheque image and get structured JSON |

**Extract endpoint:**

```bash
curl -X POST http://localhost:8000/api/v1/cheque/extract \
  -F "image=@cheque.png" \
  -F "prompt=Extract all fields" \
  -F "output_format=json" \
  -F "ocr_provider=easyocr" \
  -F "use_gpu=true"
```

### 4. Streamlit UI

A web interface that connects to the running API server. Requires the server to be running first.

```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# In another terminal, start the Streamlit app
streamlit run streamlit_app.py
```

The UI lets you:
- Upload cheque images via drag-and-drop
- Select OCR provider and GPU usage from the sidebar
- Write custom extraction prompts
- View results as JSON or formatted markdown tables
- Download results as markdown files

### 5. Python API

```python
from src.workflow import ChequeReaderGraph
from src.ocr import get_ocr_provider
from src.llm import get_llm_provider
from config.settings import settings

ocr = get_ocr_provider("easyocr", settings)
llm = get_llm_provider("openai", settings)

graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)

result = graph.run(
    image_path="cheque.png",
    user_prompt="Extract the bank name, cheque number, and amount",
)

print(result["extracted_data"])
```

Or use the top-level convenience function:

```python
from agentic_cheque_ocr import extract_cheque

result = extract_cheque(
    image_path="cheque.png",
    user_prompt="Get the payee, amount in numbers, and date",
)
```

## Adding a New Provider

### OCR Provider

1. Create `src/ocr/my_ocr.py` inheriting from `OcrProvider`
2. Implement `from_settings(cls, settings, **overrides)` and `extract(self, image) -> OcrResult`
3. Import in `src/ocr/__init__.py`:

```python
from src.ocr.my_ocr import MyOcr  # noqa: F401
```

Auto-registered as `my` (class name minus `"Ocr"` suffix).

### LLM Provider

1. Create `src/llm/my_llm.py` inheriting from `LlmProvider`
2. Implement `from_settings(cls, settings, **overrides)` and `extract_json(self, prompt, schema) -> dict`
3. Import in `src/llm/__init__.py`:

```python
from src.llm.my_llm import MyLlm  # noqa: F401
```

Auto-registered as `my` (class name minus `"Provider"` and `"Llm"` suffixes).

## Logging

Structured logging is enabled by default. Logs go to:

| Destination | Level | Format |
|---|---|---|
| stdout | INFO+ | Human-readable |
| stderr | WARNING+ | Human-readable |
| `logs/cheque_ocr.log` | DEBUG+ | JSON (rotating, 5MB max, 5 backups) |

Each pipeline step is logged: provider initialization, key resolution, OCR extraction, LLM extraction, and total execution time.
