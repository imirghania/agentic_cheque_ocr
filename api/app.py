from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ExtractResponse,
    FieldsResponse,
    HealthResponse,
    ProvidersResponse,
    StatusResponse,
    SwitchResponse,
)

from api.utils import format_markdown
from config.settings import settings
from logger import logger
from src.llm import get_llm_provider
from src.ocr import get_ocr_provider, list_ocr_providers
from src.workflow.graph import ChequeReaderGraph
from src.workflow.prompt import ALLOWED_KEYS


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cheque OCR API with OCR=%s, LLM=%s",
                settings.ocr_provider, settings.llm_provider)
    ocr = get_ocr_provider(settings.ocr_provider, settings)
    llm = get_llm_provider(settings.llm_provider, settings)
    app.state.ocr_providers = {settings.ocr_provider: ocr}
    app.state.active_ocr = settings.ocr_provider
    app.state.llm = llm
    app.state.settings = settings
    logger.info("Models loaded successfully")
    yield
    logger.info("Shutting down Cheque OCR API")


app = FastAPI(title="Cheque OCR API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_load_ocr(name: str):
    if name not in app.state.ocr_providers:
        logger.info("Lazy-loading OCR provider: %s", name)
        app.state.ocr_providers[name] = get_ocr_provider(name, app.state.settings)
    app.state.active_ocr = name
    return app.state.ocr_providers[name]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=hasattr(app.state, "ocr_providers") and hasattr(app.state, "llm"),
        ocr_provider=app.state.active_ocr,
        llm_provider=app.state.settings.llm_provider,
    )


@app.get("/api/v1/cheque/available-fields", response_model=FieldsResponse)
async def available_fields():
    return FieldsResponse(fields=sorted(ALLOWED_KEYS))


@app.get("/api/v1/cheque/available-ocr-providers", response_model=ProvidersResponse)
async def available_ocr_providers():
    return ProvidersResponse(ocr_providers=list_ocr_providers())


@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    cuda = torch.cuda.is_available()
    return StatusResponse(
        device="cuda" if cuda else "cpu",
        cuda_available=cuda,
        ocr_provider=app.state.active_ocr,
        llm_provider=app.state.settings.llm_provider,
        loaded_ocr_providers=list(app.state.ocr_providers.keys()),
    )


@app.post("/api/v1/ocr/switch/{provider}", response_model=SwitchResponse)
async def switch_cr(provider: str):
    available = list_ocr_providers()
    if provider not in available:
        raise HTTPException(status_code=400, detail=f"Unknown OCR provider: {provider}. Available: {available}")
    logger.info("Switching OCR provider: %s", provider)
    ocr = _get_or_load_ocr(provider)
    cuda = torch.cuda.is_available()
    return SwitchResponse(
        provider=provider,
        device="cuda" if cuda else "cpu",
        loaded=True,
    )


@app.post("/api/v1/cheque/extract", response_model=ExtractResponse)
async def extract_cheque(
    image: UploadFile,
    prompt: str = Form(default="Extract all available fields from this cheque."),
    output_format: str = Form(default="json"),
    ocr_provider: str | None = Form(default=None),
):
    target_ocr = ocr_provider or app.state.active_ocr
    logger.info("Extract request: filename=%s, ocr=%s",
                image.filename, target_ocr)

    ocr = _get_or_load_ocr(target_ocr)
    llm = app.state.llm

    content = await image.read()
    if not content:
        logger.warning("Empty image file uploaded")
        raise HTTPException(status_code=400, detail="Empty image file")

    with NamedTemporaryFile(delete=False, 
                            suffix=Path(image.filename or "").suffix
                            ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        graph = ChequeReaderGraph(ocr_provider=ocr, llm_provider=llm)
        result = graph.run(image_path=tmp_path, user_prompt=prompt)

        response = ExtractResponse(
            extracted_data=result.get("extracted_data", {}),
            ocr_confidence=result.get("ocr_confidence"),
            error=result.get("error"),
        )

        if output_format == "markdown":
            response.markdown = format_markdown(response)

        logger.info("Extract response: error=%s", response.error)
        return response
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)
