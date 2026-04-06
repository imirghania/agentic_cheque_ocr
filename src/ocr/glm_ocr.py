from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from logger import logger
from src.ocr.base import OcrProvider, OcrResult

if TYPE_CHECKING:
    from config.settings import Settings


class GlmOcr(OcrProvider):
    def __init__(
        self,
        model_name: str = "zai-org/GLM-OCR",
        prompt: str = "Text Recognition:",
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        logger.info("Initializing GLM-OCR: model=%s", self.model_name)

        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        logger.debug("GLM-OCR processor loaded in %.1fs", time.time() - t0)

        t0 = time.time()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.debug("GLM-OCR model loaded in %.1fs", time.time() - t0)

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "GlmOcr":
        model_name = overrides.pop("model_name", settings.glm_ocr_model)
        return cls(model_name=model_name, **overrides)

    def extract(self, image: Path | str) -> OcrResult:
        image_path = image if isinstance(image, Path) else Path(image)
        logger.debug("GLM-OCR extracting from: %s", image_path)

        pil_image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=8192
            )
        output_text = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        logger.debug("GLM-OCR generation completed in %.2fs", time.time() - t0)

        return OcrResult(text=output_text.strip(), confidence=None, blocks=[])
