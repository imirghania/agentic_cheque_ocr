from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytesseract
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification

from logger import logger
from src.ocr.base import OcrProvider, OcrResult
from src.ocr.layoutreader_helpers import boxes2inputs, parse_logits, prepare_inputs

if TYPE_CHECKING:
    from config.settings import Settings


class TesseractOcr(OcrProvider):
    def __init__(
        self,
        lang: str = "eng",
        config: str | None = None,
        gpu: bool = False,
        layoutreader_model: str = "hantian/layoutreader",
    ) -> None:
        self.lang = lang
        self.config = config or "--psm 6"
        self.gpu = gpu
        logger.info("Initializing TesseractOCR: lang=%s, config=%s, gpu=%s",
                     self.lang, self.config, self.gpu)

        t0 = time.time()
        self._layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
            layoutreader_model
        )
        self._layout_model.eval()
        if self.gpu:
            self._layout_model = self._layout_model.to("cuda")
        logger.debug("LayoutLMv3 model loaded in %.1fs", time.time() - t0)

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "TesseractOcr":
        gpu = overrides.pop("gpu", False)
        logger.debug("TesseractOcr.from_settings: lang=%s, gpu=%s",
                     settings.tesseract_lang, gpu)
        return cls(lang=settings.tesseract_lang, gpu=gpu, **overrides)

    def extract(self, image: Path | str) -> OcrResult:
        image_path = image if isinstance(image, Path) else Path(image)
        logger.debug("TesseractOCR extracting from: %s", image_path)

        pil_image = Image.open(image_path)

        t0 = time.time()
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        logger.debug("Tesseract image_to_data completed in %.2fs", time.time() - t0)

        blocks: list[dict] = []
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                blocks.append(
                    {
                        "text": data["text"][i].strip(),
                        "conf": int(data["conf"][i]),
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    }
                )

        logger.debug("Tesseract found %d non-empty text blocks", len(blocks))

        blocks = self._reorder_blocks(blocks)

        full_text = "\n".join(block["text"] for block in blocks)

        avg_conf = None
        confs = [b["conf"] for b in blocks if b["conf"] >= 0]
        if confs:
            avg_conf = sum(confs) / len(confs)

        logger.debug("TesseractOCR result: %d blocks, avg_confidence=%.1f%%",
                     len(blocks), avg_conf or 0)
        return OcrResult(text=full_text, confidence=avg_conf, blocks=blocks)

    def _reorder_blocks(self, blocks: list[dict]) -> list[dict]:
        if not blocks:
            return blocks

        boxes = []
        for block in blocks:
            left = block["left"]
            top = block["top"]
            right = left + block["width"]
            bottom = top + block["height"]
            boxes.append([left, top, right, bottom])

        max_dim = max(v for box in boxes for v in box)
        if max_dim > 0:
            boxes = [[int(v / max_dim * 1000) for v in box] for box in boxes]

        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, self._layout_model)
        with torch.no_grad():
            logits = self._layout_model(**inputs).logits.cpu().squeeze(0)
        orders = parse_logits(logits, len(boxes))

        return [blocks[i] for i in orders]
