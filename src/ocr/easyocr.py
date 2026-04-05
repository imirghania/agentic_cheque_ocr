from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import easyocr  #type: ignore
import torch    #type: ignore
from transformers import LayoutLMv3ForTokenClassification   #type: ignore

from logger import logger
from src.ocr.base import OcrProvider, OcrResult
from src.ocr.layoutreader_helpers import boxes2inputs, parse_logits, prepare_inputs

if TYPE_CHECKING:
    from config.settings import Settings


class EasyOcr(OcrProvider):
    def __init__(
        self,
        langs: list[str] | None = None,
        gpu: bool = False,
        model_storage_directory: str | None = None,
        layoutreader_model: str = "hantian/layoutreader",
    ) -> None:
        self.langs = langs or ["en"]
        self.gpu = gpu
        logger.info("Initializing EasyOCR: langs=%s, gpu=%s", self.langs, self.gpu)
        t0 = time.time()
        self._reader = easyocr.Reader(
            self.langs,
            gpu=self.gpu,
            model_storage_directory=model_storage_directory,
        )
        logger.debug("EasyOCR reader loaded in %.1fs", time.time() - t0)

        t0 = time.time()
        self._layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
            layoutreader_model
        )
        self._layout_model.eval()
        if self.gpu:
            self._layout_model = self._layout_model.to("cuda")
        logger.debug("LayoutLMv3 model loaded in %.1fs", time.time() - t0)

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides) -> "EasyOcr":
        gpu = overrides.pop("gpu", settings.use_gpu)
        logger.debug("EasyOcr.from_settings: gpu=%s", gpu)
        return cls(gpu=gpu, **overrides)

    def extract(self, image: Path | str) -> OcrResult:
        image_path = str(image if isinstance(image, Path) else Path(image))
        logger.debug("EasyOCR extracting from: %s", image_path)

        t0 = time.time()
        results = self._reader.readtext(image_path)
        logger.debug("EasyOCR readtext completed in %.2fs, found %d text regions",
                     time.time() - t0, len(results))

        blocks: list[dict] = []
        for bbox, text, conf in results:
            blocks.append(
                {
                    "text": text,
                    "conf": round(conf * 100, 2),
                    "bbox": bbox,
                }
            )

        blocks = self._reorder_blocks(blocks)

        full_text = "\n".join(block["text"] for block in blocks)

        avg_conf = None
        if blocks:
            avg_conf = sum(b["conf"] for b in blocks) / len(blocks)

        logger.debug("EasyOCR result: %d blocks, avg_confidence=%.1f%%",
                     len(blocks), avg_conf or 0)
        return OcrResult(text=full_text, confidence=avg_conf, blocks=blocks)

    def _reorder_blocks(self, blocks: list[dict]) -> list[dict]:
        if not blocks:
            return blocks

        boxes = []
        for block in blocks:
            bbox = block["bbox"]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            left, top, right, bottom = (
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            )
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
