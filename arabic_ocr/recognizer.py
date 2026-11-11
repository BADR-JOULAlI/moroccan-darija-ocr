"""Shared bounded inference for CLI and demo; checkpoint is loaded once."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from data import collate_lines, preprocess
from evaluate import load_model
from model import greedy_decode
from page_layout import detect_line_boxes, prepare_image


class OCRRecognizer:
    def __init__(self, checkpoint: Path, device: str = "auto", batch_size: int = 16):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Download best.pt or pass --checkpoint.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: auto, cpu, cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu or auto")
        self.model, self.characters, self.config, self.epoch = load_model(checkpoint, self.device)
        self.batch_size = batch_size

    @staticmethod
    def _check_image(image: Image.Image) -> None:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds 20 million pixels; resize it before OCR")

    def recognize_lines(self, images: Iterable[Image.Image]) -> list[str]:
        predictions = []
        rows = []
        for image in images:
            self._check_image(image)
            pixels, width = preprocess(image, self.config["height"], self.config["max_width"])
            rows.append({"pixels": pixels, "width": width, "target": [], "text": "", "id": str(len(predictions) + len(rows))})
            if len(rows) == self.batch_size:
                predictions.extend(self._predict_batch(rows))
                rows = []
        if rows:
            predictions.extend(self._predict_batch(rows))
        return predictions

    def _predict_batch(self, rows: list[dict]) -> list[str]:
        batch = collate_lines(rows)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16,
                                                   enabled=self.device.type == "cuda"):
            scores, lengths = self.model(batch["images"].to(self.device), batch["widths"])
        return greedy_decode(scores, lengths, self.characters)

    def recognize_line(self, image: Image.Image) -> str:
        return self.recognize_lines([image])[0]

    def recognize_page(self, image: Image.Image) -> dict:
        self._check_image(image)
        page = prepare_image(image)
        boxes = detect_line_boxes(page)
        if len(boxes) > 200:
            raise ValueError("More than 200 candidate lines; use a smaller single-column image")
        predictions = self.recognize_lines(page.crop(box) for box in boxes)
        warnings = []
        if any((right - left) * self.config["height"] / (bottom - top) > self.config["max_width"]
               for left, top, right, bottom in boxes):
            warnings.append("Some lines exceed the checkpoint maximum width and were compressed.")
        return {"text": "\n".join(predictions), "checkpoint_epoch": self.epoch,
                "reading_order": "top-to-bottom", "coordinate_space": "EXIF-oriented image",
                "warnings": warnings,
                "lines": [{"index": i, "box": list(box), "top": box[1], "bottom": box[3], "text": text}
                          for i, (box, text) in enumerate(zip(boxes, predictions))]}
