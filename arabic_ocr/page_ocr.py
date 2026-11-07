"""Run line segmentation followed by CRNN-CTC recognition on one page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data import collate_lines, preprocess
from evaluate import load_model
from model import greedy_decode
from segment_page_lines import find_line_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, characters, config, epoch = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        page = source.convert("L")
        boxes = find_line_boxes(page)
        rows = []
        for index, (top, bottom) in enumerate(boxes):
            crop = page.crop((0, top, page.width, bottom))
            pixels, width = preprocess(crop, config["height"], config["max_width"])
            rows.append({"pixels": pixels, "width": width, "target": [], "text": "", "id": f"line_{index:04d}"})
    if not rows:
        raise ValueError("no text lines detected")
    batch = collate_lines(rows)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        log_probs, lengths = model(batch["images"].to(device), batch["widths"])
    predictions = greedy_decode(log_probs, lengths, characters)
    result = {"image": str(args.image.resolve()), "checkpoint_epoch": epoch,
              "lines": [{"index": i, "top": boxes[i][0], "bottom": boxes[i][1], "text": text}
                        for i, text in enumerate(predictions)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
