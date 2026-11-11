"""Recognize a clean single-column page and export JSON plus UTF-8 text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from recognizer import OCRRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, help="Optional UTF-8 .txt export")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.text_output and args.output.resolve() == args.text_output.resolve():
        parser.error("--output and --text-output must differ")
    protected = {args.image.resolve(), args.checkpoint.resolve()}
    outputs = {args.output.resolve(), args.text_output.resolve() if args.text_output else None}
    if protected & outputs:
        parser.error("Output paths must not overwrite the input image or checkpoint")
    torch.set_num_threads(4)
    recognizer = OCRRecognizer(args.checkpoint, args.device, args.batch_size)
    with Image.open(args.image) as source:
        result = recognizer.recognize_page(source)
    result["image"] = str(args.image.resolve())
    result["device"] = str(recognizer.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(result["text"], encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
