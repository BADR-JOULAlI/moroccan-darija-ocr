"""Conservatively crop page images when text supplies one transcription per line.

The tool refuses paragraphs without explicit line breaks. It uses horizontal ink
projections only for the crop boundaries; it never invents an alignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def find_line_boxes(image: Image.Image, threshold: int = 245, gap: int = 2, pad: int = 3):
    gray = np.asarray(ImageOps.grayscale(image))
    ink = (gray < threshold).any(axis=1)
    runs = []
    start = None
    empty = 0
    for index, present in enumerate(ink):
        if present:
            if start is None:
                start = max(0, index - pad)
            empty = 0
        elif start is not None:
            empty += 1
            if empty > gap:
                runs.append((start, min(gray.shape[0], index - empty + 1 + pad)))
                start = None
                empty = 0
    if start is not None:
        runs.append((start, gray.shape[0]))
    return runs


def segment_record(record: dict, input_root: Path, output_root: Path) -> list[dict]:
    text = record.get("text")
    if not isinstance(text, str) or "\n" not in text and "\r" not in text:
        raise ValueError(f"{record.get('id', '<unknown>')}: explicit line breaks are required")
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    source = Path(record["image"])
    if not source.is_absolute():
        source = (input_root / source).resolve()
    with Image.open(source) as image:
        image = image.convert("L")
        boxes = find_line_boxes(image)
        if len(boxes) != len(lines):
            raise ValueError(f"{record.get('id', source.name)}: detected {len(boxes)} lines for {len(lines)} transcriptions")
        output_rows = []
        stem = str(record.get("id") or source.stem)
        for index, (top, bottom) in enumerate(boxes):
            crop = image.crop((0, top, image.width, bottom))
            target = output_root / f"{stem}_{index:04d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            output_rows.append({"id": f"{stem}_{index:04d}", "image": str(target.resolve()), "text": lines[index]})
        return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with page image and newline-separated text")
    parser.add_argument("--input-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line_no, raw in enumerate(args.input.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.extend(segment_record(json.loads(raw), args.input_root, args.output_dir))
        except Exception as error:
            raise ValueError(f"{args.input}:{line_no}: {error}") from error
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"pages": len(set(row["id"].rsplit("_", 1)[0] for row in rows)), "lines": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
