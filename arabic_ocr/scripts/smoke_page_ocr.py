"""Compose three existing lines and test page inference with a real checkpoint.

This is a reproducible functional smoke test, not a page benchmark. References
are used only to report fixture errors, never to generate predicted labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognizer import OCRRecognizer
from metrics import Metrics
from page_layout import prepare_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    with args.manifest.open(encoding="utf-8-sig") as stream:
        for raw in stream:
            if raw.strip():
                rows.append(json.loads(raw))
            if len(rows) == 3:
                break
    if len(rows) != 3:
        parser.error("The manifest must contain at least three line images")
    images = []
    for row in rows:
        path = Path(row["image"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        with Image.open(path) as source:
            line = prepare_image(source)
            width = min(1600, max(32, round(line.width * 64 / line.height)))
            images.append(line.resize((width, 64), Image.Resampling.BILINEAR))
    width = max(line.width for line in images) + 48
    page = Image.new("L", (width, 3 * 100 + 24), 255)
    for index, line in enumerate(images):
        page.paste(line, (width - line.width - 24, index * 100 + 24))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page.save(args.output_dir / "page.png")
    engine = OCRRecognizer(args.checkpoint, args.device, batch_size=2)
    result = engine.recognize_page(page)
    overlay = page.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for row in result["lines"]:
        draw.rectangle(row["box"], outline="#135fbc", width=2)
    overlay.save(args.output_dir / "detected_lines.png")
    metrics = Metrics()
    references = "\n".join(row["text"] for row in rows)
    metrics.update(references, result["text"])
    report = {"scope": "three composited line images; functional smoke test only",
              "device": str(engine.device), "checkpoint_epoch": engine.epoch,
              "expected_lines": 3, "detected_lines": len(result["lines"]),
              "line_count_matches": len(result["lines"]) == 3,
              "metrics_on_fixture": metrics.compute(),
              "references": references, "result": result}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("references", "result")}, indent=2))
    if not report["line_count_matches"]:
        raise SystemExit("Unexpected line count; inspect detected_lines.png")


if __name__ == "__main__":
    main()
