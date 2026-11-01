"""Generate deterministic Darija line images and CRNN manifests from text.

Input can be JSONL (``{"text": "..."}``), CSV with a ``text`` column, or a
plain UTF-8 file with one sentence per line. This is augmentation/pretraining,
not a replacement for real annotated documents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def read_texts(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        values = []
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                values.append(str(row.get("text", "")))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            values = [str(row.get("text", "")) for row in csv.DictReader(stream)]
    else:
        values = path.read_text(encoding="utf-8-sig").splitlines()
    return [" ".join(unicodedata.normalize("NFC", text).split()) for text in values if text.strip()]


def shape_text(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def render_line(text: str, font_path: Path, size: int, rng: random.Random) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size=size)
    display_text = shape_text(text)
    probe = Image.new("L", (16, 16), 255)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), display_text, font=font, stroke_width=0)
    padding_x, padding_y = 18, 12
    canvas = Image.new("L", (max(32, right - left + 2 * padding_x), bottom - top + 2 * padding_y), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding_x - left, padding_y - top), display_text, font=font, fill=rng.randint(0, 25))
    if rng.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.25))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--font-size", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    fonts = [font for font in args.fonts if font.is_file()]
    if not fonts:
        raise FileNotFoundError("No font files found")
    texts = read_texts(args.texts)
    if not texts:
        raise ValueError("No non-empty texts found")
    rng = random.Random(args.seed)
    rows = []
    for index, text in enumerate(texts):
        font = fonts[index % len(fonts)]
        image = render_line(text, font, args.font_size, rng)
        identifier = hashlib.sha1(f"{args.seed}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
        target = args.output_dir / f"{identifier}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, optimize=True)
        rows.append({"id": identifier, "image": str(target.resolve()), "text": text, "font": font.name, "synthetic": True})
    rng.shuffle(rows)
    split_at = int(len(rows) * (1 - args.val_ratio))
    splits = {"train": rows[:split_at], "val": rows[split_at:] or rows[-1:]}
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with (args.output_data_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {"samples": len(rows), "train": len(splits["train"]), "val": len(splits["val"]), "seed": args.seed, "fonts": [str(font) for font in fonts], "source": str(args.texts.resolve())}
    (args.output_data_dir / "synthetic_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
