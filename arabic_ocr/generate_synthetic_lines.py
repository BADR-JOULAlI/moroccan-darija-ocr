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

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features


def read_records(path: Path, text_column: str = "text") -> list[dict]:
    """Keep source-page provenance while normalizing and deduplicating text."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        values = []
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
                values.append(row)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            values = list(csv.DictReader(stream))
    else:
        values = [{text_column: text} for text in path.read_text(encoding="utf-8-sig").splitlines()]
    output, seen = [], set()
    for row in values:
        value = row.get(text_column)
        if value is None or not str(value).strip():
            continue
        if not isinstance(value, str):
            raise ValueError(f"{text_column} must contain strings")
        text = " ".join(unicodedata.normalize("NFC", value).split())
        if text in seen:
            continue
        seen.add(text)
        record = {"text": text}
        for key in ("source_dataset", "source_split", "source_row_idx", "source_group"):
            if key in row:
                record[key] = row[key]
        if row.get("id") is not None:
            record["source_id"] = str(row["id"])
        if record.get("source_row_idx") is not None:
            group = [record.get("source_dataset", str(path.resolve())),
                     record.get("source_split", ""), record["source_row_idx"]]
        elif record.get("source_group") is not None:
            group = [record.get("source_dataset", str(path.resolve())), record["source_group"]]
        else:
            group = ["text", text]
        record["split_group"] = hashlib.sha256(json.dumps(group, ensure_ascii=False).encode("utf-8")).hexdigest()
        output.append(record)
    return output


def read_texts(path: Path, text_column: str = "text") -> list[str]:
    """Compatibility API for callers that need only normalized unique text."""
    return [row["text"] for row in read_records(path, text_column)]


def split_records(records: list[dict], val_ratio: float, seed: int) -> dict[str, list[dict]]:
    """Split source groups before rendering, independently of augmentation RNG."""
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not records:
        raise ValueError("No non-empty texts found")
    groups = sorted({row["split_group"] for row in records})
    if val_ratio == 0:
        return {"train": list(records), "val": []}
    if len(groups) < 2:
        raise ValueError("At least two independent source groups are required for validation")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
    val_groups = set(groups[:count])
    return {"train": [row for row in records if row["split_group"] not in val_groups],
            "val": [row for row in records if row["split_group"] in val_groups]}


def shape_text(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaper = arabic_reshaper.ArabicReshaper(configuration={
            "delete_harakat": False, "shift_harakat_position": True,
        })
        return get_display(reshaper.reshape(text))
    except ImportError as error:
        raise RuntimeError("Arabic rendering requires Pillow RAQM or arabic-reshaper and python-bidi") from error


def render_line(text: str, font_path: Path, size: int, rng: random.Random, augment: bool = False) -> Image.Image:
    # RAQM shapes logical Unicode once and preserves vowel marks. Reshaping a
    # string before passing it to RAQM would apply bidi/shaping twice.
    raqm = features.check_feature("raqm")
    font = ImageFont.truetype(str(font_path), size=size,
                             layout_engine=ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC)
    display_text = text if raqm else shape_text(text)
    layout = {"direction": "rtl"} if raqm else {}
    probe = Image.new("L", (16, 16), 255)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), display_text, font=font, stroke_width=0, **layout)
    padding_x, padding_y = 18, 12
    canvas = Image.new("L", (max(32, right - left + 2 * padding_x), bottom - top + 2 * padding_y), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding_x - left, padding_y - top), display_text, font=font, fill=rng.randint(0, 25), **layout)
    if augment:
        if rng.random() < 0.45:
            canvas = canvas.rotate(rng.uniform(-0.55, 0.55), resample=Image.Resampling.BILINEAR,
                                  expand=True, fillcolor=255)
        if rng.random() < 0.5:
            canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.9, 1.1))
        if rng.random() < 0.25:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.25)))
        if rng.random() < 0.3:
            pixels = np.asarray(canvas, dtype=np.float32)
            noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 1.5, pixels.shape)
            canvas = Image.fromarray(np.uint8(np.clip(pixels + noise, 0, 255)), mode="L")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--text-column", default="text", help="CSV/JSONL field containing the target text")
    parser.add_argument("--fonts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="maximum number of texts (0 = all)")
    parser.add_argument("--augment", action="store_true", help="apply mild scan/photo-like augmentations")
    parser.add_argument("--font-size", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    fonts = [font for font in args.fonts if font.is_file()]
    if not fonts:
        raise FileNotFoundError("No font files found")
    records = read_records(args.texts, args.text_column)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.limit:
        records = records[: args.limit]
    splits = split_records(records, args.val_ratio, args.seed)
    indices = {row["text"]: index for index, row in enumerate(records)}
    rendered = {"train": [], "val": []}
    for split, split_rows in splits.items():
        for record in split_rows:
            index, text = indices[record["text"]], record["text"]
            font = fonts[index % len(fonts)]
            # Each image has its own RNG; render changes cannot alter splits or
            # the random choices made for any subsequent image.
            identifier = hashlib.sha256(f"{args.seed}:{text}".encode("utf-8")).hexdigest()[:20]
            rng = random.Random(int(identifier, 16))
            apply_augmentation = args.augment and split == "train"
            image = render_line(text, font, args.font_size, rng, augment=apply_augmentation)
            target = args.output_dir / f"{identifier}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, optimize=True)
            rendered[split].append({**record, "id": identifier, "image": str(target.resolve()),
                                    "font": font.name, "synthetic": True, "augment": apply_augmentation,
                                    "renderer": "pillow-raqm" if features.check_feature("raqm") else "reshaper-bidi"})
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rendered.items():
        with (args.output_data_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {"samples": len(records), "train": len(rendered["train"]), "val": len(rendered["val"]), "seed": args.seed, "fonts": [str(font) for font in fonts], "source": str(args.texts.resolve()), "augment": args.augment,
             "split_policy": "source groups, independent of rendering RNG", "harakat_preserved": True,
             "augmentation_split": "train only; validation rendering is fixed",
             "source_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
             "groups": {split: len({row["split_group"] for row in items}) for split, items in splits.items()}}
    (args.output_data_dir / "synthetic_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
