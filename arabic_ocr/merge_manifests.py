"""Merge compatible OCR manifests into one mixed training corpus.

Rows are prefixed by source, exact transcript duplicates are removed across
splits, and the resulting vocabulary is built from the merged training split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def rows(path: Path, prefix: str) -> list[dict]:
    path = Path(path).resolve()
    output = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or not isinstance(row.get("image"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path}:{line_no}: expected string image and text fields")
        if not row["text"].strip() or "\n" in row["text"] or "\r" in row["text"]:
            raise ValueError(f"{path}:{line_no}: expected a non-empty single-line transcript")
        image = Path(row["image"])
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_no}: image not found: {image}")
        row["image"] = str(image.resolve())
        row["text"] = " ".join(unicodedata.normalize("NFC", row["text"]).split())
        row["id"] = f"{prefix}_{row.get('id', len(output))}"
        output.append(row)
    return output


def dedupe(items: list[dict], seen_text: set[str], seen_image: set[str]) -> list[dict]:
    output = []
    for row in items:
        text_key = " ".join(unicodedata.normalize("NFC", row["text"]).split())
        image_key = str(row.get("image", ""))
        if not Path(image_key).is_file():
            raise FileNotFoundError(f"image not found: {image_key}")
        image_hash = hashlib.sha256(Path(image_key).read_bytes()).hexdigest()
        if text_key in seen_text or image_hash in seen_image:
            continue
        seen_text.add(text_key)
        seen_image.add(image_hash)
        output.append(row)
    return output


def merge_splits(train: list[dict], val: list[dict]) -> tuple[list[dict], list[dict]]:
    """Exclude exact train text/image duplicates from validation as well."""
    seen_text, seen_image = set(), set()
    train = dedupe(train, seen_text, seen_image)
    val = dedupe(val, seen_text, seen_image)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "TRAIN", "VAL"), required=True)
    args = parser.parse_args()
    train, val = [], []
    for name, train_path, val_path in args.source:
        train.extend(rows(Path(train_path), name))
        val.extend(rows(Path(val_path), name))
    train_before, val_before = len(train), len(val)
    train, val = merge_splits(train, val)
    if not train or not val:
        raise ValueError("Merged training and validation must both remain non-empty after deduplication")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in (("train", train), ("val", val)):
        (args.output_dir / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8")
    characters = sorted(set("".join(row["text"] for row in train)))
    (args.output_dir / "vocab.json").write_text(json.dumps({"characters": characters, "blank_index": 0, "normalization": "NFC + collapsed whitespace", "source_split": "train"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"sources": [name for name, _, _ in args.source], "train": len(train), "val": len(val), "vocabulary": len(characters), "train_sha256": hashlib.sha256((args.output_dir / "train.jsonl").read_bytes()).hexdigest(), "val_sha256": hashlib.sha256((args.output_dir / "val.jsonl").read_bytes()).hexdigest(),
             "removed_train": train_before - len(train), "removed_val": val_before - len(val),
             "deduplication": "NFC + collapsed text, SHA-256 image file bytes across train/validation",
             "validation_oov": sorted(set("".join(row["text"] for row in val)) - set(characters))}
    (args.output_dir / "merge_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
