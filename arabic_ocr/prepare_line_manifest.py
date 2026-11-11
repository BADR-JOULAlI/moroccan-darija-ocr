"""Validate vetted image-line/transcription JSONL files for CTC adaptation.

This tool deliberately refuses multiline text. It never crops a page or guesses
an alignment between a paragraph and visual lines. Each input row must contain
``image``, ``text`` and optionally ``id``; paths are made absolute in outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image

from data import preprocess


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_rows(path: Path, root: Path, normalize: bool) -> list[dict]:
    rows = []
    seen = set()
    seen_images = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("image"), str) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_no}: each row needs string image and text")
            if any(char in row["text"] for char in "\n\r\u2028\u2029"):
                raise ValueError(f"{path}:{line_no}: multiline text; line alignment is required")
            text = normalize_text(row["text"]) if normalize else row["text"]
            if not text or "\n" in text or "\r" in text:
                raise ValueError(f"{path}:{line_no}: text is empty or multiline; line alignment is required")
            image = Path(row["image"])
            if not image.is_absolute():
                image = (root / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: image not found: {image}")
            identifier = str(row.get("id") or image.name)
            if identifier in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {identifier}")
            seen.add(identifier)
            with Image.open(image) as sample:
                sample.load()
                if sample.width < 16 or sample.height < 8:
                    raise ValueError(
                        f"{path}:{line_no}: image is too small ({sample.width}x{sample.height})"
                    )
                _, width = preprocess(sample)
            required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            if width // 4 < required:
                raise ValueError(f"{path}:{line_no}: CTC needs {required} frames but image provides {width // 4}")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            if image_hash in seen_images:
                raise ValueError(f"{path}:{line_no}: duplicate image content: {image.name}")
            seen_images.add(image_hash)
            provenance = {key: row[key] for key in (
                "source_dataset", "source_split", "source_row_idx", "source_group",
                "source_id", "split_group", "synthetic", "augment", "renderer", "font"
            ) if key in row}
            rows.append({**provenance, "image": str(image), "text": text, "id": identifier})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalize", action="store_true", help="NFC plus collapse whitespace")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split in ("train", "val"):
        source = args.input_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        manifests[split] = read_rows(source, args.input_dir, args.normalize)
        if not manifests[split]:
            raise ValueError(f"{split} is empty")
    train_ids = {row["id"] for row in manifests["train"]}
    train_texts = {row["text"] for row in manifests["train"]}
    train_images = {hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() for row in manifests["train"]}
    for row in manifests["val"]:
        image_hash = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        if row["id"] in train_ids or row["text"] in train_texts or image_hash in train_images:
            raise ValueError(f"validation overlaps training by id/text/image: {row['id']}")
    characters = sorted(set("".join(row["text"] for row in manifests["train"])))
    for split, rows in manifests.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    vocab = {"characters": characters, "blank_index": 0, "normalization": "NFC + collapsed whitespace" if args.normalize else "source text preserved", "transcription_order": "Unicode logical order", "source_split": "train"}
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"train": len(manifests["train"]), "val": len(manifests["val"]), "vocabulary": len(characters), "train_only_vocabulary": True, "normalization": vocab["normalization"], "manifest_sha256": {split: hashlib.sha256((args.output_dir / f"{split}.jsonl").read_bytes()).hexdigest() for split in manifests}}
    (args.output_dir / "line_manifest_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
