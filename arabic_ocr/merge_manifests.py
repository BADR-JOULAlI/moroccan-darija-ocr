"""Merge compatible OCR manifests into one mixed training corpus.

Rows are prefixed by source, exact transcript duplicates are removed across
splits, and the resulting vocabulary is built from the merged training split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path, prefix: str) -> list[dict]:
    output = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        row["id"] = f"{prefix}_{row.get('id', len(output))}"
        output.append(row)
    return output


def dedupe(items: list[dict], seen_text: set[str], seen_image: set[str]) -> list[dict]:
    output = []
    for row in items:
        text_key = " ".join(str(row["text"]).split())
        image_key = str(row.get("image", ""))
        image_hash = hashlib.sha256(Path(image_key).read_bytes()).hexdigest() if Path(image_key).is_file() else image_key
        if text_key in seen_text or image_hash in seen_image:
            continue
        seen_text.add(text_key)
        seen_image.add(image_hash)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "TRAIN", "VAL"), required=True)
    args = parser.parse_args()
    train, val = [], []
    for name, train_path, val_path in args.source:
        train.extend(rows(Path(train_path), name))
        val.extend(rows(Path(val_path), name))
    train_seen_text, train_seen_image = set(), set()
    train = dedupe(train, train_seen_text, train_seen_image)
    val = dedupe(val, set(row["text"] for row in train), set())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in (("train", train), ("val", val)):
        (args.output_dir / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8")
    characters = sorted(set("".join(row["text"] for row in train)))
    (args.output_dir / "vocab.json").write_text(json.dumps({"characters": characters, "blank_index": 0, "normalization": "NFC + collapsed whitespace", "source_split": "train"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"sources": [name for name, _, _ in args.source], "train": len(train), "val": len(val), "vocabulary": len(characters), "train_sha256": hashlib.sha256((args.output_dir / "train.jsonl").read_bytes()).hexdigest(), "val_sha256": hashlib.sha256((args.output_dir / "val.jsonl").read_bytes()).hexdigest()}
    (args.output_dir / "merge_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
