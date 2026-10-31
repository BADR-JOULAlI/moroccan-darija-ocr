"""Download and audit the fixed Moroccan Gazette OCR corpus (CC BY 4.0).

Transcriptions remain in Unicode logical order. Normalization is NFC and collapsed
whitespace only: no diacritics, tatweel, letters, or bidi marks are deleted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import time
import unicodedata
import urllib.request
import zipfile

BASE_URL = "https://zenodo.org/records/21702529/files/"
MD5 = {
    "images.zip": "92a23fa2b77a04ada35812cb34c83cc7",
    "merged.jsonl": "99071657c9b295b87cefacc96fdeae30",
    "real_holdout_test.jsonl": "5dcc7d21a062b98d9401824577ba28ce",
    "README.md": "95ceff82838ece0c9c91507a87eed4ec",
    "checksums.txt": "e60a91f85629a96f510a57b4e7c35489",
}
TRAIN_QUARANTINE = {
    "1 (157)__page_001__region_010.png": "Visual inspection: annotation contains dozens of zeros that do not match the date in the crop.",
    "1 (183)__page_026__region_001.png": "Visual inspection: crop contains two visual text lines; this recognizer expects one line.",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(directory: Path, filename: str) -> Path:
    target = directory / filename
    if target.is_file() and digest(target, "md5") == MD5[filename]:
        print(f"Verified existing {filename}", flush=True)
        return target
    partial = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(BASE_URL + filename + "?download=1", headers={"User-Agent": "GazetteOCRResearch/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if digest(partial, "md5") != MD5[filename]:
                raise ValueError(f"MD5 mismatch for {filename}")
            partial.replace(target)
            print(f"Downloaded and verified {filename}: {target.stat().st_size:,} bytes", flush=True)
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def load_records(path: Path) -> list[dict]:
    records = []
    names = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("image"), str) or not isinstance(row.get("text"), str):
                raise ValueError(f"Invalid record at {path}:{line_no}")
            image = PurePosixPath(row["image"].replace("\\", "/")).name
            if image in names:
                raise ValueError(f"Duplicate filename in annotations: {image}")
            names.add(image)
            text = normalize_text(row["text"])
            if not text:
                raise ValueError(f"Empty transcription at {path}:{line_no}")
            records.append({"id": image, "text": text})
    return records


def extract_images(archive: Path, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive path escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                # Verify CRC so an interrupted or modified extraction is not reused.
                import zlib
                if zlib.crc32(target.read_bytes()) & 0xFFFFFFFF == member.CRC:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    result = {}
    for path in destination.rglob("*.png"):
        if path.name in result:
            raise ValueError(f"Ambiguous duplicate image basename: {path.name}")
        result[path.name] = path.resolve()
    return result


def page_group(filename: str) -> str:
    match = re.match(r"(.+?__page_\d+)(?:__|\.)", filename)
    return match.group(1) if match else filename.split("__region_")[0]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps({k: row[k] for k in ("image", "text", "id")}, ensure_ascii=False) + "\n")


def prepare(data_dir: Path, seed: int = 42, val_fraction: float = 0.1) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: download_file(data_dir, name), MD5))
    # The author's checksums additionally verify the three corpus files via SHA-256.
    for line in (data_dir / "checksums.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if name not in MD5 or digest(data_dir / name) != expected:
            raise ValueError(f"SHA-256 mismatch or unexpected checksum entry: {name}")
    merged = load_records(data_dir / "merged.jsonl")
    test = load_records(data_dir / "real_holdout_test.jsonl")
    if len(merged) != 12890 or len(test) != 1212:
        raise ValueError(f"Unexpected corpus size: merged={len(merged)}, test={len(test)}")
    images = extract_images(data_dir / "images.zip", data_dir / "images")
    print(f"Extracted {len(images):,} images; computing image-content hashes", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(zip(images, pool.map(digest, images.values())))
    by_id = {row["id"]: row for row in merged}
    for row in merged + test:
        if row["id"] not in images:
            raise ValueError(f"Missing image: {row['id']}")
        row["image"] = str(images[row["id"]])
        row["sha256"] = hashes[row["id"]]
    test_annotation_mismatches = []
    for row in test:
        if row["id"] not in by_id:
            raise ValueError(f"Official test filename missing from merged annotations: {row['id']}")
        if row["text"] != by_id[row["id"]]["text"]:
            test_annotation_mismatches.append({"id": row["id"], "merged_text": by_id[row["id"]]["text"], "official_test_text": row["text"]})
    test_ids = {row["id"] for row in test}
    test_texts = {row["text"] for row in test} | {by_id[row["id"]]["text"] for row in test}
    test_hashes = {row["sha256"] for row in test}
    candidates = [row for row in merged if row["id"] not in test_ids]
    removed_test = [row for row in candidates if row["text"] in test_texts or row["sha256"] in test_hashes]
    candidates = [row for row in candidates if row["text"] not in test_texts and row["sha256"] not in test_hashes]
    groups = defaultdict(list)
    for row in candidates:
        groups[page_group(row["id"])].append(row)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target_val = round(len(candidates) * val_fraction)
    val = []
    train = []
    val_groups = set()
    for group in group_ids:
        if len(val) < target_val:
            val.extend(groups[group])
            val_groups.add(group)
        else:
            train.extend(groups[group])
    val_texts = {row["text"] for row in val}
    val_hashes = {row["sha256"] for row in val}
    removed_val = [row for row in train if row["text"] in val_texts or row["sha256"] in val_hashes]
    train = [row for row in train if row["text"] not in val_texts and row["sha256"] not in val_hashes]
    train_before_quarantine = len(train)
    quarantined = [{"image": row["image"], "text": row["text"], "id": row["id"], "reason": TRAIN_QUARANTINE[row["id"]]} for row in train if row["id"] in TRAIN_QUARANTINE]
    train = [row for row in train if row["id"] not in TRAIN_QUARANTINE]
    write_json(data_dir / "quarantine.json", {"scope": "Training rows only, after frozen split assignment; validation and official test unchanged.", "records": quarantined})
    for a, b in ((train, val), (train, test), (val, test)):
        for key in ("id", "text", "sha256"):
            if {row[key] for row in a} & {row[key] for row in b}:
                raise AssertionError(f"Cross-split {key} leakage")
    train.sort(key=lambda row: row["id"])
    val.sort(key=lambda row: row["id"])
    characters = sorted(set("".join(row["text"] for row in train)))
    if not train or not val:
        raise ValueError("Train or validation split is empty")
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(data_dir / f"{split_name}.jsonl", rows)
    write_json(data_dir / "vocab.json", {"characters": characters, "blank_index": 0, "normalization": "Unicode NFC; collapse whitespace; preserve all non-whitespace symbols", "transcription_order": "Unicode logical order", "source_split": "train"})
    oov = {}
    for name, rows in (("val", val), ("test", test)):
        counts = Counter(ch for row in rows for ch in row["text"] if ch not in characters)
        oov[name] = {"characters": dict(counts), "num_characters": sum(counts.values()), "num_lines": sum(any(ch not in characters for ch in row["text"]) for row in rows)}
    test_pages = {page_group(row["id"]) for row in test}
    audit = {
        "source": "https://zenodo.org/records/21702529", "doi": "10.5281/zenodo.21702529", "license": "CC BY 4.0", "seed": seed,
        "normalization": "Unicode NFC, then collapse all whitespace to single ASCII spaces; retain diacritics, tatweel and bidi marks",
        "transcription_order": "Unicode logical order; recognition must account for Arabic right-to-left visual order",
        "counts": {"all": len(merged), "official_test": len(test), "after_filename_exclusion": len(merged)-len(test), "removed_test_content_overlap": len(removed_test), "removed_val_content_overlap_from_train": len(removed_val), "train_before_quality_quarantine": train_before_quarantine, "quarantined_train_rows": len(quarantined), "train": len(train), "val": len(val), "test": len(test), "vocabulary": len(characters), "images": len(images)},
        "split_method": "Fixed official test; remove non-test rows matching its normalized transcript or image SHA-256; group remaining rows by document+page, shuffle page groups with seed, allocate ~val_fraction to validation; remove train rows matching validation transcript or image SHA-256.",
        "val_fraction_requested": val_fraction, "test_untouched": True,
        "test_annotation_mismatches": test_annotation_mismatches,
        "test_annotation_policy": "Official holdout text is authoritative; preserve it unchanged after common normalization. Exclude test filenames and both official/merged text variants from train and validation.",
        "quality_quarantine_policy": "Explicit two source IDs, manually inspected, excluded only if assigned to training after split creation. No repeated-punctuation heuristic. Validation and official test unchanged.",
        "quarantined_train_records": quarantined,
        "train_val_page_overlap": len({page_group(row["id"]) for row in train} & val_groups),
        "train_test_page_overlap": len({page_group(row["id"]) for row in train} & test_pages),
        "val_test_page_overlap": len(val_groups & test_pages),
        "limitation": "Official test contains pages/documents also represented in other splits; no unseen-document generalization claim. Official test preserves any internal duplicates.",
        "cross_split_id_image_hash_transcript_overlap": 0,
        "within_split_duplicate_transcripts": {name: len(rows)-len({row["text"] for row in rows}) for name, rows in (("train",train),("val",val),("test",test))},
        "within_split_duplicate_images": {name: len(rows)-len({row["sha256"] for row in rows}) for name, rows in (("train",train),("val",val),("test",test))},
        "eval_oov": oov,
        "removed_for_test_overlap": [row["id"] for row in removed_test],
        "removed_for_val_overlap": [row["id"] for row in removed_val],
        "source_md5": MD5,
        "source_sha256": {name: digest(data_dir / name) for name in MD5},
        "manifest_sha256": {name: digest(data_dir / f"{name}.jsonl") for name in ("train", "val", "test")},
    }
    write_json(data_dir / "data_audit.json", audit)
    print(json.dumps({"counts": audit["counts"], "eval_oov": oov, "train_test_page_overlap": audit["train_test_page_overlap"]}, ensure_ascii=False, indent=2), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gazette"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.data_dir, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()
