"""Evaluate a frozen line-level benchmark manifest without training on it.

Multiline labels and images above a configurable height are rejected. Height is
only a heuristic: a short paragraph can still contain several visual lines. This
check cannot establish annotation quality or independence from training data.
AtlasOCRBench page/block images need page OCR and page-level scoring instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent

def validate_line_manifest(path: Path, max_height: int = 512) -> list[dict]:
    """Validate records and resolve image paths relative to their manifest."""
    path = Path(path).resolve()
    if max_height < 1:
        raise ValueError("max_height must be positive")
    rows = []
    seen_ids = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        location = f"{path}:{line_no}"
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{location}: each record must be a JSON object")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or any(char in text for char in "\n\r\u2028\u2029"):
            raise ValueError(f"{location}: non-empty, single-line string text is required")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{location}: non-empty string id is required")
        if identifier in seen_ids:
            raise ValueError(f"{location}: duplicate id {identifier!r}")
        seen_ids.add(identifier)
        image_name = row.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"{location}: non-empty string image path is required")
        image = Path(image_name)
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{location}: {image}")
        with Image.open(image) as sample:
            sample.load()
            if sample.height > max_height:
                raise ValueError(f"{location}: image height {sample.height} exceeds line-height heuristic {max_height}")
        rows.append({**row, "image": str(image)})
    if not rows:
        raise ValueError("benchmark manifest is empty")
    return rows


def run_benchmark(checkpoint: Path, manifest: Path, output_dir: Path,
                  batch_size: int = 16, device: str = "cuda", max_height: int = 512) -> dict:
    """Snapshot validated records and invoke the evaluator from any directory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint, manifest, output_dir = (Path(path).resolve() for path in (checkpoint, manifest, output_dir))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = validate_line_manifest(manifest, max_height=max_height)
    snapshot = output_dir / "benchmark_manifest.jsonl"
    if snapshot == manifest:
        raise ValueError("input manifest must differ from output benchmark_manifest.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = output_dir / "benchmark_verification.json"
    verification.unlink(missing_ok=True)
    snapshot.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    subprocess.run([sys.executable, "-X", "utf8", str(PROJECT / "evaluate.py"),
                    "--checkpoint", str(checkpoint), "--manifest", str(snapshot),
                    "--output-dir", str(output_dir), "--batch-size", str(batch_size),
                    "--device", device], cwd=PROJECT, check=True)
    result = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    result.update({"benchmark_policy": "evaluation only; training independence requires a separate provenance audit",
                   "line_validation": "label and height heuristics only; visual line annotation is not verified",
                   "max_height": max_height, "samples_validated": len(rows),
                   "source_manifest": str(manifest), "source_manifest_sha256": source_hash})
    verification.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-height", type=int, default=512, help="Heuristic only; does not prove visual line count")
    args = parser.parse_args()
    result = run_benchmark(args.checkpoint, args.manifest, args.output_dir,
                           args.batch_size, args.device, args.max_height)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
