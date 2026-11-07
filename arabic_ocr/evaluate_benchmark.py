"""Evaluate a frozen line-level benchmark manifest without training on it.

The script intentionally refuses multiline text or page-sized images. This
prevents accidentally reporting a page-recognition score with a line model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image



def validate_line_manifest(path: Path, max_height: int = 512) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("benchmark manifest is empty")
    for row in rows:
        text = str(row.get("text", ""))
        if not text or "\n" in text or "\r" in text:
            raise ValueError("benchmark manifest contains empty or multiline text; line labels are required")
        image = Path(row["image"])
        if not image.is_file():
            raise FileNotFoundError(image)
        with Image.open(image) as sample:
            if sample.height > max_height:
                raise ValueError(f"{image.name}: image height {sample.height} looks like a page, not a line")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    rows = validate_line_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-X", "utf8", "evaluate.py", "--checkpoint", str(args.checkpoint),
                    "--manifest", str(args.manifest), "--output-dir", str(args.output_dir),
                    "--batch-size", str(args.batch_size), "--device", args.device], check=True)
    result = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    result["benchmark_policy"] = "frozen line-level manifest; labels are never used for training"
    result["samples_validated"] = len(rows)
    (args.output_dir / "benchmark_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
