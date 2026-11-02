"""Fetch bounded AtlasOCR training texts for synthetic line generation.

Only the text field is retained. Images and AtlasOCR test splits are not used;
the resulting lines must be rendered synthetically and labelled as synthetic.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_rows(offset: int, length: int) -> dict:
    url = "https://datasets-server.huggingface.co/rows?" + urlencode({
        "dataset": "atlasia/atlasOCR-data", "config": "default", "split": "train",
        "offset": offset, "length": length,
    })
    request = Request(url, headers={"User-Agent": "MoroccanDarijaOCR/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def chunks(text: str, max_chars: int) -> list[str]:
    words = " ".join(text.split()).split(" ")
    result, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            result.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        result.append(" ".join(current))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.rows <= 10000:
        raise ValueError("--rows must be between 1 and 10000")
    if not 1 <= args.batch_size <= 100:
        raise ValueError("--batch-size must be between 1 and 100")
    output, seen = [], set()
    for offset in range(0, args.rows, args.batch_size):
        payload = fetch_rows(offset, min(args.batch_size, args.rows - offset))
        for item in payload.get("rows", []):
            row = item["row"]
            for part in chunks(str(row.get("text", "")), args.max_chars):
                if len(part) < 2 or part in seen:
                    continue
                seen.add(part)
                output.append({"id": f"atlas_train_{len(output):07d}", "text": part,
                               "source_dataset": "atlasia/atlasOCR-data", "source_split": "train",
                               "source_row_idx": item["row_idx"], "synthetic": True})
        print(f"fetched {min(offset + args.batch_size, args.rows)}/{args.rows}", flush=True)
        time.sleep(0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows_requested": args.rows, "synthetic_texts": len(output), "max_chars": args.max_chars}, ensure_ascii=False))


if __name__ == "__main__":
    main()
