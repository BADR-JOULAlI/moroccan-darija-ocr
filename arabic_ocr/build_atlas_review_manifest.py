"""Build a human-review manifest from bounded AtlasOCR previews.

The output is deliberately marked ``needs_manual_line_annotation`` and is not
accepted by the training loader. A reviewer must create one-line crops and
verified transcripts before calling ``prepare_line_manifest.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.audit_dir / "ATLAS_DATA_AUDIT.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_cache = {}
    for item in audit.get("sampled_images", []):
        if not item["sample"].startswith(("train_rows", "validation_rows")):
            continue
        if item["sample"] not in raw_cache:
            raw_path = next(args.audit_dir.glob(item["sample"] + "_*.json"))
            raw_cache[item["sample"]] = json.loads(raw_path.read_text(encoding="utf-8"))["rows"]
        source_row = next(row for row in raw_cache[item["sample"]] if row["row_idx"] == item["row_idx"])["row"]
        rows.append({"id": f"atlas_{item['sample']}_{item['row_idx']:05d}", "image": str(Path(item["path"]).resolve()),
                     "source_text": source_row["text"], "status": "needs_manual_line_annotation",
                     "source_dataset": "atlasia/atlasOCR-data", "source_split": item["sample"].replace("_rows", ""),
                     "source_row_idx": item["row_idx"], "dataset_revision": audit["dataset"]["revision"]})
    with (args.output_dir / "review_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    width, thumb_h = 1100, 180
    sheet = Image.new("RGB", (width, max(thumb_h * len(rows), thumb_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        image = Image.open(row["image"]).convert("RGB")
        image.thumbnail((width - 30, 110))
        top = index * thumb_h + 8
        sheet.paste(image, (15, top))
        draw.text((15, top + 120), f"{row['id']} | MANUAL LINE ANNOTATION REQUIRED", fill="black")
    sheet.save(args.output_dir / "atlas_review_contact_sheet.jpg", quality=92)
    (args.output_dir / "README.md").write_text(
        "# AtlasOCR review sample\n\n"
        "These bounded previews are candidates for manual line annotation. `source_text` is the original page/block text; "
        "it must not be copied to each crop. Create verified one-line image/transcription pairs, then run "
        "`prepare_line_manifest.py`. This folder is not a training dataset.\n",
        encoding="utf-8",
    )
    print(json.dumps({"review_rows": len(rows), "manifest": str((args.output_dir / "review_manifest.jsonl").resolve()), "contact_sheet": str((args.output_dir / "atlas_review_contact_sheet.jpg").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
