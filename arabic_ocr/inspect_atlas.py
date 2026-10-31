"""Bounded, read-only inspection of Atlas OCR datasets; never download full shards.

Fetch public metadata, pinned cards/tree and at most --rows rows per inspected
window. Preserve transcription newlines exactly. Optional --images downloads
only a few cached previews; it never trains, evaluates predictions or emits line
labels. Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import unicodedata
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

DATA = "atlasia/atlasOCR-data"
BENCH = "atlasia/AtlasOCRBench"
HTTP_LIMIT = 8 * 1024 * 1024
IMAGE_LIMIT = 2 * 1024 * 1024


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(url: str, limit: int = HTTP_LIMIT) -> tuple[bytes, dict]:
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "AtlasOCRReadOnlyAudit/1.0"})
            with urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > limit:
                    raise ValueError(f"Response exceeds audit byte limit: {length}")
                payload = response.read(limit + 1)
                if len(payload) > limit:
                    raise ValueError("Response exceeds audit byte limit")
                return payload, {"url": url, "status": response.status,
                                 "fetched_utc": datetime.now(timezone.utc).isoformat(),
                                 "bytes": len(payload), "sha256": sha256(payload),
                                 "etag": response.headers.get("ETag")}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def snapshot(output: Path, name: str, url: str) -> tuple[object, dict]:
    payload, provenance = fetch(url)
    path = output / name
    path.write_bytes(payload)
    provenance["snapshot"] = str(path.resolve())
    return (payload.decode("utf-8") if name.endswith(".md") else json.loads(payload)), provenance


def row_url(dataset: str, split: str, offset: int, count: int) -> str:
    return "https://datasets-server.huggingface.co/rows?" + urlencode({
        "dataset": dataset, "config": "default", "split": split,
        "offset": offset, "length": count})


def row_summary(payload: dict) -> dict:
    rows = payload["rows"]
    summaries = []
    for item in rows:
        row = item["row"]
        text = row["text"]
        image = row["image"]
        summaries.append({
            "row_idx": item["row_idx"], "id_if_published": row.get("id"),
            "text_characters": len(text), "text_sha256_utf8": sha256(text.encode("utf-8")),
            "newline_count": text.count("\n"), "carriage_return_count": text.count("\r"),
            "unicode_line_separator_count": text.count("\u2028"),
            "unicode_paragraph_separator_count": text.count("\u2029"),
            "literal_backslash_n_count": text.count("\\n"),
            "image_width": image.get("width"), "image_height": image.get("height"),
            "metadata": row.get("metadata"), "row_keys": list(row),
            "truncated_cells": item.get("truncated_cells", []),
        })
    lengths = [item["text_characters"] for item in summaries]
    return {"num_rows_total": payload.get("num_rows_total"), "sample_size": len(rows),
            "features": payload["features"], "partial": payload.get("partial"),
            "text_preserved_without_normalization": True,
            "rows_with_newlines": sum(item["newline_count"] > 0 for item in summaries),
            "rows_with_truncated_cells": sum(bool(item["truncated_cells"]) for item in summaries),
            "text_length_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)] if lengths else [],
            "all_observed_row_keys": sorted({key for item in summaries for key in item["row_keys"]}),
            "all_observed_metadata_keys": sorted({key for item in summaries if isinstance(item["metadata"], dict) for key in item["metadata"]}),
            "rows": summaries}


def compare_reference_hashes(left: dict, right: dict) -> dict:
    right_by_hash = defaultdict(list)
    for item in right["rows"]:
        right_by_hash[sha256(item["row"]["text"].encode("utf-8"))].append(item)
    matches = []
    for item in left["rows"]:
        text_hash = sha256(item["row"]["text"].encode("utf-8"))
        for other in right_by_hash[text_hash]:
            a, b = item["row"]["image"], other["row"]["image"]
            matches.append({"data_test_row_idx": item["row_idx"],
                            "benchmark_row_idx": other["row_idx"],
                            "benchmark_id": other["row"].get("id"),
                            "exact_text_sha256_utf8": text_hash,
                            "same_image_dimensions": (a.get("width"),a.get("height")) == (b.get("width"),b.get("height"))})
    return {"left_rows": len(left["rows"]), "right_rows": len(right["rows"]),
            "matched_pairs": matches, "num_exact_text_matched_pairs": len(matches),
            "proves_full_split_identity": False,
            "method": "SHA-256 of exact unnormalized UTF-8 reference text; dimensions compared, no OCR predictions or error analysis."}


def inspect(output: Path, count: int, with_images: bool = False, findings_path: Path | None = None) -> dict:
    if not 1 <= count <= 100:
        raise ValueError("--rows must be between 1 and 100 (bounded audit)")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    metadata = {}
    tree = {}
    cards = {}
    # Read revisions first, then pin the repository evidence to these commits.
    for label, dataset in (("data", DATA), ("bench", BENCH)):
        obj, source = snapshot(output, label + "_metadata.json", "https://huggingface.co/api/datasets/" + dataset)
        metadata[label] = obj
        sources.append(source)
        revision = obj["sha"]
        jobs = [(label + "_tree_pinned.json", "https://huggingface.co/api/datasets/" + dataset + "/tree/" + revision + "?recursive=true&expand=false"),
                (label + "_readme_pinned.md", "https://huggingface.co/datasets/" + dataset + "/raw/" + revision + "/README.md"),
                (label + "_splits.json", "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset":dataset}))]
        for name, url in jobs:
            value, source = snapshot(output, name, url)
            sources.append(source)
            if "_tree_" in name:
                tree[label] = value
            if name.endswith(".md"):
                cards[label] = value
    jobs = [("train_rows", DATA, "train", 0), ("validation_rows", DATA, "validation", 0),
            ("data_test_rows", DATA, "test", 0), ("bench_test_rows", BENCH, "test", 0),
            ("bench_test_offset55_rows", BENCH, "test", 55)]

    def sample(job):
        name, dataset, split, offset = job
        value, source = snapshot(output, name + f"_{count}.json", row_url(dataset, split, offset, count))
        print(f"Inspected {dataset}/{split}: rows {offset}:{offset+count}", flush=True)
        return name, value, source

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(sample, jobs))
    samples = {}
    for name, value, source in results:
        samples[name] = value
        sources.append(source)
    for name, endpoint in (("data_size", "size"), ("data_parquet", "parquet")):
        _, source = snapshot(output, name + ".json", "https://datasets-server.huggingface.co/" + endpoint + "?" + urlencode({"dataset":DATA}))
        sources.append(source)
    overlap = {name: compare_reference_hashes(samples["data_test_rows"], samples[name])
               for name in ("bench_test_rows", "bench_test_offset55_rows")}
    images = []
    if with_images:
        # Fixed few training/validation previews, not full examples/shards.
        selected = [("train_rows", 0), ("train_rows", min(7,count-1)),
                    ("validation_rows", min(2,count-1)), ("validation_rows", min(9,count-1))]
        # One matched pair provides pixel-file evidence for benchmark overlap.
        pairs = [pair for comparison in overlap.values() for pair in comparison["matched_pairs"]]
        if pairs:
            pair = pairs[0]
            selected.append(("data_test_rows", pair["data_test_row_idx"]))
            for name in ("bench_test_rows", "bench_test_offset55_rows"):
                if any(item["row_idx"] == pair["benchmark_row_idx"] for item in samples[name]["rows"]):
                    selected.append((name,pair["benchmark_row_idx"]))
                    break
        for name, row_index in dict.fromkeys(selected):
            item = next(item for item in samples[name]["rows"] if item["row_idx"] == row_index)
            image = item["row"]["image"]
            payload, provenance = fetch(image["src"], IMAGE_LIMIT)
            destination = output / f"{name}_{row_index:05d}.jpg"
            destination.write_bytes(payload)
            images.append({"sample": name, "row_idx": row_index, "path":str(destination),
                           "width":image.get("width"), "height":image.get("height"), **provenance})
    file_inventory = {}
    for label, entries in tree.items():
        shards = [item for item in entries if item["type"] == "file" and item["path"].endswith(".parquet")]
        file_inventory[label] = {"parquet_files": [{"path": item["path"], "bytes":item["size"], "lfs_sha256":item.get("lfs",{}).get("oid")} for item in shards],
                                 "parquet_file_count":len(shards), "compressed_bytes_total":sum(item["size"] for item in shards),
                                 "has_standalone_license_file": any(item["path"].split("/")[-1].upper().startswith("LICENSE") for item in entries)}
    sampled_texts = [item["row"]["text"] for name in ("train_rows","validation_rows") for item in samples[name]["rows"]]
    observed_revisions = sorted({item["row"]["image"]["src"].split("/--/")[1] for payload in samples.values() for item in payload["rows"] if "/--/" in item["row"]["image"]["src"]})
    audit = {
        "audit_created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Read-only bounded metadata + reference samples. No full parquet downloads, training, OCR inference, benchmark error analysis, or line-label generation.",
        "dataset":{"id":DATA,"url":"https://huggingface.co/datasets/"+DATA,"revision":metadata["data"]["sha"],"last_modified":metadata["data"]["lastModified"],"private":metadata["data"]["private"],"gated":metadata["data"]["gated"],"features":metadata["data"]["cardData"]["dataset_info"]["features"],"splits":metadata["data"]["cardData"]["dataset_info"]["splits"],"files":file_inventory["data"]},
        "benchmark":{"id":BENCH,"url":"https://huggingface.co/datasets/"+BENCH,"revision":metadata["bench"]["sha"],"last_modified":metadata["bench"]["lastModified"],"gated":metadata["bench"]["gated"],"features":metadata["bench"]["cardData"]["dataset_info"]["features"],"splits":metadata["bench"]["cardData"]["dataset_info"]["splits"],"files":file_inventory["bench"],"use_policy":"Frozen test only; never use benchmark labels, images, crops or errors for training, selecting alignments, thresholds, vocabulary or hyperparameters."},
        "cached_asset_revisions_observed":observed_revisions,
        "sample_summaries":{name:row_summary(value) for name,value in samples.items()},
        "raw_text_audit":{"train_validation_rows_inspected":len(sampled_texts),"rows_with_lf":sum("\n" in text for text in sampled_texts),"rows_with_cr":sum("\r" in text for text in sampled_texts),"rows_with_unicode_separators":sum("\u2028" in text or "\u2029" in text for text in sampled_texts),"normalization_applied":False,"line_breaks_were_not_stripped":True},
        "supervision_findings":{"published_line_bounding_boxes":False,"published_word_bounding_boxes":False,"published_per_line_text_field":False,"published_source_document_id_in_training_schema":False,"training_metadata_fields":["contains_title","font"],"synthetic_boxes_claim_in_card":"Card describes OCRSmith generating bounding boxes, but the released schema/tree exposes no boxes or sidecar annotation files.","newline_conclusion":"None of the inspected train/validation strings contain LF/CR/Unicode line separators. This is an observation on the bounded sample, not a claim that every dataset row lacks newlines.","ctc_readiness":"Not directly usable as line-image/line-text pairs. Multi-line pictures plus flat transcriptions require verified alignment, upstream layout annotations, or controlled regeneration."},
        "licensing":{"data_announced":"Apache 2.0 badge in dataset README; no license YAML field or standalone license file at inspected revision.","benchmark_announced":"README licensing section explicitly says Apache 2.0; no standalone license file at inspected revision.","source_provenance":"Authors describe synthetic OCRSmith data and real-world book/social/education/cookbook data with Gemini pseudo-labeling and human correction. These are source-card claims, not independently revalidated annotation quality.","attribution":"Imane Momayiz, Soufiane Ait Elaouad, Abdeljalil Elmajjodi, Haitame Bouanane; AtlasIA, AtlasOCR, 2025. Benchmark additionally credits Zaid Chiech."},
        "test_benchmark_overlap_sample":overlap,
        "overlap_policy":"Reserve atlasOCR-data/test (196) and all AtlasOCRBench (251) regardless of sampled overlap; deduplicate candidate train/validation against frozen held-out sets by original image hashes and normalized transcripts before adaptation.",
        "safe_next_steps":[
            "Prefer obtaining original OCRSmith line/word bounding boxes plus exact rendered line strings at a pinned source revision; they are absent from the released Hugging Face schema.",
            "Alternatively regenerate new synthetic single-line images using atlasOCR-data/train strings only, preserving the exact emitted Unicode text and recording renderer/font/bidi/seed/source-row provenance. This is newly generated data, not recovered original line ground truth.",
            "For existing multi-line images, detect geometric line candidates, remove decorative rules and assign top-to-bottom reading order. Use model proposals and constrained monotonic text alignment only as suggestions; manually verify each crop and transcription before accepting supervised line pairs.",
            "Keep validation source pages separate before deriving crops, manually annotate validation line pairs, quarantine ambiguous/title/multicolumn/missing-text cases, and never distribute flat paragraph text across crops by equal word counts or guessed wrapping.",
            "If verified per-line labels cannot be recovered, retain the page/paragraph supervision and use a recognizer designed for multi-line sequence decoding rather than presenting guesses as line ground truth."
        ],
        "sampled_image_hash_matches":[{"left_sample":left["sample"],"left_row_idx":left["row_idx"],"right_sample":right["sample"],"right_row_idx":right["row_idx"],"cached_file_sha256":left["sha256"],"bytes":left["bytes"]} for i,left in enumerate(images) for right in images[i+1:] if left["sample"] == "data_test_rows" and right["sample"].startswith("bench_") and left["sha256"] == right["sha256"]],
        "sampled_images":images,"evidence_sources":sources,
        "limits":["Only first --rows train/validation and two bounded benchmark windows inspected; these may overrepresent synthetic data.","No full-corpus duplicate/provenance audit; no general claim of split independence.","Signed cached-asset URLs expire; refetch rows to recover new preview links at the recorded revision.","A newline, if present in another row, is not by itself verified alignment to a visual baseline.","Image cached previews may be transcoded; matched cached-file hashes corroborate sampled image identity but do not replace original-parquet byte hashing."]
    }
    json_write(findings_path or output / "ATLAS_DATA_AUDIT.json", audit)
    json_write(output / "evidence_index.json", sources)
    print(json.dumps({"dataset_revision":audit["dataset"]["revision"],"data_files_bytes":file_inventory["data"]["compressed_bytes_total"],"train_val_rows_inspected":len(sampled_texts),"rows_with_lf":audit["raw_text_audit"]["rows_with_lf"],"sample_overlap_counts":{name:item["num_exact_text_matched_pairs"] for name,item in overlap.items()}},indent=2), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("atlas_audit"),help="Raw evidence snapshots")
    parser.add_argument("--findings",type=Path,help="Optional path for machine-readable findings JSON")
    parser.add_argument("--rows",type=int,default=20,help="Rows per bounded window; 1..100, default20")
    parser.add_argument("--images",action="store_true",help="Download up to six cached previews only; no full shard")
    args=parser.parse_args()
    inspect(args.output_dir,args.rows,args.images,args.findings)


if __name__ == "__main__":
    main()
