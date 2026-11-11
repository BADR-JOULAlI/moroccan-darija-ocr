"""Dataset isolation and Arabic-rendering checks for future synthetic runs."""
from __future__ import annotations

import builtins
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_synthetic_lines as synthetic
from merge_manifests import dedupe, merge_splits, rows
from prepare_line_manifest import read_rows


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def grouped_source(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"id": f"row{index}_{chunk}", "text": f"نص {index} رقم {chunk}",
                          "source_dataset": "atlasia/atlasOCR-data", "source_split": "train",
                          "source_row_idx": index}
                         for index in range(6) for chunk in range(2)])
    return source


def test_grouped_split_preserves_provenance_and_keeps_pages_separate(tmp_path):
    source = grouped_source(tmp_path)
    records = synthetic.read_records(source)
    splits = synthetic.split_records(records, 0.25, 42)
    train_pages = {row["source_row_idx"] for row in splits["train"]}
    val_pages = {row["source_row_idx"] for row in splits["val"]}
    assert train_pages and val_pages and train_pages.isdisjoint(val_pages)
    assert len(splits["train"]) + len(splits["val"]) == 12
    assert records[0]["source_id"] == "row0_0"
    assert records[0]["source_dataset"] == "atlasia/atlasOCR-data"


def test_zero_validation_ratio_has_no_repeated_training_row(tmp_path):
    records = synthetic.read_records(grouped_source(tmp_path))
    splits = synthetic.split_records(records, 0, 42)
    assert len(splits["train"]) == 12
    assert splits["val"] == []
    with pytest.raises(ValueError, match="independent source groups"):
        synthetic.split_records(records[:2], 0.1, 42)


def test_augmentation_does_not_change_membership_or_source_metadata(tmp_path, monkeypatch):
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.is_file():
        pytest.skip("Arial is unavailable")
    source = grouped_source(tmp_path)
    output = []
    for augment in (False, True):
        suffix = "augmented" if augment else "clean"
        manifest_dir = tmp_path / suffix / "manifests"
        argv = ["generate_synthetic_lines.py", "--texts", str(source), "--fonts", str(font),
                "--output-dir", str(tmp_path / suffix / "images"), "--output-data-dir", str(manifest_dir),
                "--seed", "42", "--val-ratio", "0.25"]
        monkeypatch.setattr(sys, "argv", argv + (["--augment"] if augment else []))
        synthetic.main()
        output.append({split: [json.loads(raw) for raw in (manifest_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                       for split in ("train", "val")})
    for split in ("train", "val"):
        assert [row["id"] for row in output[0][split]] == [row["id"] for row in output[1][split]]
        assert [row["source_row_idx"] for row in output[0][split]] == [row["source_row_idx"] for row in output[1][split]]
        assert all(row["synthetic"] for row in output[0][split])
    assert all(not row["augment"] for row in output[1]["val"])
    for clean, augmented in zip(output[0]["val"], output[1]["val"]):
        assert Path(clean["image"]).read_bytes() == Path(augmented["image"]).read_bytes()


def test_fallback_arabic_renderer_retains_diacritics():
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi.algorithm")
    rendered = synthetic.shape_text("بَتّ")
    assert "َ" in rendered and "ّ" in rendered


def test_missing_arabic_shaping_dependencies_raise_explicit_error(monkeypatch):
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "arabic_reshaper":
            raise ImportError("fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError, match="Arabic rendering requires"):
        synthetic.shape_text("دارجة")


def test_rendered_vowel_mark_is_visible_with_fallback(tmp_path, monkeypatch):
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.is_file():
        pytest.skip("Arial is unavailable")
    monkeypatch.setattr(synthetic.features, "check_feature", lambda name: False)
    bare = synthetic.render_line("ب", font, 42, random.Random(1))
    marked = synthetic.render_line("بَ", font, 42, random.Random(1))
    assert bare.size != marked.size or not np.array_equal(np.asarray(bare), np.asarray(marked))


def test_merge_resolves_relative_images_and_filters_cross_split_bytes(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    Image.new("L", (50, 20), 200).save(source / "train.png")
    (source / "copy.png").write_bytes((source / "train.png").read_bytes())
    Image.new("L", (50, 20), 120).save(source / "val.png")
    write_jsonl(source / "train.jsonl", [{"id": "1", "image": "train.png", "text": "دارجة"}])
    write_jsonl(source / "val.jsonl", [{"id": "2", "image": "copy.png", "text": "نص مختلف"},
                                        {"id": "3", "image": "val.png", "text": "مثال"}])
    train, val = merge_splits(rows(source / "train.jsonl", "a"), rows(source / "val.jsonl", "b"))
    assert len(train) == 1 and len(val) == 1
    assert val[0]["text"] == "مثال"
    assert Path(train[0]["image"]).is_absolute()


def test_merge_missing_image_fails_instead_of_treating_path_as_a_hash(tmp_path):
    manifest = tmp_path / "missing.jsonl"
    write_jsonl(manifest, [{"image": "absent.png", "text": "دارجة"}])
    with pytest.raises(FileNotFoundError, match="image not found"):
        rows(manifest, "source")
    with pytest.raises(FileNotFoundError, match="image not found"):
        dedupe([{"image": str(tmp_path / "absent.png"), "text": "دارجة"}], set(), set())


def test_validation_keeps_source_provenance_and_refuses_multiline_before_normalizing(tmp_path):
    Image.new("L", (300, 64), 255).save(tmp_path / "line.png")
    source = tmp_path / "lines.jsonl"
    row = {"id": "x", "image": "line.png", "text": "دارجة", "source_row_idx": 7,
           "source_dataset": "atlasia/atlasOCR-data", "split_group": "page7", "synthetic": True}
    write_jsonl(source, [row])
    validated = read_rows(source, tmp_path, normalize=True)[0]
    assert validated["source_row_idx"] == 7 and validated["split_group"] == "page7"
    write_jsonl(source, [{**row, "text": "سطر\nآخر"}])
    with pytest.raises(ValueError, match="multiline"):
        read_rows(source, tmp_path, normalize=True)
