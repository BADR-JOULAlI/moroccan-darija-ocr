"""Regression checks for OCR alignment, decoding, metrics and data isolation.

Run from the project directory: python -m pytest -q
The final manifest audit is skipped until prepare_data.py has been run locally.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from metrics import Metrics, edit_distance
from model import CRNN, greedy_decode
from data import assert_ctc_feasible, collate_lines, preprocess
from prepare_line_manifest import read_rows
from segment_page_lines import find_line_boxes, segment_record
from generate_synthetic_lines import read_texts, render_line


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def probabilities(paths: list[list[int]], classes: int) -> torch.Tensor:
    scores = torch.full((len(paths[0]), len(paths), classes), -20.0)
    for sample, path in enumerate(paths):
        for time, index in enumerate(path):
            scores[time, sample, index] = 0
    return scores.log_softmax(-1)


def test_ctc_decoding_repeated_letters_blank_and_padding():
    # The second repeated alef survives because blank separates it. Later
    # padding deliberately predicts another character and must be ignored.
    scores = probabilities([[1, 1, 0, 1, 2, 2, 0, 2, 3, 1]], 4)
    assert greedy_decode(scores, torch.tensor([9]), ["ا", "ب", " "]) == ["اابب "]


def test_decode_keeps_logical_arabic_order_and_digits():
    expected = "كتاب ١٢"
    characters = list(dict.fromkeys(expected))
    path: list[int] = []
    for character in expected:
        index = characters.index(character) + 1
        path.extend([index, index, 0])
    scores = probabilities([path], len(characters) + 1)
    assert greedy_decode(scores, torch.tensor([len(path)]), characters) == [expected]


def test_metrics_are_micro_averaged_and_count_arabic_whitespace():
    metrics = Metrics()
    metrics.update("سلام عليكم", "سلام")
    metrics.update("و", "و")
    result = metrics.compute()
    assert result["character_errors"] == 6
    assert result["reference_characters"] == 11
    assert result["cer"] == pytest.approx(6 / 11)
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["exact_accuracy"] == 0.5
    assert result["samples"] == 2


def test_metrics_empty_and_insertions_are_finite_not_clipped():
    assert Metrics().compute()["cer"] == 0
    metrics = Metrics()
    metrics.update("a", "abcdef")
    assert metrics.compute()["cer"] == 5
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(["one", "two"], ["two"]) == 1


def test_variable_width_ctc_has_finite_gradient_on_cpu():
    torch.manual_seed(17)
    model = CRNN(num_classes=6, hidden_size=16, channels=4, dropout=0)
    images = torch.randn(2, 1, 32, 83)
    log_probs, lengths = model(images, torch.tensor([83, 64]))
    assert log_probs.shape == (20, 2, 6)
    assert lengths.tolist() == [20, 16]
    assert lengths.device.type == "cpu"
    assert log_probs.dtype == torch.float32
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(20, 2), atol=1e-6)
    # Repeated adjacent labels exercise CTC's extra blank-time requirement.
    targets = torch.tensor([1, 1, 2, 3, 4], dtype=torch.long)
    loss = nn.CTCLoss(blank=0, zero_infinity=False)(
        log_probs, targets, lengths, torch.tensor([3, 2])
    )
    assert torch.isfinite(loss)
    loss.backward()
    convolution = next(module for module in model.cnn.modules() if isinstance(module, nn.Conv2d))
    assert convolution.weight.grad is not None
    assert torch.isfinite(convolution.weight.grad).all()
    assert convolution.weight.grad.abs().sum() > 0


def test_valid_rtl_features_reverse_without_moving_pad_to_front():
    features = torch.tensor([[[10.0], [20.0], [30.0], [999.0], [999.0]]])
    reversed_features = CRNN.reverse_valid(features, torch.tensor([3]))
    assert reversed_features[0, :3, 0].tolist() == [30.0, 20.0, 10.0]


def test_packed_bilstm_predictions_ignore_extra_feature_padding():
    torch.manual_seed(29)
    model = CRNN(num_classes=6, hidden_size=16, channels=4, dropout=0).eval()
    source = torch.randn(2, 32, 1, 12)

    class SuppliedFeatures(nn.Module):
        def forward(self, images):
            return source[..., : images.shape[-1] // 4]

    # Isolate RNN packing from CNN boundary effects: both forwards contain
    # exactly the same valid features and different amounts of right padding.
    model.cnn = SuppliedFeatures()
    with torch.no_grad():
        short, lengths = model(torch.zeros(2, 1, 32, 36), torch.tensor([20, 36]))
        long, _ = model(torch.zeros(2, 1, 32, 48), torch.tensor([20, 36]))
    for sample, length in enumerate(lengths):
        assert torch.allclose(short[:length, sample], long[:length, sample], atol=1e-6)


def test_invalid_widths_fail_before_silent_ctc_misalignment():
    model = CRNN(num_classes=3, hidden_size=8, channels=2)
    images = torch.zeros(1, 1, 32, 40)
    with pytest.raises(ValueError):
        model(images, torch.tensor([41]))
    with pytest.raises(ValueError):
        model(images, torch.tensor([3]))


def test_ctc_feasibility_accounts_for_adjacent_identical_arabic_letters():
    # Two identical consecutive labels need a blank frame between them.
    assert_ctc_feasible(["لل"], [12])
    assert_ctc_feasible(["لا"], [8])
    with pytest.raises(ValueError, match="CTC infeasible"):
        assert_ctc_feasible(["لل"], [8])


def test_collation_matches_white_background_and_preserves_target_lengths():
    rows = [
        {"pixels": np.full((32, 16), 255, dtype=np.uint8), "width": 16,
         "target": [1, 1], "text": "لل", "id": "a"},
        {"pixels": np.zeros((32, 24), dtype=np.uint8), "width": 24,
         "target": [2], "text": "ا", "id": "b"},
    ]
    batch = collate_lines(rows)
    assert batch["images"].shape == (2, 1, 32, 24)
    assert batch["images"][0].count_nonzero() == 0
    assert torch.all(batch["images"][1] == 1)
    assert batch["widths"].tolist() == [16, 24]
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["targets"].tolist() == [1, 1, 2]


def test_rgba_lines_composite_transparent_background_onto_white():
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, 4:8, 3] = 255       # Opaque black ink.
    pixels[:, 8:12, 3] = 128      # Antialiased black ink over white.
    pixels[:, 12:, :] = 255       # Opaque white background.
    image, width = preprocess(Image.fromarray(pixels, "RGBA"), height=16)
    assert width == 16
    assert np.all(image[:, :4] == 255), "Transparent black must become white"
    assert np.all(image[:, 4:8] == 0)
    assert np.all(image[:, 8:12] == 127)
    assert np.all(image[:, 12:] == 255)


def test_downloaded_manifests_have_no_cross_split_identity_or_image_leakage():
    data_dir = PROJECT.parents[1] / "work" / "data" / "gazette"
    manifests = {name: data_dir / f"{name}.jsonl" for name in ("train", "val", "test")}
    if not all(path.exists() for path in manifests.values()):
        pytest.skip("Run prepare_data.py before auditing downloaded manifests")
    rows = {
        name: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for name, path in manifests.items()
    }
    assert len(rows["test"]) == 1212, "The official test must remain intact"
    identifiers = {name: {row["id"] for row in records} for name, records in rows.items()}
    transcriptions = {name: {row["text"] for row in records} for name, records in rows.items()}
    hashes = {
        name: {hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() for row in records}
        for name, records in rows.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert identifiers[left].isdisjoint(identifiers[right]), (left, right, "filenames")
        assert hashes[left].isdisjoint(hashes[right]), (left, right, "image content")
        assert transcriptions[left].isdisjoint(transcriptions[right]), (left, right, "exact transcript")
    vocabulary = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))["characters"]
    assert set(vocabulary) == set("".join(row["text"] for row in rows["train"]))
    audit = json.loads((data_dir / "data_audit.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        assert audit["counts"][split] == len(rows[split])
    quarantined = audit["quarantined_train_records"]
    quarantined_ids = {row["id"] for row in quarantined}
    assert len(quarantined_ids) == audit["counts"]["quarantined_train_rows"]
    assert quarantined_ids.isdisjoint(identifiers["train"])
    assert all(row["reason"] for row in quarantined)
    assert (
        audit["counts"]["train_before_quality_quarantine"] - len(rows["train"])
        == len(quarantined_ids)
    )
    quarantine_file = json.loads((data_dir / "quarantine.json").read_text(encoding="utf-8"))
    assert quarantine_file["records"] == quarantined


def test_line_manifest_refuses_multiline_text(tmp_path):
    image = tmp_path / "line.png"
    Image.new("L", (160, 64), 255).save(image)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps({"image": str(image), "text": "سطر\nآخر", "id": "x"}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="multiline"):
        read_rows(manifest, tmp_path, normalize=False)


def test_line_manifest_accepts_vetted_single_line(tmp_path):
    image = tmp_path / "line.png"
    Image.new("L", (320, 64), 255).save(image)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps({"image": str(image), "text": "دارجة", "id": "x"}, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = read_rows(manifest, tmp_path, normalize=True)
    assert rows[0]["image"] == str(image.resolve())
    assert rows[0]["text"] == "دارجة"


def test_line_manifest_rejects_duplicate_image_content(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("L", (320, 64), 240).save(first)
    Image.open(first).save(second)
    manifest = tmp_path / "rows.jsonl"
    records = [
        {"image": first.name, "text": "دارجة", "id": "first"},
        {"image": second.name, "text": "عربية", "id": "second"},
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate image content"):
        read_rows(manifest, tmp_path, normalize=True)


def test_line_manifest_rejects_tiny_images(tmp_path):
    image = tmp_path / "tiny.png"
    Image.new("L", (8, 8), 255).save(image)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "text": "دارجة", "id": "tiny"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="too small"):
        read_rows(manifest, tmp_path, normalize=True)


def test_page_segmentation_requires_explicit_text_lines(tmp_path):
    image = Image.new("L", (120, 80), 255)
    pixels = image.load()
    for y in range(10, 20):
        for x in range(5, 80):
            pixels[x, y] = 0
    for y in range(45, 55):
        for x in range(5, 70):
            pixels[x, y] = 0
    image_path = tmp_path / "page.png"
    image.save(image_path)
    assert len(find_line_boxes(image)) == 2
    with pytest.raises(ValueError, match="explicit line breaks"):
        segment_record({"id": "page", "image": image_path.name, "text": "paragraph"}, tmp_path, tmp_path / "lines")
    rows = segment_record({"id": "page", "image": image_path.name, "text": "première\nseconde"}, tmp_path, tmp_path / "lines")
    assert [row["text"] for row in rows] == ["première", "seconde"]
    assert all(Path(row["image"]).is_file() for row in rows)


def test_synthetic_generator_reads_text_and_renders_line(tmp_path):
    source = tmp_path / "texts.txt"
    source.write_text("سلام\n\nكيف داير؟\n", encoding="utf-8")
    assert read_texts(source) == ["سلام", "كيف داير؟"]
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.exists():
        pytest.skip("Windows Arial font is unavailable")
    image = render_line("سلام", font, 32, random.Random(1))
    assert image.mode == "L"
    assert image.width > 32 and image.height > 8
