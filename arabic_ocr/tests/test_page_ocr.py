"""Functional regressions for page candidates and bounded recognition."""
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recognizer
from page_layout import detect_line_boxes, prepare_image


def sample_page():
    page = Image.new("L", (240, 140), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle((40, 20, 180, 34), fill=0)
    draw.rectangle((70, 80, 200, 94), fill=0)
    draw.point((8, 55), fill=0)
    return page


def test_layout_filters_speck_and_keeps_two_lines_in_order():
    boxes = detect_line_boxes(sample_page())
    assert len(boxes) == 2
    assert boxes[0] == (36, 16, 185, 39)
    assert boxes[1] == (66, 76, 205, 99)


def test_blank_noise_and_transparent_background_are_empty():
    rng = np.random.default_rng(7)
    assert detect_line_boxes(Image.fromarray(rng.integers(240, 256, (80, 120), dtype=np.uint8))) == []
    assert detect_line_boxes(Image.new("RGBA", (120, 80), (0, 0, 0, 0))) == []
    assert np.asarray(prepare_image(Image.new("RGBA", (120, 80), (0, 0, 0, 0)))).min() == 255


def test_arabic_satellite_dots_are_attached_to_body():
    page = Image.new("L", (160, 90), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle((50, 16, 55, 18), fill=0)
    draw.rectangle((20, 23, 130, 36), fill=0)
    draw.rectangle((20, 65, 130, 78), fill=0)
    boxes = detect_line_boxes(page)
    assert len(boxes) == 2
    assert boxes[0][1] <= 16 and boxes[0][3] >= 37


@pytest.fixture
def mock_recognizer(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    class DummyModel:
        def __init__(self):
            self.batch_sizes = []
        def __call__(self, images, widths):
            self.batch_sizes.append(len(widths))
            scores = torch.full((3, len(widths), 2), -10.)
            scores[:, :, 1] = 0
            return scores, torch.full((len(widths),), 3, dtype=torch.long)
    model = DummyModel()
    monkeypatch.setattr(recognizer, "load_model", lambda *_: (model, ["س"], {"height": 64, "max_width": 1600}, 5))
    return recognizer.OCRRecognizer(checkpoint, "cpu", batch_size=2), model


def test_inference_batches_generator_and_preserves_logical_text(mock_recognizer):
    engine, model = mock_recognizer
    assert engine.recognize_lines(Image.new("L", (80, 32), 255) for _ in range(5)) == ["س"] * 5
    assert model.batch_sizes == [2, 2, 1]


def test_page_result_has_boxes_text_and_handles_blank(mock_recognizer):
    engine, model = mock_recognizer
    result = engine.recognize_page(sample_page())
    assert result["text"] == "س\nس"
    assert result["lines"][0]["box"] == [36, 16, 185, 39]
    assert result["checkpoint_epoch"] == 5
    model.batch_sizes.clear()
    assert engine.recognize_page(Image.new("L", (120, 80), 255))["lines"] == []
    assert model.batch_sizes == []


def test_invalid_parameters_and_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        recognizer.OCRRecognizer(tmp_path / "missing.pt")
    with pytest.raises(ValueError):
        detect_line_boxes(sample_page(), max_gap=-1)


def test_exif_orientation_is_applied_once():
    page = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(page).rectangle((0, 0, 10, 20), fill="black")
    page.getexif()[274] = 6
    oriented = prepare_image(page)
    assert oriented.size == (80, 120)
    assert oriented.getpixel((79, 0)) == 0
    assert prepare_image(oriented).size == oriented.size


def test_oversized_image_rejected_before_model_work():
    with pytest.raises(ValueError, match="20 million"):
        recognizer.OCRRecognizer._check_image(Image.new("L", (5000, 4001)))


@pytest.mark.parametrize("output_flag", ["--output", "--text-output"])
def test_cli_cannot_overwrite_checkpoint(tmp_path, monkeypatch, output_flag):
    import page_ocr
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trusted original")
    argv = ["page_ocr.py", "--checkpoint", str(checkpoint), "--image", str(tmp_path / "page.png"),
            "--output", str(checkpoint if output_flag == "--output" else tmp_path / "page.json")]
    if output_flag == "--text-output":
        argv.extend(["--text-output", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        page_ocr.main()
    assert error.value.code == 2
    assert checkpoint.read_bytes() == b"trusted original"
