"""Exercise demo dispatch and validation without launching a web server."""
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import LINE_MODE, PAGE_MODE, recognize_image
import demo


class StubRecognizer:
    epoch = 5
    device = "cpu"

    def recognize_line(self, image):
        assert isinstance(image, Image.Image)
        return "سلام عليكم"

    def recognize_page(self, image):
        assert isinstance(image, Image.Image)
        return {"text": "السطر الأول\nالسطر الثاني", "checkpoint_epoch": self.epoch,
                "lines": [{"index": 0, "box": [0, 0, 100, 20], "text": "السطر الأول"},
                          {"index": 1, "box": [0, 30, 100, 50], "text": "السطر الثاني"}],
                "reading_order": "top-to-bottom", "warnings": []}


def test_demo_line_keeps_logical_arabic_text():
    text, detail = recognize_image(StubRecognizer(), Image.new("L", (120, 30)), LINE_MODE)
    assert text == "سلام عليكم"
    assert detail == {"mode": "line", "checkpoint_epoch": 5, "device": "cpu"}


def test_demo_page_preserves_line_order_and_boxes():
    text, detail = recognize_image(StubRecognizer(), Image.new("L", (120, 90)), PAGE_MODE)
    assert text.splitlines() == [line["text"] for line in detail["lines"]]
    assert detail["lines"][1]["box"] == [0, 30, 100, 50]
    assert detail["reading_order"] == "top-to-bottom"


def test_demo_requires_an_image_and_known_mode():
    with pytest.raises(ValueError, match="Chargez"):
        recognize_image(StubRecognizer(), None, LINE_MODE)
    with pytest.raises(ValueError, match="image lisible"):
        recognize_image(StubRecognizer(), "path.png", LINE_MODE)
    with pytest.raises(ValueError, match="Choisissez"):
        recognize_image(StubRecognizer(), Image.new("L", (20, 20)), "unknown")


def test_demo_launch_is_local_and_serialized_by_default(tmp_path, monkeypatch):
    checkpoint = tmp_path / "local.pt"
    checkpoint.touch()
    observed = {}

    class StubInterface:
        def queue(self, **kwargs):
            observed["queue"] = kwargs
            return self

        def launch(self, **kwargs):
            observed["launch"] = kwargs

    monkeypatch.setattr(demo, "build_demo", lambda *args: StubInterface())
    monkeypatch.setattr(sys, "argv", ["demo.py", "--checkpoint", str(checkpoint), "--device", "cpu"])
    demo.main()
    assert observed["queue"]["default_concurrency_limit"] == 1
    assert observed["launch"]["server_name"] == "127.0.0.1"
    assert observed["launch"]["share"] is False
