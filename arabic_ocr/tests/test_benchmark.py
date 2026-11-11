"""Manifest and subprocess boundary checks without benchmark data or labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import evaluate_benchmark as benchmark


def write_manifest(tmp_path, records):
    path = tmp_path / "references.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return path


def test_relative_images_resolve_from_manifest_preserving_target(tmp_path, monkeypatch):
    Image.new("L", (200, 40), 255).save(tmp_path / "line.png")
    path = write_manifest(tmp_path, [{"id": "one", "text": "سلام ١٢", "image": "line.png"}])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rows = benchmark.validate_line_manifest(path)
    assert rows == [{"id": "one", "text": "سلام ١٢", "image": str(tmp_path / "line.png")}]


@pytest.mark.parametrize("record", [[], {"id": "one", "text": 123, "image": "a.png"},
    {"id": "one", "text": "  ", "image": "a.png"},
    {"id": "one", "text": "a\nb", "image": "a.png"},
    {"id": "one", "text": "a\u2028b", "image": "a.png"},
    {"text": "one", "image": "a.png"}, {"id": "one", "text": "one"}])
def test_malformed_records_are_rejected_before_evaluation(tmp_path, record):
    path = write_manifest(tmp_path, [record])
    with pytest.raises(ValueError, match=r"references.jsonl:1:"):
        benchmark.validate_line_manifest(path)


def test_empty_and_invalid_json_are_rejected(tmp_path):
    path = write_manifest(tmp_path, [])
    with pytest.raises(ValueError, match="empty"):
        benchmark.validate_line_manifest(path)
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        benchmark.validate_line_manifest(path)


def test_duplicate_ids_and_page_height_are_rejected(tmp_path):
    Image.new("L", (200, 40), 255).save(tmp_path / "line.png")
    record = {"id": "one", "text": "one", "image": "line.png"}
    path = write_manifest(tmp_path, [record, record])
    with pytest.raises(ValueError, match="duplicate id"):
        benchmark.validate_line_manifest(path)
    Image.new("L", (500, 700), 255).save(tmp_path / "page.png")
    path = write_manifest(tmp_path, [{**record, "image": "page.png"}])
    with pytest.raises(ValueError, match="height 700"):
        benchmark.validate_line_manifest(path)


def test_run_snapshots_paths_and_records_source_hash(tmp_path, monkeypatch):
    Image.new("L", (200, 40), 255).save(tmp_path / "line.png")
    manifest = write_manifest(tmp_path, [{"id": "one", "text": "سلام", "image": "line.png"}])
    original = manifest.read_bytes()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "results"
    def fake_run(command, **kwargs):
        assert Path(command[3]) == PROJECT / "evaluate.py"
        assert kwargs == {"cwd": PROJECT, "check": True}
        snapshot = Path(command[command.index("--manifest") + 1])
        assert json.loads(snapshot.read_text(encoding="utf-8"))["image"] == str(tmp_path / "line.png")
        (output / "metrics.json").write_text('{"cer": 0.0, "samples": 1}', encoding="utf-8")
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark.run_benchmark(checkpoint, manifest, output, device="cpu")
    assert result["source_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["samples_validated"] == 1
    assert "heuristics" in result["line_validation"]
    assert manifest.read_bytes() == original
    assert (output / "benchmark_verification.json").is_file()


def test_subprocess_failure_does_not_write_success_verification(tmp_path, monkeypatch):
    Image.new("L", (200, 40), 255).save(tmp_path / "line.png")
    manifest = write_manifest(tmp_path, [{"id": "one", "text": "test", "image": "line.png"}])
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "output"
    output.mkdir()
    (output / "benchmark_verification.json").write_text('{"old": true}', encoding="utf-8")
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_benchmark(checkpoint, manifest, output)
    assert not (output / "benchmark_verification.json").exists()
