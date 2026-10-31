"""Create a darija-adaptation checkpoint from the Gazette checkpoint.

Shared character classifier rows are copied; newly observed characters keep a
fresh Xavier-initialized row. Training still runs through train.py, so the
adaptation manifest and vocabulary are recorded and checked on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from data import load_characters
from model import CRNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Output of prepare_line_manifest.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    old_chars = checkpoint["characters"]
    new_chars = load_characters(args.data_dir / "vocab.json")
    if not new_chars:
        raise ValueError("Adaptation vocabulary is empty")
    if not set(old_chars).issubset(set(new_chars)):
        missing = sorted(set(old_chars) - set(new_chars))
        raise ValueError(f"Adaptation vocabulary removed base characters: {missing[:10]}")
    config = dict(checkpoint["config"])
    model = CRNN(len(new_chars) + 1, hidden_size=config["hidden_size"], channels=config["channels"])
    state = model.state_dict()
    base = checkpoint["model"]
    for key, value in base.items():
        if key not in state or key.startswith("classifier."):
            continue
        state[key] = value
    state["classifier.weight"][0] = base["classifier.weight"][0]
    state["classifier.bias"][0] = base["classifier.bias"][0]
    old_index = {char: index + 1 for index, char in enumerate(old_chars)}
    new_index = {char: index + 1 for index, char in enumerate(new_chars)}
    for char, source in old_index.items():
        target = new_index[char]
        state["classifier.weight"][target] = base["classifier.weight"][source]
        state["classifier.bias"][target] = base["classifier.bias"][source]
    model.load_state_dict(state)
    config["adapted_from"] = str(args.base_checkpoint.resolve())
    config["adaptation_vocabulary"] = len(new_chars)
    config["manifest_sha256"] = {split: hashlib.sha256((args.data_dir / f"{split}.jsonl").read_bytes()).hexdigest() for split in ("train", "val")}
    config["resume_note"] = "Classifier rows for shared characters copied from base; new rows freshly initialized."
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 0.001)), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", True)), init_scale=1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(), "epoch": 0, "best_cer": float("inf"), "config": config, "characters": new_chars, "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": None}}, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "base_characters": len(old_chars), "adapted_characters": len(new_chars), "new_characters": sorted(set(new_chars) - set(old_chars))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
