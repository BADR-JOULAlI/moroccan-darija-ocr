"""Evaluate the validation-selected checkpoint on a frozen manifest."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from data import LineDataset
from model import CRNN
from train import evaluate, environment, make_loader


def load_model(checkpoint_path, device):
    # Load only locally generated trusted checkpoints; optimizer/RNG need pickle.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, characters = checkpoint["config"], checkpoint["characters"]
    model = CRNN(len(characters) + 1, config["hidden_size"], config["channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, characters, config, checkpoint["epoch"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, characters, config, epoch = load_model(args.checkpoint, device)
    dataset = LineDataset(args.manifest, characters, config["height"], config["max_width"])
    loader = make_loader(dataset, args.batch_size, False, args.workers)
    # Warm up CUDA separately, so reported throughput excludes initialization.
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                               enabled=device.type == "cuda" and not args.no_amp):
        batch = next(iter(loader))
        model(batch["images"].to(device), batch["widths"])
    result = evaluate(model, loader, characters, device, not args.no_amp,
                      args.output_dir / "predictions.jsonl")
    result.update({"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": epoch,
                   "manifest": str(args.manifest.resolve()),
                   "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   "environment": environment(device), "decoder": "CTC greedy, no language model",
                   "oov_characters": sorted(set(''.join(row['text'] for row in dataset.rows)) - set(characters))})
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
