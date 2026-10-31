"""Recognize one already-cropped line image using a local checkpoint."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
import torch

from data import preprocess, collate_lines
from evaluate import load_model
from model import greedy_decode


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, characters, config, epoch = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        pixels, width = preprocess(image, config["height"], config["max_width"])
    batch = collate_lines([{"pixels": pixels, "width": width, "target": [], "text": "", "id": args.image.name}])
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                               enabled=device.type == "cuda" and not args.no_amp):
        log_probs, lengths = model(batch["images"].to(device), batch["widths"])
    print(json.dumps({"image": str(args.image), "prediction": greedy_decode(log_probs, lengths, characters)[0],
                      "checkpoint_epoch": epoch}, ensure_ascii=False))


if __name__ == "__main__":
    main()
