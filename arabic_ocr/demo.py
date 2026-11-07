"""Small Gradio demo for line-image OCR."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from data import collate_lines, preprocess
from evaluate import load_model
from model import greedy_decode


def build_demo(checkpoint: Path, device_name: str):
    device = torch.device(device_name)
    model, characters, config, epoch = load_model(checkpoint, device)

    def recognize(image):
        if image is None:
            return ""
        pixels, width = preprocess(Image.fromarray(image), config["height"], config["max_width"])
        batch = collate_lines([{"pixels": pixels, "width": width, "target": [], "text": "", "id": "upload"}])
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            log_probs, lengths = model(batch["images"].to(device), batch["widths"])
        return greedy_decode(log_probs, lengths, characters)[0]

    return gr.Interface(fn=recognize, inputs=gr.Image(type="numpy", label="Image d'une ligne"),
                        outputs=gr.Textbox(label=f"Transcription (checkpoint epoch {epoch})"),
                        title="Moroccan Darija OCR", description="Chargez une image contenant une seule ligne de texte.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mixed_darija/best.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo(args.checkpoint, args.device).launch(share=args.share)


if __name__ == "__main__":
    main()
