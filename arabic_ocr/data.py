"""Line images, logical Unicode targets, and length-aware batching."""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_characters(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)["characters"]


def preprocess(image, height=64, max_width=1600):
    """Resize from pixels only; target text never changes input dimensions."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        foreground = image.convert("RGBA")
        background = Image.new("RGBA", foreground.size, "white")
        image = Image.alpha_composite(background, foreground)
    image = image.convert("L")
    width = max(16, min(max_width, int(round(image.width * height / image.height))))
    width = max(16, 4 * math.ceil(width / 4))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    # White background becomes zero, the same value as right-padding.
    return np.asarray(image, dtype=np.uint8).copy(), width


class LineDataset(Dataset):
    def __init__(self, manifest, characters, height=64, max_width=1600, limit=0, cache=True):
        self.rows = read_jsonl(manifest)
        if limit:
            self.rows = self.rows[:limit]
        self.characters = characters
        self.char_to_id = {char: index + 1 for index, char in enumerate(characters)}
        self.height, self.max_width, self.cache = height, max_width, cache
        self._cache = {}
        self.widths = []
        for row in self.rows:
            with Image.open(row["image"]) as image:
                width = max(16, min(max_width, round(image.width * height / image.height)))
            self.widths.append(max(16, 4 * math.ceil(width / 4)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if index not in self._cache:
            with Image.open(row["image"]) as image:
                array, width = preprocess(image, self.height, self.max_width)
            if self.cache:
                self._cache[index] = array
        else:
            array = self._cache[index]
            width = array.shape[1]
        # Evaluation can contain unseen symbols. Keep their references for CER;
        # only training uses encoded targets, and explicitly rejects any OOV.
        target = [self.char_to_id.get(char, -1) for char in row["text"]]
        return {"pixels": array, "width": width, "target": target,
                "text": row["text"], "id": row["id"]}


def collate_lines(rows):
    widths = torch.tensor([row["width"] for row in rows], dtype=torch.long)
    images = torch.zeros(len(rows), 1, rows[0]["pixels"].shape[0], int(widths.max()))
    for i, row in enumerate(rows):
        images[i, 0, :, :row["width"]] = 1 - torch.from_numpy(row["pixels"]).float() / 255
    return {"images": images, "widths": widths,
            "targets": torch.tensor([c for row in rows for c in row["target"]], dtype=torch.long),
            "target_lengths": torch.tensor([len(row["target"]) for row in rows], dtype=torch.long),
            "texts": [row["text"] for row in rows], "ids": [row["id"] for row in rows]}


class WidthBatchSampler(Sampler):
    """Shuffle broad width buckets to reduce padded convolution work."""
    def __init__(self, widths, batch_size, shuffle=True, seed=42):
        self.widths, self.batch_size, self.shuffle = widths, batch_size, shuffle
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return math.ceil(len(self.widths) / self.batch_size)

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = np.argsort(self.widths).tolist()
        buckets = [indices[i:i + self.batch_size * 20]
                   for i in range(0, len(indices), self.batch_size * 20)]
        batches = []
        for bucket in buckets:
            if self.shuffle:
                generator.shuffle(bucket)
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches


def assert_ctc_feasible(texts, widths):
    for text, width in zip(texts, widths):
        steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if width // 4 < steps:
            raise ValueError(f"CTC infeasible: {width // 4} time steps < {steps}; increase --max-width/--height")
