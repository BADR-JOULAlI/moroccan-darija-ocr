"""Compact CNN + BiLSTM recognizer for right-to-left Arabic text lines.

The input image keeps its ordinary visual orientation.  Only its valid CNN
features are reversed, so CTC can learn transcriptions in Unicode logical
order.  No Arabic reshaping or visual-order conversion is applied to labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int], *, double: bool = False):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if double:
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.MaxPool2d(pool, stride=pool))
        super().__init__(*layers)


class CRNN(nn.Module):
    """Recognize variable-width grayscale lines; CTC blank has class index 0.

    ``images`` has shape [batch, 1, height, padded_width]. ``widths`` contains
    each resized image's width before right padding. Height must be >=16 and
    widths >=4. Horizontal downsampling is exactly floor(width / 4).
    """

    def __init__(self, num_classes: int, hidden_size: int = 128, channels: int = 32, dropout: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include blank and at least one character")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.channels = channels
        self.dropout = dropout
        self.cnn = nn.Sequential(
            ConvBlock(1, channels, (2, 2)),
            ConvBlock(channels, channels * 2, (2, 2)),
            ConvBlock(channels * 2, channels * 4, (2, 1), double=True),
            ConvBlock(channels * 4, channels * 8, (2, 1), double=True),
        )
        self.rnn = nn.LSTM(
            channels * 8, hidden_size, num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        # Avoid an artificial preference for blank before seeing any examples.
        with torch.no_grad():
            self.classifier.bias[0] = -1.0

    @staticmethod
    def output_lengths(widths: Tensor) -> Tensor:
        return torch.div(widths.to(device="cpu", dtype=torch.long), 4, rounding_mode="floor")

    @staticmethod
    def reverse_valid(features: Tensor, lengths: Tensor) -> Tensor:
        """Reverse valid [B,T,C] features without moving padding to the front."""
        positions = torch.arange(features.size(1), device=features.device)[None, :]
        reverse_indices = (lengths.to(features.device)[:, None] - 1 - positions).clamp_min(0)
        return features.gather(1, reverse_indices[:, :, None].expand(-1, -1, features.size(2)))

    def forward(self, images: Tensor, widths: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.size(1) != 1 or images.size(2) < 16:
            raise ValueError("Expected grayscale BCHW images with height >=16")
        widths = widths.to(device="cpu", dtype=torch.long)
        if widths.numel() != images.size(0) or torch.any(widths < 4) or torch.any(widths > images.size(3)):
            raise ValueError("widths must contain one valid unpadded width >=4 per image")
        lengths = self.output_lengths(widths)
        features = self.cnn(images).mean(dim=2).transpose(1, 2)  # B,T,C
        features = self.reverse_valid(features, lengths)
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        recurrent, _ = self.rnn(packed)
        recurrent, _ = pad_packed_sequence(recurrent, batch_first=True, total_length=features.size(1))
        logits = self.classifier(recurrent)
        # CTCLoss is evaluated in float32 even under CUDA mixed precision.
        return logits.float().log_softmax(dim=-1).transpose(0, 1).contiguous(), lengths


def greedy_decode(log_probs: Tensor, lengths: Tensor, characters: Sequence[str]) -> list[str]:
    """Collapse repeated classes and remove blank; retain repeated letters across blank.

    ``characters[0]`` corresponds to class 1. The blank is deliberately absent
    from this character list, matching the dataset vocabulary JSON.
    """
    if log_probs.ndim != 3 or log_probs.size(2) != len(characters) + 1:
        raise ValueError("Expected [time,batch,len(characters)+1] log probabilities")
    if len(lengths) != log_probs.size(1):
        raise ValueError("One decoding length is required for each sample")
    best = log_probs.detach().argmax(dim=-1).transpose(0, 1).cpu().tolist()
    result: list[str] = []
    for row, length in zip(best, lengths.cpu().tolist()):
        if length < 0 or length > len(row):
            raise ValueError("Decoding length falls outside the time dimension")
        previous = -1
        text: list[str] = []
        for index in row[:length]:
            if index != 0 and index != previous:
                text.append(characters[index - 1])
            previous = index
        result.append("".join(text))
    return result
