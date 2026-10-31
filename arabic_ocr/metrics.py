"""Corpus-level OCR metrics, calculated on Unicode logical-order strings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # Small, dependency-free fallback for deployment.
    Levenshtein = None

T = TypeVar("T")


def edit_distance(reference: Sequence[T], prediction: Sequence[T]) -> int:
    if Levenshtein is not None:
        return Levenshtein.distance(reference, prediction)
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, actual in enumerate(prediction, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


@dataclass
class Metrics:
    """Micro-averaged edit error rates (fractions; may exceed 1 with insertions)."""

    samples: int = 0
    character_errors: int = 0
    reference_characters: int = 0
    word_errors: int = 0
    reference_words: int = 0
    exact_matches: int = 0

    def update(self, reference: str, prediction: str) -> None:
        self.samples += 1
        self.character_errors += edit_distance(reference, prediction)
        self.reference_characters += len(reference)
        expected_words, predicted_words = reference.split(), prediction.split()
        self.word_errors += edit_distance(expected_words, predicted_words)
        self.reference_words += len(expected_words)
        self.exact_matches += int(reference == prediction)

    def compute(self) -> dict[str, int | float]:
        return {
            "samples": self.samples,
            "cer": self.character_errors / max(1, self.reference_characters),
            "wer": self.word_errors / max(1, self.reference_words),
            "exact_accuracy": self.exact_matches / max(1, self.samples),
            "character_errors": self.character_errors,
            "reference_characters": self.reference_characters,
            "word_errors": self.word_errors,
            "reference_words": self.reference_words,
        }
