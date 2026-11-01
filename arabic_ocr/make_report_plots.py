"""Create compact plots used by the README and release artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from metrics import edit_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.run_dir / "test" / "metrics.json").read_text(encoding="utf-8"))
    names = ["CER", "WER", "Exactitude"]
    values = [metrics["cer"] * 100, metrics["wer"] * 100, metrics["exact_accuracy"] * 100]
    colors = ["#2878b5", "#e1812c", "#3a923a"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    bars = ax.bar(names, values, color=colors, width=0.58)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_title("Résultats sur le test officiel (1 212 lignes)")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_metrics_summary.png", bbox_inches="tight")
    plt.close(fig)

    errors = []
    for raw in (args.run_dir / "test" / "predictions.jsonl").read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(raw)
        reference = row.get("reference", "")
        prediction = row.get("prediction", "")
        errors.append(100 * edit_distance(list(prediction), list(reference)) / max(1, len(reference)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.hist(errors, bins=[0, 1, 2, 5, 10, 20, 40, 60, 100, 200], color="#2878b5", alpha=0.9, edgecolor="white")
    ax.set_xlabel("CER par ligne (%)")
    ax.set_ylabel("Nombre de lignes")
    ax.set_title("Distribution des erreurs caractère — test officiel")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_cer_distribution.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"summary": str(args.output_dir / "test_metrics_summary.png"), "distribution": str(args.output_dir / "test_cer_distribution.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
