"""Render the project architecture as a publication-ready PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


BLUE = "#173f73"
MID = "#dcecff"
LIGHT = "#f7fbff"
GREEN = "#dff2e5"
ORANGE = "#fff0d6"
GRAY = "#5e6b78"


def box(ax, x, y, w, h, label, fill=MID, edge=BLUE, size=10, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           linewidth=1.4, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
            color="#15283d", weight=weight, wrap=True)
    return patch


def arrow(ax, x1, y1, x2, y2, label=None, color=GRAY, dashed=False):
    style = "-|>"
    line = "--" if dashed else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 linewidth=1.4, linestyle=line, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.025, label, ha="center", va="bottom",
                fontsize=8, color=GRAY, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=1.4, edgecolor="#8da7c7", facecolor="none"))
    ax.text(x + 0.06, y + h - 0.22, title, color=BLUE, fontsize=12, weight="bold", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})


def main() -> None:
    output = Path(__file__).resolve().parent / "assets" / "project_architecture.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=150)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    panel(ax, 0.25, 5.25, 17.5, 4.85, "(A) Model architecture — inference")
    labels = ["Image de ligne\n(hauteur 64)", "Prétraitement\nNiveaux de gris + padding", "CNN\nExtraction visuelle", "RTL\nAxe temporel valide", "BiLSTM\n2 couches, 128 unités", "Projection linéaire\nclasses + blank", "CTC greedy\ndecode", "Transcription"]
    xs = [0.55, 2.55, 4.7, 6.65, 8.55, 10.75, 13.05, 15.25]
    ws = [1.45, 1.7, 1.45, 1.35, 1.55, 1.65, 1.45, 1.45]
    for x, w, label in zip(xs, ws, labels):
        box(ax, x, 7.05, w, 0.95, label, fill=LIGHT if "Image" in label or "Transcription" in label else MID, size=9, weight="bold" if "CNN" in label or "BiLSTM" in label else "normal")
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + ws[i], 7.52, xs[i + 1], 7.52)
    ax.text(9.0, 6.25, "Une ligne image → une séquence de caractères", ha="center", fontsize=10, color=BLUE, style="italic")

    panel(ax, 0.25, 0.35, 17.5, 4.45, "(B) Data, training, evaluation and release")
    box(ax, 0.6, 3.0, 2.05, 0.85, "Bulletin officiel\nArabic baseline", fill=GREEN, size=9)
    box(ax, 0.6, 1.65, 2.05, 0.85, "DODa + AtlasOCR\nDarija text", fill=ORANGE, size=9)
    box(ax, 3.3, 2.35, 2.2, 1.0, "Text fetch + filtering\nmax 100 chars", fill=MID, size=9)
    box(ax, 6.15, 2.35, 2.2, 1.0, "Synthetic renderer\nArabic fonts + seed", fill=MID, size=9)
    box(ax, 9.0, 2.35, 2.2, 1.0, "Manifest validation\nCTC feasibility + split", fill=MID, size=9)
    box(ax, 11.85, 2.35, 2.2, 1.0, "CTC training\nRTX 5070 / AMP", fill=GREEN, size=9)
    box(ax, 14.7, 3.0, 2.35, 0.85, "CER / WER / exact\nvalidation + test", fill=LIGHT, size=9)
    box(ax, 14.7, 1.65, 2.35, 0.85, "GitHub + Hugging Face\ncode, docs, checkpoints", fill=LIGHT, size=9)
    arrow(ax, 2.65, 3.42, 3.3, 2.85, "real lines")
    arrow(ax, 2.65, 2.07, 3.3, 2.75, "texts")
    arrow(ax, 5.5, 2.85, 6.15, 2.85)
    arrow(ax, 8.35, 2.85, 9.0, 2.85)
    arrow(ax, 11.2, 2.85, 11.85, 2.85)
    arrow(ax, 14.05, 2.85, 14.7, 3.35)
    arrow(ax, 14.05, 2.55, 14.7, 2.05)
    ax.text(9.1, 1.1, "CTC Loss : alignement latent entre les trames CNN/BiLSTM et le texte cible", ha="center", fontsize=9.5, color=BLUE)
    ax.text(9.1, 0.68, "Les images synthétiques valident le pipeline ; elles ne remplacent pas une évaluation sur documents Darija réels.", ha="center", fontsize=8.5, color=GRAY)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
