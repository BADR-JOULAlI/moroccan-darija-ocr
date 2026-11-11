"""Local Gradio interface for line recognition and experimental page OCR.

Install optional UI dependencies with ``pip install -r requirements-demo.txt``.
The server binds to localhost; ``--share`` explicitly enables a public link.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent
LINE_MODE = "Une ligne"
PAGE_MODE = "Page simple (expérimental)"


def recognize_image(recognizer, image: Image.Image | None, mode: str) -> tuple[str, dict]:
    """Run either workflow with the same loaded recognizer as the CLI."""
    if image is None:
        raise ValueError("Chargez une image avant de lancer la reconnaissance.")
    if not isinstance(image, Image.Image):
        raise ValueError("Le fichier fourni doit être une image lisible.")
    if mode == LINE_MODE:
        text = recognizer.recognize_line(image)
        return text, {"mode": "line", "checkpoint_epoch": recognizer.epoch,
                      "device": str(recognizer.device)}
    if mode == PAGE_MODE:
        result = recognizer.recognize_page(image)
        return result["text"], result
    raise ValueError("Choisissez le mode ligne ou page simple.")


def build_demo(checkpoint: Path, device_name: str = "auto", batch_size: int = 16):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "Gradio est absent. Installez la démo : python -m pip install -r requirements-demo.txt"
        ) from error
    from recognizer import OCRRecognizer

    recognizer = OCRRecognizer(checkpoint, device=device_name, batch_size=batch_size)

    def recognize(image, mode):
        try:
            return recognize_image(recognizer, image, mode)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except RuntimeError as error:
            raise gr.Error(
                "La reconnaissance a échoué. Vérifiez la taille de l'image et les ressources du périphérique."
            ) from error

    return gr.Interface(
        fn=recognize,
        inputs=[gr.Image(type="pil", label="Image à transcrire"),
                gr.Radio([LINE_MODE, PAGE_MODE], value=LINE_MODE, label="Type d'image")],
        outputs=[gr.Textbox(label="Transcription", lines=8, rtl=True),
                 gr.JSON(label="Lignes détectées et informations du modèle")],
        title="Moroccan Darija OCR",
        description=(
            "Chargez une ligne ou une page simple à une colonne. La segmentation des pages est expérimentale : "
            "les tableaux et les colonnes multiples peuvent être mal lus. Les résultats Darija du modèle "
            "ont été mesurés sur des images synthétiques."
        ),
        analytics_enabled=False,
        flagging_mode="never",
        concurrency_limit=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/baseline/best.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="créer volontairement un lien public")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint introuvable : {args.checkpoint}")
    if args.batch_size < 1:
        parser.error("--batch-size doit être positif")
    if not 1 <= args.port <= 65535:
        parser.error("--port doit être compris entre 1 et 65535")
    try:
        demo = build_demo(args.checkpoint, args.device, args.batch_size)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=args.share, show_error=False,
    )


if __name__ == "__main__":
    main()
