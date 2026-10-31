"""Generate a factual French report and a static learning curve from saved results.

Run after evaluate.py, for example:
    python -X utf8 summarize_run.py --run-dir runs/baseline
No model is loaded and no CUDA computation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import edit_distance


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Résultat absent : {path}. Terminer l'entraînement et l'évaluation avant le rapport.")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Résultat absent : {path}. Terminer l'entraînement et l'évaluation avant le rapport.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def completed_epochs(history):
    """Keep the final saved record for each completed epoch if a run was resumed."""
    records = {row["epoch"]: row for row in history if row.get("epoch", 0) > 0 and "validation" in row}
    if not records:
        raise ValueError("Aucune époque terminée dans history.jsonl ; le rapport final attend un entraînement réel.")
    return [records[epoch] for epoch in sorted(records)]


def percent(value):
    return f"{100 * float(value):.2f}".replace(".", ",") + " %"


def number(value, decimals=0):
    if value is None or isinstance(value, str) and value == "inconnu":
        return "non renseigné"
    return f"{float(value):,.{decimals}f}".replace(",", " ").replace(".", ",")


def duration(seconds):
    return f"{number(seconds, 1)} s ({number(seconds / 60, 2)} min)"


def relative_link(path: Path, project: Path):
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def plot_learning_curve(history, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = completed_epochs(history)
    baseline = next((row for row in history if row.get("epoch") == 0 and "cer" in row), None)
    points = [(row["epoch"], row["validation"]) for row in epochs]
    if baseline is not None:
        points.insert(0, (0, baseline))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
        x = [epoch for epoch, _ in points]
        axes[0].plot(x, [100 * metrics["cer"] for _, metrics in points], "o-", color="#1764a0", label="CER validation")
        axes[0].plot(x, [100 * metrics["wer"] for _, metrics in points], "s-", color="#c4552b", label="WER validation")
        axes[0].set(xlabel="Époque (0 : modèle initial)", ylabel="Erreur (%)", title="Reconnaissance sur la validation", ylim=(0, None))
        axes[0].legend(frameon=False)
        axes[1].plot([row["epoch"] for row in epochs], [row["train_ctc_loss"] for row in epochs],
                     "o-", color="#23724c")
        axes[1].set(xlabel="Époque", ylabel="Perte CTC moyenne", title="Apprentissage", ylim=(0, None))
        for axis in axes:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)


def choose_examples(predictions):
    """Use the 40th, 50th, 60th percentiles of all per-line errors, without manual curation."""
    if not predictions:
        raise ValueError("predictions.jsonl ne contient aucune prédiction.")
    ranked = sorted(
        predictions,
        key=lambda row: (edit_distance(row["reference"], row["prediction"]) / max(1, len(row["reference"])), row["id"]),
    )
    chosen = []
    indices = set()
    for percentile in (0.4, 0.5, 0.6):
        index = round((len(ranked) - 1) * percentile)
        if index not in indices:
            row = ranked[index]
            cer = edit_distance(row["reference"], row["prediction"]) / max(1, len(row["reference"]))
            chosen.append((round(percentile * 100), row, cer))
            indices.add(index)
    return chosen


def quote_text(value):
    # Prefix each line so source text cannot introduce Markdown structure.
    return "\n".join("> " + line for line in str(value).replace("\r", "").split("\n"))


def build_report(config, history, audit, test, predictions, run_dir, project, tests_result=None):
    epochs = completed_epochs(history)
    required = ("samples", "cer", "wer", "exact_accuracy", "checkpoint_epoch", "seconds", "lines_per_second")
    missing = [field for field in required if field not in test]
    if missing:
        raise ValueError("Métriques de test incomplètes : " + ", ".join(missing))
    if len(predictions) != int(test["samples"]):
        raise ValueError("Le nombre de prédictions diffère du nombre de lignes évaluées.")
    selected_epoch = test["checkpoint_epoch"]
    selected = next((row for row in epochs if row["epoch"] == selected_epoch), None)
    if selected is None:
        raise ValueError(f"L'époque du checkpoint testé ({selected_epoch}) manque dans history.jsonl.")
    best_record = min(epochs, key=lambda row: row["validation"]["cer"])
    if selected["validation"]["cer"] != best_record["validation"]["cer"]:
        raise ValueError("Le checkpoint testé ne correspond pas au meilleur CER de validation enregistré.")
    validation = selected["validation"]
    baseline = next((row for row in history if row.get("epoch") == 0 and "cer" in row), None)
    train_seconds = sum(row["train_seconds"] for row in epochs)
    val_seconds = sum(row["validation"]["seconds"] for row in epochs)
    peak_memory = max(row.get("peak_cuda_memory_gb", 0) for row in epochs)
    env = config.get("environment", {})
    counts = audit.get("counts", {})
    quarantined = audit.get("quarantined_train_records", [])
    run = relative_link(run_dir, project)
    train_count = config.get("train_samples", counts.get("train", "inconnu"))
    val_count = config.get("validation_samples", counts.get("val", "inconnu"))
    lines = [
        "# Résultats : OCR arabe sur RTX 5070",
        "",
        f"Le checkpoint choisi à l'époque **{selected_epoch}** obtient un **CER de {percent(test['cer'])}**, "
        f"un **WER de {percent(test['wer'])}** et **{percent(test['exact_accuracy'])} de lignes exactes** "
        f"sur les **{number(test['samples'])} lignes du test officiel**. "
        "Il s'agit d'une expérience sur l'arabe administratif imprimé du Bulletin officiel marocain ; aucune performance en darija n'est mesurée.",
        "",
        "## Protocole et matériel",
        "",
        f"Exécution réelle sur **{env.get('gpu', 'matériel non enregistré')}**, "
        f"avec {number(env.get('gpu_total_gb'), 2)} Gio de mémoire GPU détectée, "
        f"Python {env.get('python', '?')}, PyTorch {env.get('torch', '?')} et CUDA {env.get('cuda_runtime', '?')}. "
        f"Le modèle CNN + BiLSTM + CTC comporte **{number(config.get('parameters'))} paramètres**. "
        f"Le décodage est glouton, sans modèle de langue. La graine est {config.get('seed', '?')}.",
        "",
        "| Élément | Mesure enregistrée |",
        "|---|---:|",
        f"| Lignes d'entraînement | {number(train_count)} |",
        f"| Lignes de validation | {number(val_count)} |",
        f"| Lignes de test évaluées | {number(test['samples'])} |",
        f"| Caractères du vocabulaire, sans blanc CTC | {number(counts.get('vocabulary'))} |",
        f"| Époques terminées | {len(epochs)} |",
        f"| Taille du lot | {config.get('batch_size', '?')} |",
        f"| Hauteur / largeur maximale | {config.get('height', '?')} / {config.get('max_width', '?')} px |",
        f"| AMP float16 | {'activée' if config.get('amp') else 'désactivée'} |",
        f"| Durée entraînement + validation | {duration(train_seconds + val_seconds)} |",
        f"| Dont entraînement | {duration(train_seconds)} |",
        f"| Dont validations des époques | {duration(val_seconds)} |",
        f"| Pic de mémoire CUDA allouée par PyTorch | {number(peak_memory, 3)} Gio |",
        f"| Temps du test | {duration(test['seconds'])} |",
        f"| Débit du test | {number(test['lines_per_second'], 2)} lignes/s |",
        "",
        "La durée d'apprentissage additionne les temps d'entraînement et de validation des époques terminées. "
        "Elle exclut préparation des données, échauffement CUDA, évaluation initiale et test. "
        "Le débit du test inclut lecture et décodage des PNG, prétraitement, calcul du modèle, décodage CTC et métriques ; "
        "l'échauffement CUDA est exclu. La mémoire indiquée mesure les allocations PyTorch, pas la mémoire totale utilisée sur la carte.",
        "",
        "## Progression et sélection du checkpoint",
        "",
    ]
    if baseline is not None:
        improvement = 100 * (baseline["cer"] - validation["cer"])
        lines += [f"Le CER de validation passe de **{percent(baseline['cer'])} avant apprentissage** à "
                  f"**{percent(validation['cer'])} au meilleur checkpoint** "
                  f"({number(improvement, 2)} points de pourcentage de diminution). "
                  f"Le WER de validation à cette époque est de {percent(validation['wer'])}.", ""]
    else:
        lines += ["L'évaluation du modèle initial n'est pas enregistrée ; aucune amélioration par rapport à un modèle non entraîné n'est chiffrée.", ""]
    lines += [
        "Le checkpoint est sélectionné uniquement sur le CER de validation. "
        "Le test officiel est évalué séparément avec ce checkpoint.",
        "",
        f"![Courbes de validation et perte d'entraînement]({run}/learning_curve.png)",
        "",
        "| Époque | Perte CTC entraînement | CER validation | WER validation |",
        "|---:|---:|---:|---:|",
    ]
    for row in epochs:
        lines.append(f"| {row['epoch']} | {number(row['train_ctc_loss'], 4)} | "
                     f"{percent(row['validation']['cer'])} | {percent(row['validation']['wer'])} |")
    lines += [
        "",
        "## Préparation des données et limites",
        "",
        f"Source : [Moroccan Official Gazette, Zenodo](https://zenodo.org/records/21702529), "
        f"DOI {audit.get('doi', 'non enregistré')}, licence annoncée {audit.get('license', 'non enregistrée')}. "
        f"Les {number(counts.get('official_test', test['samples']))} fichiers du test officiel sont exclus du corpus d'apprentissage. "
        f"La déduplication retire {number(counts.get('removed_test_content_overlap'))} lignes recouvrant le test puis "
        f"{number(counts.get('removed_val_content_overlap_from_train'))} lignes recouvrant la validation. "
        "Les doublons sont recherchés par nom, SHA-256 d'image et transcription normalisée. La validation est séparée par groupes de pages.",
        "",
        f"L'audit compte **{audit.get('cross_split_id_image_hash_transcript_overlap', 'non renseigné')} chevauchement "
        f"entre splits par nom/image/texte**, mais **{audit.get('train_test_page_overlap', 'non renseigné')} pages communes "
        f"entre entraînement et test** et {audit.get('val_test_page_overlap', 'non renseigné')} entre validation et test. "
        "Des documents sont donc représentés dans plusieurs splits ; ces résultats ne démontrent pas une généralisation à des documents entièrement nouveaux.",
        "",
        f"Pour **{len(audit.get('test_annotation_mismatches', []))} fichiers**, la transcription du test officiel diffère de celle de "
        "`merged.jsonl`. La transcription officielle est conservée après la même normalisation NFC et réduction des espaces. "
        "Les diacritiques, le tatweel et les signes non blancs sont conservés.",
        "",
    ]
    if quarantined:
        lines += [f"**{len(quarantined)} exemples sont mis en quarantaine uniquement dans l'entraînement**, "
                  "après inspection des images ; validation et test restent inchangés :", ""]
        for row in quarantined:
            sample_id = row.get("id", row.get("image", "identifiant absent"))
            reason = row.get("reason", "raison non enregistrée")
            lines.append(f"- `{sample_id}` : {reason}.")
        lines.append("")
    else:
        lines += ["Aucune liste de quarantaine d'entraînement n'est enregistrée dans l'audit de ce run.", ""]
    oov = audit.get("eval_oov", {})
    if oov:
        lines += [f"Caractères hors vocabulaire : {oov.get('val', {}).get('num_characters', '?')} en validation et "
                  f"{oov.get('test', {}).get('num_characters', '?')} au test. Le vocabulaire provient uniquement de l'entraînement.", ""]
    lines += [
        "Le parcours global de droite à gauche des caractéristiques reste une approximation pour les lignes mêlant arabe, chiffres et segments latins. "
        "Certains crops contiennent des défauts de scan ou des annotations imparfaites. Les images longues sont comprimées à la largeur maximale indiquée. "
        "Une adaptation à la darija demande un jeu représentatif d'images et de transcriptions et une évaluation distincte.",
        "",
        "## Exemples de prédictions",
        "",
        "Exemples illustratifs choisis mécaniquement aux 40e, 50e et 60e percentiles du CER par ligne "
        "après tri de tout le test par CER puis identifiant. Il n'y a pas de sélection manuelle des meilleures lignes. "
        "Le CER par ligne sert seulement à ce choix ; les CER et WER du rapport sont calculés sur le corpus entier.",
        "",
    ]
    for percentile, row, cer in choose_examples(predictions):
        lines += [f"**{percentile}e percentile — `{row['id']}` — CER ligne {percent(cer)}**", "",
                  "Référence :", "", quote_text(row["reference"]), "", "Prédiction :", "", quote_text(row["prediction"]), ""]
    lines += [
        f"Toutes les prédictions : [predictions.jsonl]({run}/test/predictions.jsonl). "
        f"Métriques complètes : [metrics.json]({run}/test/metrics.json). "
        f"Configuration : [config.json]({run}/config.json). "
        f"Audit des données : [data_audit.json]({run}/data_audit.json).",
        "",
    ]
    if tests_result is not None:
        passed = tests_result.get("passed")
        failed = tests_result.get("failed", "non renseigné")
        skipped = tests_result.get("skipped", "non renseigné")
        lines += ["## Vérifications du code", "",
                  f"Résultat enregistré : **{passed if passed is not None else 'non renseigné'} tests réussis**, "
                  f"{failed} en échec, {skipped} ignorés. "
                  f"Détail : [tests_result.json]({run}/tests_result.json).", ""]
    return "\n".join(lines)


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=project / "runs/baseline")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = read_json(run_dir / "config.json")
    history = read_jsonl(run_dir / "history.jsonl")
    audit = read_json(run_dir / "data_audit.json")
    test = read_json(run_dir / "test/metrics.json")
    predictions = read_jsonl(run_dir / "test/predictions.jsonl")
    tests_path = run_dir / "tests_result.json"
    tests_result = read_json(tests_path) if tests_path.is_file() else None
    report = build_report(config, history, audit, test, predictions, run_dir, project, tests_result)
    plot_learning_curve(history, run_dir / "learning_curve.png")
    (project / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(project / "RESULTS.md"), "curve": str(run_dir / "learning_curve.png")}))


if __name__ == "__main__":
    main()
