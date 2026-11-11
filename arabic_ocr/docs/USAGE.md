# Utiliser le modèle et la démo

Exécuter les commandes depuis `arabic_ocr/` avec Python 3.12 et les dépendances de `requirements.txt` installées.
Le checkpoint arabe `runs/baseline/best.pt` est inclus dans le dépôt. Il traite des lignes imprimées arabes ; les
performances publiées en Darija concernent encore des images synthétiques.

## Démo locale

```powershell
python -m pip install -r requirements-demo.txt
python -X utf8 demo.py --device auto
```

Ouvrir `http://127.0.0.1:7860`. Charger une image puis choisir « Une ligne » ou « Page simple (expérimental) ».
Le texte et les coordonnées des lignes sont affichés. Le modèle est chargé une seule fois et les demandes sont
exécutées l'une après l'autre. `auto` choisit CUDA si disponible, sinon CPU. `--device cpu` force le mode CPU.
Le serveur reste local par défaut ; `--share` permet explicitement de créer un lien public. `--port 7861` change le port.

## Utiliser le checkpoint mixte

Le checkpoint mixte est publié séparément sur Hugging Face ; il n'est pas inclus dans le clone GitHub.

```powershell
python -m pip install huggingface_hub
hf download sailu4/moroccan-darija-ocr experiments/mixed_darija/best.pt --local-dir model_cache
python -X utf8 demo.py --checkpoint model_cache/experiments/mixed_darija/best.pt --device auto
```

Charger uniquement les checkpoints de confiance : le format historique contient l'optimiseur et utilise le chargement
PyTorch avec pickle. L'[audit des expériences](EXPERIMENT_AUDIT.md) détaille les limites des métriques Darija actuelles.

## Exporter une page

```powershell
python -X utf8 page_ocr.py --checkpoint runs/baseline/best.pt --image page.png --output results/page.json --text-output results/page.txt --batch-size 8 --device auto
```

Le JSON contient `text`, `lines`, `warnings` et l'époque du checkpoint. Chaque ligne possède un `index`, une
`box = [gauche, haut, droite, bas]` et son texte. Les coordonnées concernent l'image après correction d'orientation EXIF.
Les limites droite/bas sont exclusives. Les champs `top`/`bottom` restent disponibles pour compatibilité.
Une page blanche produit zéro ligne et un texte vide. Les images transparentes sont compositées sur fond blanc.

La segmentation repose sur un seuil de gris et la projection horizontale, avec filtrage des pixels isolés et
regroupement de petits points proches des caractères. Elle reste expérimentale : pages fortement bruitées,
photographies inclinées, tableaux, colonnes et texte manuscrit peuvent échouer. Les images sont limitées à
20 millions de pixels et 200 lignes candidates. La mémoire GPU dépend de `--batch-size`, qui limite le nombre
de lignes traitées ensemble. Un avertissement indique les lignes comprimées à la largeur maximale du modèle.

## Vérification reproductible locale

Créer une page de contrôle à partir de trois images de lignes disponibles :

```powershell
python -X utf8 scripts/smoke_page_ocr.py --manifest chemin/train.jsonl --checkpoint runs/baseline/best.pt --output-dir work/page_smoke --device auto
python -m pytest -q tests
```

Le script produit `page.png`, `detected_lines.png` et `report.json`. Ce contrôle vérifie le chemin de calcul et le
nombre de lignes. Les trois lignes assemblées ne constituent pas un benchmark de documents réels.

## Évaluation de lignes réservées

```powershell
python -X utf8 evaluate_benchmark.py --checkpoint runs/baseline/best.pt --manifest chemin/test_lignes.jsonl --output-dir work/line_eval --device cpu
```

Chaque objet JSONL doit contenir un `id` unique et non vide, `image` (absolu ou relatif au manifest), et `text`
(chaîne non vide sur une ligne). Le script conserve un snapshot normalisé et le SHA-256 du manifest source.
Les contrôles de format/hauteur ne prouvent ni l'exactitude des annotations ni l'absence de recouvrement avec train.
Un benchmark de pages doit être évalué après segmentation et reconstruction du texte complet.
