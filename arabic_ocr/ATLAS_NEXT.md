# Adapter ensuite le reconnaisseur à la darija

L'audit confirme que [atlasia/atlasOCR-data](https://huggingface.co/datasets/atlasia/atlasOCR-data) apporte des images et transcriptions en darija. Son format publié demande toutefois un travail d'alignement avant un apprentissage sur des lignes : les champs sont `text`, `image` et `metadata {contains_title, font}`. Aucun champ de coordonnées ni transcription par ligne n'est exposé dans la version inspectée.

Sur les **20 premiers exemples d'entraînement et 20 de validation**, les textes bruts ne contiennent **aucun saut de ligne**. Ils ont été conservés sans normalisation dans les snapshots de l'audit. Les quatre images examinées visuellement montrent plusieurs lignes, parfois des titres et des traits décoratifs. Découper l'image puis répartir arbitrairement les mots du paragraphe ne produirait pas des annotations fiables.

Le README décrit la génération de coordonnées par OCRSmith ; ces coordonnées ne figurent pas dans le schéma ni dans des fichiers annexes du dépôt inspecté. Cette différence entre l'outil de génération et le contenu effectivement publié est déterminante pour le CNN + BiLSTM + CTC.

## Volumes et séparation des données

| Source | Split | Exemples | Fichiers Parquet compressés |
|---|---|---:|---:|
| atlasOCR-data | train | 26 162 | 8 116 933 767 octets |
| atlasOCR-data | validation | 3 930 | 1 246 909 437 octets |
| atlasOCR-data | test | 196 | 56 217 599 octets |
| AtlasOCRBench | test | 251 | 61 912 188 octets |

Le téléchargement intégral d'atlasOCR-data atteint **9,42 Go décimaux** : il n'a pas été effectué. L'audit utilise des API de métadonnées, des fenêtres de lignes et six petites prévisualisations.

Les exemples de test d'atlasOCR-data doivent rester réservés, comme l'ensemble d'[AtlasOCRBench](https://huggingface.co/datasets/atlasia/AtlasOCRBench). L'audit a établi un chevauchement concret : les textes des **20 premiers exemples de test** sont identiques à ceux des exemples **55 à 74** du benchmark. Pour la paire test 0 / benchmark 55, les prévisualisations d'image ont aussi exactement le même SHA-256. Cela prouve le chevauchement de cet échantillon, sans établir l'identité des 196 exemples. Aucun résultat OCR sur le benchmark n'a servi à cette vérification.

## Obtenir des couples image de ligne / transcription fiables

1. **Chercher les annotations de génération originales.** Obtenir auprès du projet les coordonnées de lignes ou mots et les chaînes exactement rendues par OCRSmith, avec leur version et leur correspondance aux images. Vérifier les boîtes, l'ordre de lecture et les caractères sur un échantillon avant de créer des crops.
2. **À défaut, générer des données synthétiques nouvelles.** Utiliser uniquement les textes du split `train`, les découper explicitement en chaînes de lignes, puis rendre chaque chaîne dans une image indépendante. Enregistrer exactement la chaîne rendue, la police, le moteur de rendu, le traitement bidi, la graine et la ligne source. Ces images seront de nouvelles données synthétiques ; elles ne reconstituent pas les anciennes annotations de mise en page.
3. **Pour les images existantes, valider l'alignement.** Détecter les lignes géométriques et isoler titres et traits décoratifs. Une proposition OCR et un alignement monotone avec le texte du paragraphe peuvent assister le travail, mais chaque limite de crop et chaque chaîne associée doivent être vérifiées avant acceptation. Mettre en quarantaine les cas ambigus, le texte manquant et les mises en page complexes.
4. **Figer les sources avant de produire les crops.** Tous les crops d'une page restent dans le même split. Dédupliquer les candidats par image et texte normalisé contre les ensembles réservés. Étendre le vocabulaire à partir des données d'entraînement approuvées uniquement, puis réinitialiser les nouvelles classes du classifieur avant adaptation.
5. **Évaluer sans contamination.** Constituer une validation en darija avec des couples de lignes vérifiés indépendamment, choisir le modèle sur cette validation et conserver AtlasOCRBench pour une évaluation finale figée. Ses images, textes et erreurs ne doivent pas servir à choisir les alignements, seuils, caractères ou hyperparamètres.

Si les couples de lignes ne peuvent pas être vérifiés, conserver la supervision au niveau paragraphe/page et choisir un modèle adapté au texte multiligne. Un éventuel saut de ligne rencontré ailleurs dans le corpus ne suffit pas, seul, à prouver son alignement sur une ligne visuelle.

## Reproduire cet audit sans télécharger les gros fichiers

Depuis le dossier du projet OCR :

```powershell
python -X utf8 inspect_atlas.py --output-dir ../../work/atlas_audit --findings ATLAS_DATA_AUDIT.json --rows 20
```

Ajouter `--images` pour récupérer au maximum six prévisualisations limitées en taille. Le script ne télécharge aucun shard Parquet et ne lance aucun entraînement. Les réponses brutes, leurs URL, dates et empreintes sont archivées dans le dossier de preuves. Les champs `text` conservent leurs sauts de ligne exacts. Les URL signées de prévisualisation expirent ; un nouvel appel à l'API fournit des liens renouvelés.

Sources figées de cet audit : atlasOCR-data révision `9ecbcbb8c964057f7635ac33032fb89117612154` et AtlasOCRBench révision `1e860eb5c22cbb3cc43ffa721ab36029e6849db3`. [ATLAS_DATA_AUDIT.json](ATLAS_DATA_AUDIT.json) contient les tailles, schémas, statistiques par exemple, preuves du chevauchement et limites de l'inspection.

Le [README d'atlasOCR-data à cette révision](https://huggingface.co/datasets/atlasia/atlasOCR-data/blob/9ecbcbb8c964057f7635ac33032fb89117612154/README.md) affiche un badge Apache 2.0 ; aucun champ YAML de licence ni fichier `LICENSE` séparé n'était présent. Le [README d'AtlasOCRBench](https://huggingface.co/datasets/atlasia/AtlasOCRBench/blob/1e860eb5c22cbb3cc43ffa721ab36029e6849db3/README.md) annonce explicitement Apache 2.0. L'audit rapporte ces déclarations sans réexaminer les droits des sources réelles. Attribution : AtlasIA, Imane Momayiz, Soufiane Ait Elaouad, Abdeljalil Elmajjodi et Haitame Bouanane ; le benchmark crédite également Zaid Chiech.
