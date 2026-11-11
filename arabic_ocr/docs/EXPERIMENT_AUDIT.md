# Audit de reproductibilité des expériences historiques

Audit local du 11 novembre 2026 sur les manifests et journaux conservés. Les métriques déjà enregistrées restent
disponibles ; leur interprétation doit tenir compte des limites suivantes.

## Essais d'augmentation

Le générateur historique utilisait un même RNG pour transformer les images puis mélanger les lignes. Changer les
transformations modifiait donc aussi les phrases en validation. Sur les 533 phrases de validation sans augmentation,
seules 56 sont communes avec l'essai fort et 63 avec l'essai léger. Les CER 6,63 %, 9,52 % et 7,52 % ne constituent
pas une comparaison contrôlée de l'effet de l'augmentation. Ils ne prouvent pas qu'une force particulière est meilleure.

## Expérience mixte

| Split | Gazette réel | Atlas synthétique | Total |
|---|---:|---:|---:|
| Train | 10 131 | 4 796 | 14 927 |
| Validation | 1 152 | 533 | 1 685 |

DODa ne fait pas partie de cette expérience. Les segments Atlas étaient séparés après découpage des paragraphes :
397 des 399 lignes source Atlas représentées en validation ont aussi un segment en train. Ce recouvrement a été
retrouvé par correspondance des textes avec `atlas_texts_filtered.jsonl` et son `source_row_idx`. Les métriques
mixtes restent exploratoires, sans validation indépendante par document Darija. Le contrôle SHA-256 des images
locales trouve néanmoins zéro image binaire identique entre les deux splits mixtes.

Le CLI demandait un LR de 0,0005 ; la restauration de l'optimiseur depuis `init.pt` a imposé **0,001**, enregistré
dans `history.jsonl`. Le premier passage appelé `untrained_validation` évaluait déjà le modèle arabe pré-entraîné
avec classifieur étendu. Il ne s'agissait pas d'un modèle initialisé aléatoirement.

## Rendu arabe

La version historique appelait `arabic_reshaper.reshape()` avec sa configuration par défaut, qui supprime les harakat.
Les références conservaient ces marques : 2 390 des 4 796 labels Atlas filtrés train en contiennent. Certains labels
pouvaient ainsi demander au modèle des signes non rendus. Le rendu corrigé les conserve ; les anciens checkpoints
n'ont pas été réentraînés dans cette livraison.

## Corrections pour les prochaines générations

Le générateur sépare désormais la sélection des splits de l'aléatoire du rendu, garde les segments d'une même source
ensemble et préserve leur provenance. L'augmentation s'applique uniquement au train ; le rendu de validation reste
identique à graine, corpus et polices fixes. Il conserve les diacritiques dans le rendu. La fusion vérifie également les
hashes train contre validation et refuse les chemins d'image absents. Un nouveau corpus doit être généré pour
profiter de ces corrections ; elles ne rendent pas rétroactivement indépendants les anciens jeux.

La prochaine comparaison devra réutiliser les mêmes groupes train/validation, conserver la validation fixe,
contrôler l'alignement rendu/label, enregistrer le LR effectivement utilisé et publier les résultats séparément par
source. AtlasOCRBench reste réservé ; aucune nouvelle performance sur ce benchmark n'est annoncée ici.
