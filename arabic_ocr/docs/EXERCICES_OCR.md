# Exercices corrigés et préparation à l'entretien

Compagnon du cours théorique et technique de Moroccan Darija OCR. Les calculs correspondent à `model.py`,
`data.py`, `metrics.py` et `train.py`. Les exemples chiffrés des exercices sont pédagogiques ; ils ne représentent
pas de nouvelles mesures du modèle. Essayer chaque exercice avant de lire son corrigé.

## 1. Redimensionner une ligne sans perdre le rapport d'aspect

**Énoncé.** Une image mesure 1 001 pixels de large et 80 pixels de haut. `preprocess()` utilise une hauteur de 64,
une largeur maximale de 1 600 et arrondit la largeur au multiple de quatre supérieur. Calculer la largeur finale
et le nombre de trames CTC. Que se passe-t-il pour une image de largeur 2 500, de même hauteur ?

**Corrigé.** La largeur proportionnelle vaut `1001 × 64 / 80 = 800,8`. Le code effectue d'abord `round()`, soit 801,
puis `4 × ceil(801 / 4)`, soit **804 pixels**. Le CNN réduit horizontalement par quatre : `T = 804 / 4 = 201` trames.
Une trame est une position dans la séquence de caractéristiques, pas une annotation de caractère.

Pour la seconde image, la largeur proportionnelle serait 2 000, plafonnée à **1 600**. Cette limitation comprime
horizontalement le texte : le rapport d'aspect initial n'est plus conservé. Elle limite le coût de calcul mais peut
réduire la lisibilité. Augmenter seulement la hauteur ne résout pas ce problème lorsque la largeur est déjà plafonnée.

## 2. Suivre les dimensions dans le réseau

**Énoncé.** Pour un lot de quatre images `[4, 1, 64, 800]`, retrouver les dimensions après chaque bloc CNN,
après la moyenne sur la hauteur, après le BiLSTM et après la projection. Le vocabulaire contient 92 caractères.

**Corrigé.** Les convolutions 3 × 3 avec padding 1 conservent les dimensions spatiales ; les poolings les réduisent.

| Étape | Dimensions |
|---|---|
| Entrée | `[4, 1, 64, 800]` |
| Bloc 1, pooling `(2,2)` | `[4, 32, 32, 400]` |
| Bloc 2, pooling `(2,2)` | `[4, 64, 16, 200]` |
| Bloc 3, pooling `(2,1)` | `[4, 128, 8, 200]` |
| Bloc 4, pooling `(2,1)` | `[4, 256, 4, 200]` |
| Moyenne verticale puis transposition | `[4, 200, 256]` |
| BiLSTM, 128 unités par direction | `[4, 200, 256]` |
| Projection linéaire | `[4, 200, 93]` |
| Log-probabilités pour CTC | `[200, 4, 93]` |

Les 93 classes correspondent aux 92 caractères et au blank d'indice zéro. Les deux couches du BiLSTM ne doublent
pas deux fois la dimension finale : chaque couche produit la concaténation de ses deux directions, soit 256.

## 3. Distinguer extraction visuelle et contexte séquentiel

**Énoncé.** Pourquoi ajouter un BiLSTM après le CNN ? Expliquer aussi pourquoi le modèle ne retourne pas simplement
les pixels de l'image arabe avant de les traiter.

**Corrigé.** Le CNN extrait des motifs visuels : courbes, jonctions, points et combinaisons plus complexes. Son champ
réceptif apporte déjà un contexte local. Le BiLSTM relie les positions de la séquence et utilise un contexte dans
les deux directions pour interpréter une forme ambiguë. Il ne garantit cependant pas une compréhension linguistique
complète et n'est pas un modèle de langue externe.

Le code conserve l'orientation naturelle de l'image. Il inverse l'ordre des **caractéristiques valides** sur l'axe
horizontal avant le BiLSTM, afin que leur parcours soit adapté à des cibles arabes en ordre Unicode logique.
Retourner les pixels donnerait des glyphes en miroir. Cette inversion globale reste une approximation pour une ligne
mêlant arabe, nombres et segments latins : la bidirectionnalité du LSTM ne remplace pas un traitement complet de bidi.

## 4. Appliquer correctement le collapse CTC

**Énoncé.** Réduire les chemins suivants. Les virgules séparent les trames ; `<blank>` est le symbole CTC :

```text
A = [a, a, <blank>, a, b, b, <blank>]
B = [a, a, a]
C = [a, <blank>, a]
```

Pourquoi ne faut-il pas supprimer tous les blanks avant de fusionner les répétitions ? Un espace est-il un blank ?

**Corrigé.** Le collapse fusionne les répétitions **consécutives du chemin**, puis retire les blanks. On obtient
`A → aab`, `B → a` et `C → aa`. Supprimer d'abord le blank de C transformerait le chemin en `[a,a]`, puis en `a` :
on perdrait la répétition réelle. L'implémentation de `greedy_decode()` conserve donc le symbole précédent, blank compris.

L'espace est un caractère ordinaire du vocabulaire lorsqu'il existe dans les transcriptions. Il sépare les mots
dans le résultat final. Le blank exprime l'absence d'émission à une trame et disparaît du texte final.

## 5. Vérifier qu'un alignement CTC est possible

**Énoncé.** Combien de trames minimales faut-il pour les cibles `book`, `لل` et `سلام` ? Pour `book`, une image
prétraitée de largeur 16 est-elle suffisante ?

**Corrigé.** La borne minimale utilisée dans le projet est :

```text
T_min = longueur(cible) + nombre de paires adjacentes identiques
```

`book` contient quatre caractères et une paire `oo` : il faut **5 trames**. `لل` contient deux caractères identiques
adjacents : il faut **3 trames**. `سلام` contient quatre caractères sans répétition adjacente : il faut **4 trames**.
Dans `book`, une trame blank doit séparer les deux émissions de `o`, sinon leur fusion donnerait un seul `o`.

Une largeur 16 ne fournit que `floor(16/4) = 4` trames ; elle est insuffisante pour `book`. Une largeur 20 en fournit
cinq. Respecter cette borne rend un alignement possible ; cela ne garantit pas que l'image soit lisible ou le modèle
assez entraîné. Une perte infinie doit conduire à contrôler les longueurs et les labels avant de masquer l'erreur.

## 6. Calculer CER, WER et exactitude

**Énoncé.** Calculer le CER de `chat` reconnu `chats`, puis le WER de `le chat noir` reconnu `le chat`. Un autre
modèle fait une substitution dans `chat noir` et prédit `chat soir`. Calculer ses trois métriques sur cette ligne.

**Corrigé.** La distance d'édition autorise substitution, insertion et suppression. `chat → chats` demande une
insertion : `CER = 1/4 = 25 %`. `le chat noir → le chat` demande la suppression d'un mot : `WER = 1/3 ≈ 33,33 %`.
Le dénominateur vient toujours de la **référence**, pas de la prédiction.

`chat noir → chat soir` demande une substitution de caractère. La référence contient neuf caractères, espace inclus :
`CER = 1/9 ≈ 11,11 %`. Un des deux mots est différent : `WER = 1/2 = 50 %`. La ligne n'est pas identique, donc
l'exactitude complète vaut **0 %**. `100 - CER` ne constitue pas l'exactitude des lignes.

Une longue séquence d'insertions peut produire un CER supérieur à 100 %. Le code compte des points de code Unicode,
ce qui n'est pas forcément identique au nombre de glyphes ou de caractères visuellement perçus, notamment avec les diacritiques.

## 7. Choisir entre moyenne micro et moyenne macro

**Énoncé.** Une ligne de deux caractères comporte une erreur ; une ligne de cent caractères en comporte cinq.
Comparer la moyenne des CER par ligne et le CER calculé sur le corpus.

**Corrigé.** Les CER individuels sont 50 % et 5 %. La moyenne macro est `(50+5)/2 = 27,5 %` : chaque ligne pèse
autant. Le calcul micro additionne les erreurs et les longueurs de référence : `(1+5)/(2+100) ≈ 5,88 %`.

Le projet utilise la seconde définition pour son CER global. La différence peut être très importante lorsque les
longueurs varient. Une comparaison de modèles doit préciser l'agrégation utilisée. Il est utile de publier le CER
micro et une distribution des erreurs par ligne, sans remplacer l'un par l'autre. Deux CER calculés différemment ne
doivent pas être comparés comme s'ils mesuraient exactement la même chose.

## 8. Construire un lot de largeurs variables

**Énoncé.** Deux lignes prétraitées mesurent 120 et 200 pixels de large, toutes deux à hauteur 64. Décrire le lot,
les longueurs valides, le padding et le rôle de `pack_padded_sequence()`.

**Corrigé.** Le lot a la forme `[2,1,64,200]`. La première ligne reçoit 80 colonnes de padding à droite. Les largeurs
valides restent `[120,200]`, donc les longueurs CNN sont `[30,50]`. Dans `collate_lines()`, `1 - pixel/255` transforme
le blanc en zéro et le noir en un ; le padding zéro correspond ainsi au fond blanc normalisé.

Le réseau inverse seulement les caractéristiques valides, puis compacte les séquences pour que la récurrence ne
traite pas les trames ajoutées comme du contenu. Le décodage s'arrête lui aussi à la longueur valide. Sans cela,
des caractères prédits sur le padding pourraient apparaître dans la sortie. Regrouper des lignes de largeur proche
réduit les calculs inutiles du CNN ; le packing du LSTM ne supprime pas à lui seul ce coût convolutionnel.

## 9. Diagnostiquer une fuite de données

**Énoncé.** Un paragraphe source est découpé en dix segments synthétiques ; huit vont dans train et deux dans
validation. Aucun fichier image n'est identique. La validation mesure-t-elle une généralisation à de nouveaux documents ?

**Corrigé.** Elle mesure au moins partiellement la reconnaissance de nouveaux segments d'une source déjà présente.
Elle n'est pas indépendante par document. Même sans copie binaire d'image, le contenu, le vocabulaire et les thèmes
du paragraphe peuvent être partagés. Il faut séparer les pages ou sources **avant** de répartir leurs segments et
conserver cette provenance dans les manifests. Les hashes d'images et de textes complètent ce contrôle ; ils ne le remplacent pas.

L'audit historique du projet a retrouvé cette situation dans les données Atlas de l'expérience mixte. La correction
du générateur ne change pas rétroactivement ses anciens scores : il faut régénérer puis réentraîner pour évaluer
un corpus regroupé par source. Une réponse d'entretien doit distinguer clairement correctif logiciel et nouvelle preuve expérimentale.

## 10. Concevoir une comparaison d'augmentation valide

**Énoncé.** Sans augmentation, le CER vaut 6,63 % ; avec une augmentation forte, il vaut 9,52 %. Peut-on conclure
que l'augmentation détériore le modèle ? Proposer une expérience contrôlée.

**Corrigé.** Les deux chiffres seuls ne suffisent pas. Dans les essais historiques, le RNG du rendu servait ensuite
au mélange des splits : modifier les transformations changeait aussi les phrases de validation. Le niveau de
difficulté évalué n'était donc pas fixé.

Pour comparer correctement : figer les groupes source et la validation ; utiliser les mêmes textes, vocabulaire,
initialisation, budget d'entraînement et règle de sélection ; augmenter uniquement train ; évaluer chaque modèle sur
les mêmes images de validation. Répéter avec plusieurs graines aide à estimer la variabilité. Une validation réelle
indépendante est nécessaire pour conclure à un gain face au domaine cible. L'augmentation peut compliquer train tout
en améliorant la généralisation ; une baisse temporaire des performances synthétiques ne tranche pas cette question.

## 11. Ordonner une mise à jour AMP et contrôler une reprise

**Énoncé.** Ordonner : forward, backward, unscale, clipping, step, mise à jour du scaler. Pourquoi un `--lr` demandé
à 0,0005 peut-il donner un LR effectif de 0,001 après une reprise ?

**Corrigé.** Après `zero_grad()`, le forward s'effectue sous autocast ; la perte CTC reste en float32. Ensuite :

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
scaler.step(optimizer)
scaler.update()
```

Il faut retirer l'échelle **avant** l'écrêtage, sinon la norme calculée dépend artificiellement du facteur du scaler.
AMP réduit le coût de certaines opérations, mais n'élimine pas le besoin de vérifier les pertes et gradients non finis.
AdamW ajuste les paramètres à partir des gradients et applique une décroissance des poids séparée de ce calcul adaptatif.

Recharger l'état de l'optimiseur peut restaurer son ancien LR après sa création avec l'argument CLI. C'est ce qui a
été observé dans l'expérience mixte. Le LR réellement enregistré doit guider l'interprétation ; une option de reprise
doit préciser si elle conserve l'état historique ou impose volontairement un nouveau taux.

## 12. Passer du prototype à une page réelle

**Énoncé.** La démo reconnaît trois lignes assemblées. Un utilisateur fournit ensuite une photo inclinée avec deux
colonnes et un tableau. Décrire les risques et proposer une évaluation utile.

**Corrigé.** Une projection horizontale peut fusionner des lignes de colonnes différentes, découper les points arabes,
prendre des bordures pour du texte et reconstruire un ordre incorrect. Le reconnaisseur peut aussi échouer sur le flou,
l'éclairage, une police inconnue ou une compression excessive. Ce sont des erreurs distinctes de segmentation et de reconnaissance.

Il faut recueillir un jeu de documents représentatifs, indépendant des sources train, puis mesurer la transcription
complète et inspecter les erreurs par étape. Comparer les crops automatiquement détectés avec des crops fiables permet
d'isoler le reconnaisseur. Prévoir correction d'orientation, détection de mise en page et ordre de lecture pour les
documents complexes. La réussite des tests et d'une page composée vérifie le fonctionnement ; elle ne remplace pas
une mesure sur des documents Darija réels. Le benchmark réservé ne doit pas devenir un jeu d'ajustement au fil des essais.

## Dix questions d'entretien avec réponses courtes

**1. Quel problème votre système résout-il ?** Il transcrit des images de lignes imprimées. Une chaîne expérimentale
ajoute la détection de lignes pour des pages simples à une colonne. L'adaptation Darija utilise actuellement du synthétique.

**2. Pourquoi CNN + BiLSTM + CTC ?** Le CNN extrait le visuel, le BiLSTM contextualise la séquence et CTC entraîne
sur une transcription complète sans position de caractère. Ce choix donne une base compacte et explicable.

**3. Pourquoi ne pas découper chaque lettre arabe ?** Les connexions et variations contextuelles compliquent ce
découpage. CTC apprend un alignement latent entre positions de caractéristiques et caractères, sur des lignes correctement annotées.

**4. CTC réalise-t-il la détection de lignes ?** Non. La détection choisit les régions visuelles ; CTC relie ensuite
les trames d'une région à une transcription. Un paragraphe et son texte global ne deviennent pas automatiquement des couples de lignes.

**5. La darija est-elle un alphabet distinct ?** Elle emploie notamment l'écriture arabe et peut mêler d'autres
écritures. Son vocabulaire, son orthographe variable et ses usages créent un domaine particulier à représenter dans les données.

**6. Que signifie un CER de 1,49 % ?** Sur le test arabe concerné, le total des éditions vaut environ 1,49 % des
caractères de référence. Cela ne signifie ni 98,51 % de lignes parfaites ni la même performance sur la Darija réelle.

**7. Quel est le principal apport du GPU ?** Il accélère les opérations tensorisées du CNN et du LSTM. Le débit
dépend aussi des largeurs, du lot, du prétraitement, de la précision et de la méthode de mesure.

**8. Quelle limite scientifique avez-vous identifiée ?** L'écart synthétique/réel et les défauts de séparation des
sources. L'audit a aussi révélé une comparaison d'augmentation non contrôlée et des diacritiques absents du rendu historique.

**9. Qu'avez-vous corrigé ?** La conservation des diacritiques et de la provenance, la séparation par source et la
validation fixe, ainsi que la déduplication entre splits. Ces corrections préparent de nouvelles expériences ; les anciens checkpoints restent historiques.

**10. Quelle serait la prochaine expérience ?** Régénérer les données corrigées, fine-tuner à configuration tracée,
comparer sur une validation fixe et indépendante, puis réserver le test final à une évaluation du pipeline complet.

Pour relire les constats précis et leurs limites, consulter [l'audit des expériences](EXPERIMENT_AUDIT.md).
