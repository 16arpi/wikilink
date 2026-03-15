# WikiLink

**WikiLink** est un outil de détection et d'insertion automatique de [liens internes Wikipédia](https://fr.wikipedia.org/wiki/Aide:Liens_internes) (« liens bleus ») dans du texte brut français. Il repose sur un modèle de type NER entraîné à reconnaître les segments de texte devant être liés vers un autre article Wikipédia.

## Usage (webapp)

D'abord, installer les dépendances dans le `requirements.txt`. Ensuite, utiliser fastapi CLI avec :

```bash
$ fastapi run wikilink
```

> Attention : l'inférence s'effectue sur CPU par défaut.

## Procédure méthodologique

Nous avons suivi quatre étapes :

1. Nettoyage du dump de Wikipédia français (02/2026) en enlevant toutes les balises (XML, wikicode) *sauf* les balises hyperliens `[[…]]`.
2. Constitution du dataset d'entraînement : tokenisation du corpus avec le tokenizer de CamemBERTv2 +  préparation des données (chaque token reçoit une étiquette BIO).
3. Entraînement d'un [perceptron multicouche](https://fr.wikipedia.org/wiki/Perceptron_multicouche) par-dessus les embeddings de CamemBERTv2 pour prédire ces étiquettes BIO.
4. Développement d'une interface web fonctionnant avec FastAPI, où l'utilisateur peut donner du texte brut et obtenir du HTML avec les liens Wikipédia insérés.

## Architecture du modèle NER

Le modèle NER se compose de deux blocs :

### 1. CamemBERTv2-base

Nous utilisons [`almanach/camembertv2-base`](https://huggingface.co/almanach/camembertv2-base), un modèle RoBERTa pré-entraîné pour le français (~110 M de paramètres).
Ce choix se justifie notamment parce qu'il a été entraîné sur plus de données francophones, dont une version plus récente de Wikipédia que la première version de camembert. Ses auteurs ([INRIA/ALMAnaCH](https://almanach.inria.fr/)) rapportent de meilleures performances que la première version de Camembert-base.

Ce modèle a été affiné avec le script [train.py](https://github.com/16arpi/wikilink/blob/main/scripts/train.py) qui réentraîne les deux dernières couches de CamemBERTv2-base (11 et 12ème couches). Cela permet d'adapter les représentations de CamemBERT à cette tâche NER spécifique tout en évitant un entraînement complet, qui serait plus coûteux.

### 2. Classifieur : MLP (Perceptron multicouche)

Les embeddings de la dernière couche cachée de CamemBERT sont transmis à un réseau MLP défini avec PyTorch. Le MLP reçoit un vecteur de dimension 768 (taille des embeddings CamemBERTv2-base) pour chaque token de la séquence et produit 3 logits correspondant aux 3 classes BIO.

## Schéma d'annotation BIO

Chaque token de la séquence reçoit l'une des étiquettes suivantes :

| Étiquette | Valeur | Signification |
|-----------|--------|---------------|
| `O`       | `0`    | Token hors lien (*Outside*) |
| `B-Link`  | `1`    | Premier token d'un lien (*Beginning*) |
| `I-Link`  | `2`    | Token intérieur d'un lien (*Inside*) |
| *Spécial* | `-100` | Token spécial (CLS, SEP, padding) — ignoré par la loss |

## Corpus d'entraînement

Le corpus nettoyé (2,24 Go, au format Parquet subdivisé en plusieurs *shards*) est disponible publiquement (à ce lien)[https://www.kaggle.com/datasets/gwendaltsang/wikipedia-first-512-tokens]. Ce corpus résulte de plusieurs étapes de nettoyage successives mais n'est pas encore parfait.

Fichiers présents dans ce repository :

* `data/csv/wikipedia.csv` : réduction de la collecte Wikipedia initiale pour ne garder que 100 000 paragraphes.
* `data/parquet/dataset.parquet` : wikipedia.csv augmenté des tokenisations avec du texte à l'aie de du tokeniseur de `almanach/camembertv2-base`

### Sous-échantillonnage

Pour des contraintes de temps et de hardware (GPU L4), seule une sous-partie du corpus a été utilisée pour l'entraînement (~90 000 segments textuels, dont ~72 000 pour le *train*).

### Source

Le corpus est construit à partir du **dump de Wikipédia français de février 2026**. Toutes les balises (XML, wikicode) ont été retirées **à l'exception des balises hyperliens** `[[…]]`, qui servent de « vérité terrain » pour l'annotation NER.
