# WikiLink

**WikiLink** est un outil de détection et d'insertion automatique de [liens internes Wikipédia](https://fr.wikipedia.org/wiki/Aide:Liens_internes) (Hyperliens bleus) dans du texte brut français. Il repose sur un modèle de type NER entraîné à reconnaître les segments de texte devant être liés vers un autre article Wikipédia.

## Utilisation

```bash
git clone https://github.com/16arpi/wikilink.git
cd wikilink
pip install -r requirements.txt
```

```bash
fastapi run wikilink
```
L'interface est ensuite accessible depuis [http://localhost:8000](http://localhost:8000).

> Attention : l'inférence s'effectue sur CPU par défaut.

## Sections :

1. [Procédure méthodologique](#procédure-méthodologique)
2. [Architecture du modèle NER](#architecture-du-modèle-ner)
3. [Corpus d'entraînement](#corpus-dentraînement)
4. [Remarques méthodologiques et pistes futures](remarques-méthodologiques-et-pistes-futures)



## Procédure méthodologique

Nous avons suivi quatre étapes :

1. Nettoyage du dump de Wikipédia français (02/2026) en enlevant toutes les balises (XML, wikicode) *sauf* les balises hyperliens `[[…]]`.
2. Constitution du dataset d'entraînement : tokenisation du corpus avec le tokenizer de CamemBERTv2 +  préparation des données (chaque token reçoit une étiquette BIO).
3. Entraînement d'un [perceptron multicouche](https://fr.wikipedia.org/wiki/Perceptron_multicouche) par-dessus les embeddings de CamemBERTv2 pour prédire ces étiquettes BIO.
4. Développement d'une interface web fonctionnant avec FastAPI, où l'utilisateur peut donner du texte brut et obtenir du HTML avec les liens Wikipédia insérés.

## Architecture du modèle NER

Le modèle NER se compose d'un bloc transformeur-encoder (CamemBERTv2-base) suivi d'un bloc MLP (Perceptron multicouche) :

![schema](schema_ner_model.svg)

Nous utilisons [`camembertv2-base`](https://huggingface.co/almanach/camembertv2-base), un modèle RoBERTa pré-entraîné pour le français (~110 M de paramètres).
Ce choix se justifie notamment parce qu'il a été entraîné sur plus de données francophones, dont une version plus récente de Wikipédia que la première version de camemBERT. Ses auteurs ([INRIA/ALMAnaCH](https://almanach.inria.fr/)) rapportent de meilleures performances que la première version de CamemBERT.

Ce modèle a été affiné avec le script [`train.py`](scripts/train.py) qui réentraîne les deux dernières couches de CamemBERTv2-base (11 et 12ème couches). Cela permet d'adapter les représentations de CamemBERT à cette tâche NER spécifique tout en évitant un entraînement complet, qui serait plus coûteux.

Les embeddings de la dernière couche cachée de CamemBERT sont transmis à un réseau MLP défini avec PyTorch. Le MLP reçoit un vecteur de dimension 768 (taille des embeddings CamemBERTv2-base) pour chaque token de la séquence et produit 3 logits correspondant aux 3 classes BIO.


## Corpus d'entraînement

Le corpus est construit à partir du dump de Wikipédia français de février 2026. Toutes les balises (XML, wikicode) ont été retirées à l'exception des balises hyperliens `[[…]]`.

La dernière version du corpus nettoyé (2,24 GB) est disponible [à ce lien](https://www.kaggle.com/datasets/gwendaltsang/wikipedia-fr-fevrier2026-presqueclean), il contient 1,7 millions de lignes de petits wikipédia découpés. Bien que ce corpus résulte de plusieurs étapes de nettoyage successives, il n'est pas parfait.

### Sous-échantillonnage

Pour des contraintes de temps et de hardware, seule une sous-partie du corpus a été utilisée pour l'entraînement (100 000 segments textuels, dont ~72 000 pour le *train*). Cette sous-partie est disponible dans le fichier [`wikipedia.csv`](data/wikipedia.csv).


### Préparation du jeu de données

Le script [`datasets.py`](scripts/datasets.py) convertit le corpus nettoyé en un jeu de données prêt à l'entraînement. Ce script produit [`dataset.parquet`](data/dataset.parquet) contenant les colonnes suivantes :

| Colonne           | Type                | Description |
|-------------------|---------------------|-------------|
| `text`            | `str`               | Texte original avec les balises `[[…]]` |
| `input`           | `list[int]`         | Les `input_ids` : séquence de tokens encodés par le tokenizer |
| `attention_mask`  | `list[int]`         | Masque d'attention : `1` pour les vrais tokens, `0` pour le padding — permet au modèle d'ignorer les positions de remplissage |
| `offset_mapping`  | `list[tuple[int]]`  | Correspondance entre chaque token et sa position `(début, fin)` dans le texte source — sert à retrouver le texte original à partir des tokens |
| `output`          | `list[int]`         | Étiquettes BIO : `0` (Token hors lien (*Outside*)), `1` (Premier token d'un lien (*Beginning*)), `2` (Token intérieur d'un lien (*Inside*)), ou `-100` (token spécial (CLS, SEP, padding), ignoré par la loss) |

## Remarques méthodologiques et pistes futures

### Note sur le ratio de liens

Les [guidelines de Wikipédia](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Linking) recommandent de ne lier que la première occurrence d'un terme dans un article :

> « As a rule of thumb, link only the first occurrence of a term in both the lead and body of the article. »

Cela pourrait entraîner *in fine* une légère sous-annotation par le modèle NER. Nous espérons que la pondération des classes mise en oeuvre dans [`weights.py`](scripts/weights.py) atténue cet éventuel biais.

Il serait possible de stratifier les articles du corpus en fonction de leur densité de liens hypertextes, afin de réduire cet éventuel biais lié à la pratique wikipédienne de ne lier que la première occurrence d'un terme.

### Pistes futures

- L'INRIA propose également [`almanach/camembertav2-base`](https://huggingface.co/almanach/camembertav2-base), une variante basée sur l'architecture DeBERTaV2 (au lieu de RoBERTa). Des tests préliminaires suggèrent qu'un MLP entraîné sur ce modèle pourrait offrir de très bonnes performances. Toutefois, DeBERTaV2 est plus coûteux en calcul ce qui augmente le temps d'entraînement et les besoins en mémoire GPU.
- Tester le même modèle mais avec une couche linéaire par dessus afin de comparer les performances MLP _vs_ modèle linéaire.
