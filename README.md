# WikiLink

**WikiLink** est un outil de détection et d'insertion automatique de [liens internes Wikipédia](https://fr.wikipedia.org/wiki/Aide:Liens_internes) (« liens bleus ») dans du texte brut français. Il repose sur un modèle de type NER entraîné à reconnaître les segments de texte devant être liés vers un autre article Wikipédia.

## Principe général

Nous avons suivi quatre étapes :

1. **Corpus** : nettoyage du dump de Wikipédia français (février 2026) en enlevant toutes les balises (XML, wikicode) *sauf* les balises hyperliens `[[…]]`.
2. **Formatage** : tokenisation d'une sous-partie avec le tokenizer de CamemBERTv2 +  préparation des données (chaque token reçoit une étiquette BIO).
3. **Entraînement** : entraînement d'un classifieur [perceptron multicouche](https://fr.wikipedia.org/wiki/Perceptron_multicouche) par-dessus les embeddings de CamemBERTv2 pour prédire ces étiquettes BIO.
4. **Inférence** : interface web fonctionnant avec FastAPI, où l'utilisateur peut donner facilement du texte brut et obtenir du HTML avec les liens Wikipédia insérés.

## Architecture du modèle

Le modèle se compose de deux blocs :

### CamemBERTv2-base

Nous utilisons [`almanach/camembertv2-base`](https://huggingface.co/almanach/camembertv2-base), un modèle RoBERTa pré-entraîné pour le français (~110 M de paramètres).
Il a été entraîné sur une version plus récente de Wikipédia et d'autres corpus francophones ; ses auteurs ([INRIA/ALMAnaCH](https://almanach.inria.fr/)) rapportent de meilleures performances que la première version de CamemBERT-base.

**Stratégie de fine-tuning :**

La totalité des paramètres de CamemBERT est gelée (`requires_grad = False`) à l'exception des deux dernières couches de l'encodeur Transformer (couches 10 et 11) et du *pooler*. Ce compromis permet d'adapter les représentations internes de CamemBERT à la tâche NER spécifique tout en évitant un entraînement complet, qui serait plus coûteux.




Fichiers présents :
* `data/csv/wikipedia.csv` : réduction de la collecte Wikipedia initiale pour ne garder que 100 000 paragraphes.
* `data/parquet/dataset.parquet` : wikipedia.csv augmenté des tokenisations avec du texte à l'aie de du tokeniseur de `almanach/camembertv2-base`

## Usage (webapp)

D'abord, installer les dépendances dans le `requirements.txt`. Ensuite, utiliser fastapi CLI avec :

```bash
$ fastapi run wikilink
```

> Attention : l'inférence s'effectue sur CPU par défaut.
