# Wikipedia linker

Fichiers présents :
* `data/csv/wikipedia.csv` : réduction de la collecte Wikipedia initiale pour ne garder que 100 000 paragraphes.
* `data/parquet/dataset.parquet` : wikipedia.csv augmenté des tokenisations avec du texte à l'aie de du tokeniseur de `almanach/camembertv2-base`

## Usage (webapp)

D'abord, installer les dépendances dans le `requirements.txt`. Ensuite, utiliser fastapi CLI avec :

```bash
$ fastapi run wikilink
```

> Attention : l'inférence s'effectue sur CPU par défaut.
