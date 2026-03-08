
### Création d'un fichier parquet avec 50 000 lignes :


```py
!python formattage_data.py \
    -i ./final_corpus \
    -o . \
    -n 50000 \
    --per_file 50000 \
    --batch_size 512 \
    --log_every 10000 \
    --seed 42
```

### Split du jeu de donnée en train / test / val :

```py
!python prepare_data.py --input ./dataset_50000.parquet --output-dir ./prepared_data --seed 42
```

### Lançement d'une optimisation des hyperparamètres :

```py
!python optimize_hyperparams.py \
    --dataset ./prepared_data \
    --n-trials 10 \
    --seed 42 \
    --trials-dir ./optuna_results \
    --subsample-frac 0.15
```
