"""
Script pour observer la répartition des classes dans le corpus
et établir le pondérage pour le calcul de la loss lors
de l'entraînement
"""
import pandas as pd

df = pd.read_parquet("./data/parquet/dataset-small.parquet")

all_labels = [label for sublist in df['output'] for label in sublist if label != -100]

counts = pd.Series(all_labels).value_counts().sort_index()
print("Distribution des classes :\n", counts)

total_tokens = len(all_labels)
nb_classes = 3
weights = total_tokens / (nb_classes * counts)

weights = weights / weights[0]

print("\nPoids calculés (à copier dans train.py) :")
print(list(weights.values))
