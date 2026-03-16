"""
Définition PyTorch de notre modèle
"""
import torch.nn as nn

from transformers import CamembertModel

# Classe qui définie notre réseau de neurones
# __init__() : définition des nœuds de notre réseau
# forward() : méthode de construction de nos couches

# NB : forward() retourne des logits car, lors de
# l'entraînement, le calcul de la loss se fait depuis
# les logits
class BertLinkAnnotator(nn.Module):
    def __init__(self, camembert, nb_labels=3, map_device="cuda"):
        super(BertLinkAnnotator, self).__init__()

        # On charge notre modèle Camembert
        self.bert = camembert

        # On ne veut pas ré-entraîner BERT, ce serait trop coûteux...
        for param in self.bert.parameters():
            param.requires_grad = False

        # ...mais on s'autorise le fine-tuning des 2 dernières couches
        # de BERT car ça améliore les performances
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
                # Optionnel: print(f"Dégelé : {name}") # Pour vérifier au lancement

        # Les embeddings de BERT sont envoyés à notre MLP
        # qui se compose d'une première couche dense de la taille embeddings x sequence length
        # puis une couche d'activation ReLU, puis un DropOut
        # puis une couche cachée de 256 noeuds
        # puis une activation, puis notre projection linéaire de la taille de nos classes (3)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, nb_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        sequence_outputs = self.dropout(last_hidden_state)
        logits = self.mlp(sequence_outputs)

        return logits
