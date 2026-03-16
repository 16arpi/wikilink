"""
Script d'entraînement du modèle
"""
import pandas as pd
import torch
import tqdm
import torch.nn as nn

from transformers import CamembertModel, AutoModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import BertLinkAnnotator
from torch.utils.data import Dataset

FROM_CHECKPOINT = ""

class WikipediaDataset(Dataset):
    def __init__(self, dataframe):

        self.inputs = dataframe["input"].tolist()
        self.masks = dataframe["attention_mask"].tolist()
        self.outputs = dataframe["output"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.masks[idx], dtype=torch.long),
            "output_ids": torch.tensor(self.outputs[idx], dtype=torch.long)
        }

SEED = 42
INPUT = "./data/parquet/dataset-small.parquet"
BERT_MODEL = "almanach/camembertv2-base"

BATCH_SIZE = 128

torch.manual_seed(SEED)

bert = AutoModel.from_pretrained(BERT_MODEL, device_map="cuda")

print("Loading dataset...")
dataset = pd.read_parquet(INPUT)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
train, dev = train_test_split(train, test_size=0.1, random_state=42)

# On charge nos deux dataset
dataset_train = WikipediaDataset(train)
dataset_dev = WikipediaDataset(dev)

# On les place dans un dataloader pour faciliter
# l'accès aux batchs côté GPU
dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dl_dev = DataLoader(dataset_dev, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# On instancie notre modèle
model = BertLinkAnnotator(bert, map_device="cuda").to("cuda")

# On instancie le pondérage, la loss et l'optimiseur
calculated_weights = torch.tensor([1.0, 22.0, 4.0]).to("cuda")
loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=calculated_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Si jamais on a des checkpoint, on part de là...
if FROM_CHECKPOINT:
    checkpoint = torch.load(FROM_CHECKPOINT, map_location=torch.device("cuda"))
    _, _ = model.load_state_dict(checkpoint, strict=False)

# Fonction d'entraînement
def run_train(epoch, dataloader, model, loss_fn, optimizer):
    size = len(train)
    model.train()

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=(len(train) // BATCH_SIZE)):

        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output_ids = batch["output_ids"].to("cuda")

        # Notre prediction est dans un batch
        # on applatie donc notre tenseur
        # pour récupérer les logits de la dernière couche
        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(pred.view(-1, pred.shape[-1]), output_ids.view(-1))

        # on fait la rétropropagation
        loss.backward()
        optimizer.step()
        # on remet les gradients à zero...
        optimizer.zero_grad()

        # toutes les 50 étapes
        # on enregistre un checkpoint
        if i % 50 == 0:
            loss, current = loss.item(), (i + 1) * BATCH_SIZE
            print(f"batch {i}. loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Checkpoint des acquis
            to_save = {
                k: v for k, v in model.state_dict().items()
                if model.get_parameter(k).requires_grad
            }
            torch.save(to_save, f"checkpoints/mlp_epoch_{epoch}_{i}.pth")

# Fonction de test (après chaque epoch)
def run_test(dataloader, model, loss_fn):
    size = len(dev)
    num_batches = size // BATCH_SIZE + 1

    model.eval()
    test_loss, correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            output_ids = batch["output_ids"].to("cuda")

            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(pred.view(-1, pred.shape[-1]), output_ids.view(-1))

            predictions = pred.argmax(dim=-1)
            # On utilise un mask pour ne pas prendre
            # en compte les tokens spéciaux dans le calcul
            # de l'accuracy
            mask = (output_ids != -100)

            correct += (predictions[mask] == output_ids[mask]).sum().item()
            total_tokens += mask.sum().item()
    correct = (correct / total_tokens) if total_tokens > 0 else 0
    test_loss = test_loss / num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Training...")
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    run_train(t, dl_train, model, loss_fn, optimizer)
    run_test(dl_dev, model, loss_fn)

print("Done!")

