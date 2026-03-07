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

class WikipediaDataset(Dataset):
    def __init__(self, dataframe):
        # Convertir en listes pour un accès O(1)
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

BATCH_SIZE = 64

bert = AutoModel.from_pretrained(BERT_MODEL, device_map="cuda")

print("Loading dataset...")
dataset = pd.read_parquet(INPUT)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
train, dev = train_test_split(train, test_size=0.1, random_state=42)

dataset_train = WikipediaDataset(train)
dataset_dev = WikipediaDataset(dev)
dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
dl_dev = DataLoader(dataset_dev, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = BertLinkAnnotator(bert, map_device="cuda").to("cuda")
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

def run_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    model.bert.eval()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output_ids = batch["output_ids"].to("cuda")

        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(pred.view(-1, pred.shape[-1]), output_ids.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 500 == 0:
            loss, current = loss.item(), (len(batch) + 1) * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Cf. https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def run_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

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
            mask = (output_ids != -100)

            correct += (predictions[mask] == output_ids[mask]).sum().item()
            total_tokens += mask.sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Training...")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    run_train(dl_train, model, loss_fn, optimizer)
    run_test(dl_dev, model, loss_fn)

print("Done!")

