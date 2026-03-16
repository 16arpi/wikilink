"""
Script d'évaluation du modèle
"""
import pandas as pd
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from transformers import AutoModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import BertLinkAnnotator
from torch.utils.data import Dataset

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
INPUT = "./data/dataset.parquet"
BERT_MODEL = "almanach/camembertv2-base"
CHECKPOINT_PATH = "./wikilink/torch/mlp_epoch_7_550.pth"
DEVICE = "mps"

BATCH_SIZE = 128

torch.manual_seed(SEED)

bert = AutoModel.from_pretrained(BERT_MODEL, device_map=DEVICE)

print("Loading dataset...")
dataset = pd.read_parquet(INPUT)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
#train, test = [], test[:200]

dataset_test = WikipediaDataset(test)
dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE)

size = len(test)
num_batches = size // BATCH_SIZE + 1

# Initialisation du modèle
model = BertLinkAnnotator(bert).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

print(f"Clés ignorées (normalement BERT) : {len(missing_keys)}")
print(f"Clés inattendues (devrait être vide) : {unexpected_keys}")

model.eval()
model.bert.eval()
test_loss, correct, total_tokens = 0, 0, 0

with torch.no_grad():
    for batch in tqdm.tqdm(dl_test, total=(len(test) // BATCH_SIZE)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        output_ids = batch["output_ids"].to(DEVICE)

        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        test_loss += loss_fn(pred.view(-1, pred.shape[-1]), output_ids.view(-1))

        predictions = pred.argmax(dim=-1)
        mask = (output_ids != -100)

        correct += (predictions[mask] == output_ids[mask]).sum().item()
        total_tokens += mask.sum().item()

correct = correct / total_tokens
test_loss = test_loss / num_batches

print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
