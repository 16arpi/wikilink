import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

HG_BERT = "almanach/camembertv2-base"

class BertLinkAnnotator(nn.Module):
    def __init__(self, camembert, nb_labels=3):
        super(BertLinkAnnotator, self).__init__()

        self.bert = camembert

        # --- LA LIGNE CLÉ : On gèle BERT ---
        for param in self.bert.parameters():
            param.requires_grad = False

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

class WikiLink:
    def __init__(self, torch_path, device="cuda"):
        self.bert = AutoModel.from_pretrained(HG_BERT, device_map=device)
        self.model = BertLinkAnnotator(self.bert).to(device)
        self.device = device

        checkpoint = torch.load(torch_path, map_location=torch.device(device))
        _, _ = self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(HG_BERT)

    def generate(self, text):
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_offsets_mapping=True
        )

        ids, mapping, mask = tokens["input_ids"], tokens["offset_mapping"], tokens["attention_mask"]

        t_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        t_mask = torch.tensor([mask], dtype=torch.long).to(self.device)

        logits = self.model(input_ids=t_ids, attention_mask=t_mask)
        preds = logits.argmax(dim=-1)
        preds = preds.tolist()
        preds = preds[0]


        for i in range(1, len(preds) - 1):
            if preds[i-1] > 0 and preds[i+1] > 0 and preds[i] == 0:
                preds[i] = 2

        for i in range(1, len(preds) - 2):
            if preds[i] > 0 and preds[i-1] == 0 and preds[i+1] == 0:
                preds[i] = 0

        result = []
        last_char_idx = 0
        is_in_link = False

        for (char_start, char_end), label in zip(mapping, preds):
            if char_start == char_end:
                continue

            if label > 0:
                result.append(text[last_char_idx:char_start])
                if not is_in_link:
                    result.append("<a href=\"#\" >")
                    is_in_link = True
                last_char_idx = char_start

            elif label == 0:
                result.append(text[last_char_idx:char_start])
                if is_in_link:
                    result.append("</a>")
                    is_in_link = False
                last_char_idx = char_start

        result.append(text[last_char_idx:])

        return ''.join(result)




