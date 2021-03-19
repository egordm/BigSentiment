import os
import pandas as pd
from collections import defaultdict

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraModel, AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertPooler
from torchviz import make_dot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_DIR = '../data/sentiment_dataset'
N_OUTPUTS = 18
PAD_TOKEN_ID = 1
MAX_LEN = 256
BATCH_SIZE = 64


class SentimentDataset(Dataset):
    def __init__(self, dataset_dir):
        self.table = pd.read_parquet(dataset_dir)
        self.iterator = self.table.iterrows()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        data = next(self.iterator, None)
        if data is None:
            self.iterator = self.table.iterrows()
            data = next(self.iterator, None)

        input_ids_tmp, labels = data[1][['input_ids', 'labels']]
        input_len = min(len(input_ids_tmp), MAX_LEN)
        # Pad the input ids
        input_ids = torch.full((MAX_LEN,), PAD_TOKEN_ID)
        input_ids[:input_len] = torch.from_numpy(input_ids_tmp[:input_len])
        # Create a attaion mask
        attention_mask = torch.zeros((MAX_LEN,))
        attention_mask[:input_len] = 1
        # Convert labels
        labels = torch.from_numpy(labels.astype(np.float32))
        return dict(input_ids=input_ids, attention_mask=attention_mask, targets=labels)


def create_dataloader(dataset_dir):
    ds = SentimentDataset(dataset_dir)
    return DataLoader(ds, batch_size=BATCH_SIZE)


train_ds = create_dataloader(os.path.join(DATASET_DIR, 'train'))
valid_ds = create_dataloader(os.path.join(DATASET_DIR, 'valid'))

MODEL_DIR = '../data/models/discriminator'


class SentimentPredictor(nn.Module):
    def __init__(self, n_predictions):
        super().__init__()
        self.model = ElectraModel.from_pretrained(MODEL_DIR)
        self.model.tie_weights()
        self.pooler = BertPooler(self.model.config)
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(self.model.config.hidden_size, n_predictions)

    def forward(self, input_ids, attention_mask):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = model_output['last_hidden_state']
        pooled_output = self.pooler(hidden_states)
        dropout_output = self.dropout(pooled_output)
        output = self.output(dropout_output)
        return output


# Initialize the model
model = SentimentPredictor(N_OUTPUTS)
model = model.to(device)

# Initialize optimizers
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_ds) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.MSELoss().to(device)


# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()

    losses = []
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

    return np.mean(losses)


history = defaultdict(list)
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_loss = train_epoch(model, train_ds, loss_fn, optimizer, device, scheduler)

    print(f'Train loss {train_loss}')
    val_loss = eval_model(model, valid_ds, loss_fn, device, )
    print(f'Val   loss {val_loss}')
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
