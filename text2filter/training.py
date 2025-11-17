import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np


class PhraseFilterDataset(Dataset):
    def __init__(self, phrases, filter_labels, intensity_labels, tokenizer, max_len=32):
        self.phrases = phrases
        self.filter_labels = filter_labels
        self.intensity_labels = intensity_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = str(self.phrases[idx])
        encoding = self.tokenizer.encode_plus(
            phrase,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        filter_label = torch.tensor(self.filter_labels[idx], dtype=torch.long)
        intensity_label = torch.tensor(self.intensity_labels[idx], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'filter_label': filter_label,
            'intensity_label': intensity_label
        }

class TwoHeadClassifier(nn.Module):
    def __init__(self, transformer_name='sentence-transformers/paraphrase-mpnet-base-v2',
                 num_filters=12, num_intensities=3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size
        self.filter_head = nn.Linear(hidden_size, num_filters)
        self.intensity_head = nn.Linear(hidden_size, num_intensities)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        filter_logits = self.filter_head(cls_output)
        intensity_logits = self.intensity_head(cls_output)
        return filter_logits, intensity_logits


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(dataset_csv, model_name='sentence-transformers/paraphrase-mpnet-base-v2',
                batch_size=16, lr=2e-5, epochs=5, max_len=32,
                save_path='twohead_model.pt'):

    set_seed(42)

    df = pd.read_csv(dataset_csv)  # columns: phrase, filter_label, intensity_label

    filter_encoder = LabelEncoder()
    intensity_encoder = LabelEncoder()
    df['filter_encoded'] = filter_encoder.fit_transform(df['Filter_Label'])
    df['intensity_encoded'] = intensity_encoder.fit_transform(df['Intensity'])

    os.makedirs('encoders', exist_ok=True)
    with open('encoders/filter_encoder.pkl', 'wb') as f:
        pickle.dump(filter_encoder, f)
    with open('encoders/intensity_encoder.pkl', 'wb') as f:
        pickle.dump(intensity_encoder, f)

    phrases = df['Phrase'].tolist()
    filter_labels = df['filter_encoded'].tolist()
    intensity_labels = df['intensity_encoded'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PhraseFilterDataset(phrases, filter_labels, intensity_labels, tokenizer, max_len=max_len)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TwoHeadClassifier(transformer_name=model_name,
                              num_filters=len(filter_encoder.classes_),
                              num_intensities=len(intensity_encoder.classes_)).to(device)

    for name, param in model.transformer.named_parameters():
        param.requires_grad = False  # freeze all
        if 'encoder.layer.10' in name or 'encoder.layer.11' in name:  # last 2 layers
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            filter_labels_batch = batch['filter_label'].to(device)
            intensity_labels_batch = batch['intensity_label'].to(device)

            filter_logits, intensity_logits = model(input_ids, attention_mask)
            loss_filter = criterion(filter_logits, filter_labels_batch)
            loss_intensity = criterion(intensity_logits, intensity_labels_batch)
            loss = loss_filter + loss_intensity
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Train loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct_filter = 0
        correct_intensity = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                filter_labels_batch = batch['filter_label'].to(device)
                intensity_labels_batch = batch['intensity_label'].to(device)

                filter_logits, intensity_logits = model(input_ids, attention_mask)
                pred_filter = filter_logits.argmax(dim=1)
                pred_intensity = intensity_logits.argmax(dim=1)

                correct_filter += (pred_filter == filter_labels_batch).sum().item()
                correct_intensity += (pred_intensity == intensity_labels_batch).sum().item()
                total += filter_labels_batch.size(0)

        print(f"[Epoch {epoch+1}] Val Filter Acc: {correct_filter/total:.4f}, "
              f"Intensity Acc: {correct_intensity/total:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved at {save_path}")


if __name__ == "__main__":
    train_model("dataset.csv")
