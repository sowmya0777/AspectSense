# ==========================================
# SECTION 1: ENVIRONMENT SETUP
# ==========================================

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report
)

# -------------------------------
# Random seed for reproducibility
# -------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Device
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Training will use CPU.")
# ==========================================
# SECTION 2: LOAD DATASET
# ==========================================

TRAIN_PATH = r"C:\Users\katta\Downloads\01_Aspect_Based_Sentiment_Analysis\01_Aspect_Based_Sentiment_Analysis\dataset\train\train.csv"

df = pd.read_csv(TRAIN_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLabel distribution:")
print(df["label"].value_counts())
# ==========================================
# SECTION 3: PREPARE INPUTS AND LABELS
# ==========================================

# Convert sentiment labels into numbers
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

df["label_id"] = df["label"].map(label_map)

# Combine aspect and review
df["model_input"] = (
    df["aspect_term"].astype(str)
    + " [SEP] "
    + df["text"].astype(str)
)

print("\nLabel mapping:")
print(label_map)

print("\nExample model inputs:")
print(df[["aspect_term", "text", "label", "model_input"]].head(3))

print("\nMissing values:")
print(df[["text", "aspect_term", "label"]].isnull().sum())
# ==========================================
# SECTION 4: TRAIN / VALIDATION SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

print("\nTrain samples:", len(train_df))
print("Validation samples:", len(val_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(val_df["label"].value_counts())
# =========================
# SECTION 5: LOAD TOKENIZER
# =========================

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer loaded successfully!")
print("Model:", MODEL_NAME)
# =========================
# SECTION 6: CREATE DATASET
# =========================

class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        aspect = str(row["aspect_term"])
        text = str(row["text"])
        label = int(row["label_id"])

        encoding = self.tokenizer(
            aspect,
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


train_dataset = ABSADataset(train_df, tokenizer)
val_dataset = ABSADataset(val_df, tokenizer)

print("\nDatasets created successfully!")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
# =========================
# SECTION 7: CREATE MODEL
# =========================

class ABSAModel(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use the [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


model = ABSAModel(
    model_name=MODEL_NAME,
    num_labels=3
)

model = model.to(device)

print("\nModel created successfully!")
print("Number of classes:", 3)
print("Device:", device)
# =========================
# SECTION 8: CREATE DATALOADERS
# =========================

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoaders created successfully!")
print("Batch size:", BATCH_SIZE)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
# =========================
# SECTION 9: TRAINING SETUP
# =========================

EPOCHS = 2
LEARNING_RATE = 2e-5

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print("\nTraining setup completed!")
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Total training steps:", total_steps)
# =========================
# SECTION 10: TRAINING LOOP
# =========================

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")

    for batch_idx, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1} completed!")
    print(f"Average Training Loss: {average_loss:.4f}")
