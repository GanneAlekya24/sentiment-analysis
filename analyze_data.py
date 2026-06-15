import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification

# --------------------------
# Setup
# --------------------------
os.makedirs("graphs", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Map labels
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Sample small dataset for demo
df = df.sample(2000, random_state=42).reset_index(drop=True)

# Train/Test split
train_texts = df["review"][:1600].tolist()
train_labels = df["label"][:1600].tolist()
test_texts = df["review"][1600:].tolist()
test_labels = df["label"][1600:].tolist()

# --------------------------
# Dataset class
# --------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# --------------------------
# Tokenizer & Datasets
# --------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --------------------------
# Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# --------------------------
# Training
# --------------------------
epochs = 2
train_losses, train_accuracies = [], []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# --------------------------
# Save Graphs
# --------------------------
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs+1), train_accuracies, marker="o", label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()
plt.savefig("graphs/training_accuracy.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs+1), train_losses, marker="o", color="red", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("graphs/training_loss.png")
plt.close()

print("✅ Graphs saved in 'graphs' folder.")
