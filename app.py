from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import re

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "sentiment_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 2

# ----------------------------
# Preprocess text
# ----------------------------
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# ----------------------------
# Dataset class
# ----------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(label)
        return item

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ----------------------------
# Train model if not exists
# ----------------------------
if os.path.exists(MODEL_PATH):
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS)
    model = DistilBertForSequenceClassification(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("Training model, please wait...")
    df = pd.read_csv("dataset/IMDB Dataset.csv")

    # LIMIT DATASET TO 2000 samples for faster training
    df = df.sample(n=2000, random_state=42)

    df['review'] = df['review'].apply(preprocess)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(df['review'].tolist(), df['sentiment'].tolist(), test_size=0.1, random_state=42)

    train_dataset = IMDBDataset(X_train, y_train, tokenizer)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # you can increase for better accuracy
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=50,
        save_strategy='no',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete!")

model.to(DEVICE)
model.eval()

# ----------------------------
# Prediction function
# ----------------------------
def predict_sentiment(text):
    text = preprocess(text)
    encodings = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        max_prob, pred_class = torch.max(probs, dim=1)
        max_prob = float(max_prob)

    if max_prob < 0.6:
        sentiment = "Neutral"
    elif pred_class.item() == 0:
        sentiment = "Negative"
    else:
        sentiment = "Positive"

    confidence = max_prob * 100
    return sentiment, confidence

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review", "")
        if review.strip() == "":
            return jsonify({"error": "Please enter a review!"})

        sentiment, confidence = predict_sentiment(review)
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "review_length": len(review.strip().split())
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Something went wrong!"})

# ----------- New route for file upload (AJAX) -----------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    text = file.read().decode('utf-8', errors='ignore')
    sentiment, confidence = predict_sentiment(text)
    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "review_length": len(text.strip().split())
    })

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
