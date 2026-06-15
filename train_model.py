import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Make sure the model folder exists
if not os.path.exists("model"):
    os.makedirs("model")

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Rename columns if needed
df = df.rename(columns={"review": "text", "sentiment": "label"})

# Features and labels
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# Test accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"Model trained! Accuracy: {accuracy*100:.2f}%")
