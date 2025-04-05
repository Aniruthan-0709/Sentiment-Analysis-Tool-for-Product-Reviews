import pandas as pd
import logging
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bias_detection.log"), logging.StreamHandler()]
)
logging.info("Starting Bias Detection...")

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "naive_bayes_sentiment.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# Load and prepare data
data_path = os.path.join("Data", "Data.csv")
df = pd.read_csv(data_path)

def map_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["label"] = df["star_rating"].apply(map_sentiment)
df = df.dropna(subset=["review_body"])
df["review_body"] = df["review_body"].astype(str)
label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["label"] = df["label"].map(label_mapping)

# Balance the dataset
positive = df[df["label"] == 2]
negative = df[df["label"] == 0]
neutral = df[df["label"] == 1]
negative_upsampled = negative.sample(len(positive), replace=True, random_state=42)
neutral_upsampled = neutral.sample(len(positive), replace=True, random_state=42)
balanced_df = pd.concat([positive, negative_upsampled, neutral_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split test set
_, X_test, _, y_test = train_test_split(
    balanced_df["review_body"], balanced_df["label"], test_size=0.2, random_state=42
)
X_test_tfidf = vectorizer.transform(X_test)

# Create a bias slice: for demonstration, slice by review length (short vs. long)
review_lengths = X_test.apply(lambda x: len(x.split()))
median_length = review_lengths.median()
short_idx = review_lengths <= median_length
long_idx = review_lengths > median_length

y_pred = model.predict(X_test_tfidf)

accuracy_short = accuracy_score(y_test[short_idx], y_pred[short_idx])
accuracy_long = accuracy_score(y_test[long_idx], y_pred[long_idx])

logging.info(f"Accuracy for short reviews (<= median length): {accuracy_short:.4f}")
logging.info(f"Accuracy for long reviews (> median length): {accuracy_long:.4f}")

# Check for significant disparity (example threshold: 0.1 difference)
if abs(accuracy_short - accuracy_long) > 0.1:
    logging.warning("Significant bias detected between review length groups.")
else:
    logging.info("No significant bias detected between review length groups.")
