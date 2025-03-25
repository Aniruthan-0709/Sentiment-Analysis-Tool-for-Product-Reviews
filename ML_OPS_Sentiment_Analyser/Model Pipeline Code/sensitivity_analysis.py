import pandas as pd
import logging
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sensitivity_analysis.log"), logging.StreamHandler()]
)
logging.info("Starting Sensitivity Analysis using SHAP...")

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

# For sensitivity analysis, we only need a small sample
sample_df = df.sample(n=50, random_state=42)
texts = sample_df["review_body"].tolist()
X_sample = vectorizer.transform(texts)

# Using SHAP KernelExplainer for the MultinomialNB model
# Note: KernelExplainer can be slow; using a small sample for demonstration.
explainer = shap.KernelExplainer(model.predict_proba, X_sample[:10].toarray())
shap_values = explainer.shap_values(X_sample[:5].toarray())

# Plot SHAP summary for the first class (e.g., Negative)
plt.figure()
shap.summary_plot(shap_values[0], X_sample[:5].toarray(), feature_names=vectorizer.get_feature_names_out(), show=False)
os.makedirs("artifacts", exist_ok=True)
plot_path = os.path.join("artifacts", "shap_summary.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
logging.info(f"SHAP summary plot saved at {plot_path}")
