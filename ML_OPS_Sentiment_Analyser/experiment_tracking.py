import mlflow
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_tracking.log"), logging.StreamHandler()]
)
logging.info("Starting Experiment Tracking with MLflow...")

# Set the tracking URI (assumes MLflow server is running as in the YAML file)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment_Model_Experiments")

with mlflow.start_run():
    # Log dummy hyperparameters (these can be replaced with real ones)
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    
    # Log dummy metrics (replace with actual metrics)
    mlflow.log_metric("accuracy", 0.85)
    
    # Log model version (assume versioning file is updated)
    version_file = os.path.join("models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)
    
    logging.info("Experiment tracked with MLflow.")
