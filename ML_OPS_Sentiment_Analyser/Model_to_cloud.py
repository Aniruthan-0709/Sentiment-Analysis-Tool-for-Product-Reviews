import os
from google.cloud import storage

# ======================= GCP CONFIG =======================
BUCKET_NAME = "mlops_dataset123"  # Your GCS bucket name
LOCAL_MODEL_PATH = "models/naive_bayes_sentiment.pkl"  # Path to local model
GCS_MODEL_PATH = "models/naive_bayes_sentiment.pkl"  # Destination in GCS

# ======================= UPLOAD MODEL TO GCS =======================
def upload_to_gcp(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to GCP Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# Upload the model
upload_to_gcp(BUCKET_NAME, LOCAL_MODEL_PATH, GCS_MODEL_PATH)
