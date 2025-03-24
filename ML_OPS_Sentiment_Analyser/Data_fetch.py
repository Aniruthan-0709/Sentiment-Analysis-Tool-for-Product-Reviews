import os
from google.cloud import storage

# ======================= GCP CONFIG =======================
BUCKET_NAME = "mlops_dataset123"  # Replace with your GCS bucket name
FILE_NAME = "data/raw/Sampled_Chunk.csv"  # Replace with your file name in GCS
LOCAL_DIR = "Data/"  # Local folder to save the file
LOCAL_FILE_NAME="Data.csv"
LOCAL_FILE_PATH = os.path.join(LOCAL_DIR, LOCAL_FILE_NAME)

# Ensure the Data folder exists
os.makedirs(LOCAL_DIR, exist_ok=True)

# ======================= DOWNLOAD DATA FROM GCP =======================
def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    """Download a file from GCP Storage to the local machine."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from GCP bucket {bucket_name} to {destination_file_name}")

# Download dataset
download_from_gcp(BUCKET_NAME, FILE_NAME, LOCAL_FILE_PATH)
