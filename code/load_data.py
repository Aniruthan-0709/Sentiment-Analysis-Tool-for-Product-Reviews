import os
import json
import subprocess
import sys
from google.cloud import storage

def setup_kaggle_credentials():
    """
    Writes Kaggle credentials from environment variables into ~/.kaggle/kaggle.json.
    Expects KAGGLE_USERNAME and KAGGLE_KEY to be set.
    """
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_config_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            print("Error: Kaggle credentials not provided in environment variables.")
            sys.exit(1)
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print("Kaggle credentials file created at", kaggle_json_path)
    else:
        print("Kaggle credentials already exist.")

def download_and_extract_dataset():
    """
    Uses the Kaggle CLI to download and unzip the dataset.
    Downloads into a folder named 'dataset' in the current directory.
    """
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "cynthiarempel/amazon-us-customer-reviews-dataset",
        "--unzip",
        "-p",
        dataset_dir
    ]
    print("Downloading dataset. This may take a while due to its large size...")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print("Error during dataset download:", e)
        sys.exit(1)
    print("Dataset downloaded and extracted into folder:", dataset_dir)
    return dataset_dir

def upload_folder_to_gcs(source_folder, bucket_name, destination_prefix=""):
    """
    Recursively uploads files from source_folder to the specified GCS bucket.
    Maintains folder structure under destination_prefix.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, _, files in os.walk(source_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_folder)
            blob_path = os.path.join(destination_prefix, relative_path)
            print(f"Uploading {local_path} to gs://{bucket_name}/{blob_path} ...")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
    print("All files have been uploaded successfully.")

def main():
    # Step 1: Set up Kaggle credentials
    setup_kaggle_credentials()
    
    # Step 2: Download and extract the 50GB dataset from Kaggle
    dataset_dir = download_and_extract_dataset()
    
    # Step 3: Upload dataset files to GCS
    bucket_name = os.environ.get("GCS_BUCKET", "mlops_data_staging")
    upload_folder_to_gcs(dataset_dir, bucket_name, destination_prefix="amazon_reviews")

if __name__ == "__main__":
    main()
