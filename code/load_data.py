import os
import json
import subprocess
import sys
import pandas as pd
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
    print("Downloading dataset. This may take a while...")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print("Error during dataset download:", e)
        sys.exit(1)
    print("Dataset downloaded and extracted into folder:", dataset_dir)
    return dataset_dir

def convert_csv_to_parquet(dataset_dir):
    """
    Converts CSV files in the dataset directory to Parquet format.
    """
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                parquet_path = csv_path.replace('.csv', '.parquet')
                try:
                    df = pd.read_csv(csv_path)
                    df.to_parquet(parquet_path, index=False)
                    print(f"Converted {file} to Parquet format.")
                except Exception as e:
                    print(f"Error converting {file}: {e}")

def upload_folder_to_gcs(source_folder, bucket_name, destination_prefix=""):
    """
    Uses gsutil for parallelized upload of Parquet files to GCS.
    """
    command = [
        "gsutil", "-m", "cp", "-r", source_folder,
        f"gs://{bucket_name}/{destination_prefix}"
    ]
    print(f"Uploading {source_folder} to gs://{bucket_name}/{destination_prefix} ...")
    try:
        subprocess.check_call(command)
        print("Upload completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during GCS upload:", e)
        sys.exit(1)

def main():
    # Step 1: Set up Kaggle credentials
    setup_kaggle_credentials()
    
    # Step 2: Download and extract the 50GB dataset from Kaggle
    dataset_dir = download_and_extract_dataset()
    
    # Step 3: Convert CSVs to Parquet for optimized storage
    convert_csv_to_parquet(dataset_dir)
    
    # Step 4: Upload Parquet files to GCS
    bucket_name = os.environ.get("GCS_BUCKET", "mlops_staging")
    upload_folder_to_gcs(dataset_dir, bucket_name, destination_prefix="amazon_reviews")

if __name__ == "__main__":
    main()
