import os

from google.cloud import storage
from utils.config_loader import load_config


def download_from_gcs(bucket_name, source_path, destination_path):
    os.makedirs(destination_path, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_path)

    for blob in blobs:
        dest_file_path = os.path.join(destination_path, blob.name.replace(source_path, "").lstrip("/"))
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        blob.download_to_filename(dest_file_path)

if __name__ == "__main__":
    path = load_config()['data']['data_path']
    available_paths = ["data/raw", "data/raw_subset"]

    if path not in available_paths:
        raise ValueError(f"Invalid data path: {path}. Available paths: {available_paths}")

    BUCKET_NAME = "02476-data"

    download_from_gcs(BUCKET_NAME, path, '/app/' + path)

    print("Data pulled successfully.")
