import os

from google.cloud import storage


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
    BUCKET_NAME = "02476-data"
    RAW_PATH = "data/raw"
    RAW_SUBSET_PATH = "data/raw_subset"

    #download_from_gcs(BUCKET_NAME, RAW_PATH, "/app/data/raw")
    download_from_gcs(BUCKET_NAME, RAW_SUBSET_PATH, "/app/data/raw_subset")

    print("Data pulled successfully.")
