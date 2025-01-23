from google.cloud import storage


def upload_to_gcs(bucket_name, source_path, destination_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(source_path)
    print(f"Model uploaded to gs://{bucket_name}/{destination_path}")
