#!/bin/bash

BUCKET_NAME="gs://02476-data"
RAW_PATH="data/raw"
RAW_SUBSET_PATH="data/raw_subset"

echo "Pulling raw data from GCS..."
gsutil -m cp -r ${BUCKET_NAME}/${RAW_PATH} /app/${RAW_PATH}

echo "Pulling raw subset data from GCS..."
gsutil -m cp -r ${BUCKET_NAME}/${RAW_SUBSET_PATH} /app/${RAW_SUBSET_PATH}

echo "Data pulled successfully."
