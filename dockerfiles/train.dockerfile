FROM python:3.11-slim

# Install system dependencies (including Google Cloud SDK)
RUN apt update && \
    apt install --no-install-recommends -y \
        build-essential \
        gcc \
        curl \
        gnupg \
        google-cloud-sdk && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose

COPY requirements_dev.txt requirements_dev.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt --verbose

# Copy application files
COPY data data/
COPY tasks.py tasks.py
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY model_config.yaml model_config.yaml

# Ensure invoke is installed
RUN pip install invoke --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Ensure gsutil and invoke work
RUN which gsutil && which invoke

# Use bash as entrypoint to allow debugging
ENTRYPOINT ["/bin/bash", "-c", "invoke train"]
