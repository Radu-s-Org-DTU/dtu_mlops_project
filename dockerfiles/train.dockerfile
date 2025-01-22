FROM python:3.11-slim

# Install system dependencies (including Google Cloud SDK)
RUN apt update && apt install --no-install-recommends -y \
        build-essential \
        gcc \
        curl \
        gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt update && apt install --no-install-recommends -y google-cloud-sdk && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt --verbose

# Copy the rest of the application files
COPY data data/
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs/model_config.yaml configs/model_config.yaml
COPY tasks.py tasks.py

# Ensure invoke is installed
RUN pip install invoke --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Use bash as entrypoint to allow debugging
ENTRYPOINT ["/bin/bash", "-c", "invoke train"]
