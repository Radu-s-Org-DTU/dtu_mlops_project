#FROM python:3.11-slim

# Install system dependencies (including Google Cloud SDK)
#RUN apt update && apt install --no-install-recommends -y \
#        build-essential \
#        gcc \
#        curl \
#        gnupg && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    apt update && apt install --no-install-recommends -y google-cloud-sdk && \
#    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
#WORKDIR /app

# Copy dependency files first for caching
#COPY requirements.txt requirements.txt


# Install dependencies **without using --mount**
#RUN pip install -r requirements.txt --verbose


# Copy the rest of the application files
#COPY data data/
#COPY src src/
#COPY README.md README.md
#COPY pyproject.toml pyproject.toml
#COPY model_config.yaml model_config.yaml
#COPY tasks.py tasks.py

# Ensure invoke is installed
#RUN pip install invoke --no-cache-dir --verbose
#RUN pip install . --no-deps --no-cache-dir --verbose

# Use bash as entrypoint to allow debugging
#ENTRYPOINT ["/bin/bash", "-c", "invoke train"]




FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl ca-certificates && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install google-cloud-storage

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir --verbose

COPY data data/
COPY tasks.py tasks.py
COPY model_config.yaml model_config.yaml
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY model_config.yaml model_config.yaml
COPY tasks.py tasks.py

RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["sh", "-c", "python src/mushroomclassification/pull_data.py && invoke train"]
