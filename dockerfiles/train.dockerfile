FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
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
COPY model_config.yml model_config.yml
COPY tasks.py tasks.py

RUN pip install . --no-deps --no-cache-dir --verbose

RUN mkdir -p /app
COPY pull_data.sh /app/pull_data.sh
RUN chmod +x /app/pull_data.sh

ENTRYPOINT ["/bin/bash", "-c", "/app/pull_data.sh && invoke train"]
