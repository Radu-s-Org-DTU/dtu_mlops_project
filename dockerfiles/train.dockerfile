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
