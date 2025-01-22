FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install dependencies first so docker can cache each 'requirements' file layer separately
# This way, if the individual requirements file doesn't change, the docker cache will be used
# Instead of reinstalling everything
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir --verbose

COPY requirements_dev.txt requirements_dev.txt
RUN pip install -r requirements_dev.txt --no-cache-dir --verbose

COPY data data/
COPY tasks.py tasks.py
COPY model_config.yaml model_config.yaml
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY model_config.yml model_config.yml
COPY tasks.py tasks.py

RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["invoke", "train"]
