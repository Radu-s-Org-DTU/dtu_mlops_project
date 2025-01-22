# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc unzip curl && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose

COPY requirements_dev.txt requirements_dev.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt --verbose

COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install . --no-deps --no-cache-dir --verbose


EXPOSE 8080

ENTRYPOINT ["uvicorn", "src.mushroomclassification.api:app", "--host", "0.0.0.0", "--port", "8080"]
