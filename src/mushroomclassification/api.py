import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv  # Add this import
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from loguru import logger
from PIL import Image
from torchvision import transforms

import wandb
from src.mushroomclassification.data import MushroomDataset
from src.mushroomclassification.utils.config_loader import load_config

from .model import MushroomClassifier


def download_model(bucket_name, source_blob_name, destination_file_name):
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def download_best_model():
    """Download the model with the :best alias from the artifacts collection."""
    load_dotenv(override=True)
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    artifact = api.artifact(f"{os.getenv('WANDB_MODEL_NAME')}:best", type="model")
    artifact_dir = artifact.download()
    logger.info(f"Model downloaded to: {artifact_dir}")

if __name__ == "__main__":
    download_best_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classes

    download_model(
        "02476-api-model",
        "models/api-model/mushroom_model.ckpt",
        "models/api-model/mushroom_model.ckpt"
    )

    model = MushroomClassifier.load_from_checkpoint(
        "models/api-model/mushroom_model.ckpt",
        learning_rate=load_config()['trainer']['learning_rate']
    )
    model.eval()
    dataset = MushroomDataset(data_path='')
    classes = dataset.classes
    yield

app = FastAPI(
    title="Mushroom Classification API",
    description=(
        "An API for predicting the class of a mushroom "
        "('conditionally_edible', 'deadly', 'edible', 'poisonous') "
        "based on an uploaded image. The model is dynamically loaded "
        "from Google Cloud Storage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(file.content_type)
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a JPEG or PNG image."
            )

        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        return dict(zip(classes, probabilities))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
