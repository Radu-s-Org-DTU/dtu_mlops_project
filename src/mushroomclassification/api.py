import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from torchvision import transforms

import wandb
from data import MushroomDataset
from utils.config_loader import load_config

from model import MushroomClassifier

def download_best_model():
    """Download the model with the :best alias from the artifacts collection."""
    load_dotenv(override=True)
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    artifact = api.artifact(f"radugrecu97-dtu-org/wandb-registry-model/model_collection:best", type="model")
    artifact_dir = artifact.download()
    logger.info(f"Model downloaded to: {artifact_dir}")
    return os.path.join(artifact_dir, "mushroom_model.ckpt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classes, device

    model_path = download_best_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MushroomClassifier.load_from_checkpoint(
        model_path,
        learning_rate=load_config()['trainer']['learning_rate']
    ).to(device)
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
        "from WandB."
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
    return transform(image).unsqueeze(0).to(device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
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
