import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image
from torchvision import transforms

from src.mushroomclassification.data import MushroomDataset

from .model import MushroomClassifier


def download_model(bucket_name, source_blob_name, destination_file_name):
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classes
    
    download_model(
        "02476-api-model",
        "models/api-model/mushroom_model.ckpt",
        "models/api-model/mushroom_model.ckpt"
    )
    
    model = MushroomClassifier.load_from_checkpoint("models/api-model/mushroom_model.ckpt")
    model.eval()
    dataset = MushroomDataset(data_path='')
    classes = dataset.classes
    yield
    
app = FastAPI(lifespan=lifespan)

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
        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        return dict(zip(classes, probabilities))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
