from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
from torchvision import transforms
from .model import MushroomClassifier
from .utils.config_loader import load_config
from src.mushroomclassification.data import MushroomDataset

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, classes
    model_path = f"models/{load_config()['model']['file_name']}.ckpt"
    model = MushroomClassifier.load_from_checkpoint(model_path)
    model.eval()
    
    dataset = MushroomDataset(data_path='')
    classes = dataset.classes

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
        
        return {class_name: prob for class_name, prob in zip(classes, probabilities)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
