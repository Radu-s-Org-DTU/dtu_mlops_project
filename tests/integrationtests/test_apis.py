from fastapi.testclient import TestClient

from src.mushroomclassification.api import app

client = TestClient(app)

def test_predict_with_real_image():
    with TestClient(app) as client:
        with open(r"data\raw_subset\Classes\edible\Agaricus_bisporus\Agaricus_bisporus29.png", "rb") as img_file:
            response = client.post("/predict/", files={"file": ("image.jpg", img_file, "image/jpeg")})
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

def test_predict_with_invalid_file_type():
    with TestClient(app) as client:
        response = client.post("/predict/", files={"file": ("invalid.txt", b"This is not an image.")})
        assert response.status_code == 500