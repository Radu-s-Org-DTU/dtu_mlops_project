import os
from locust import HttpUser, task

class SimpleUser(HttpUser):

    @task
    def test_predict(self):
        """Test the `/predict/` endpoint with an existing image."""
        file_path = os.path.join(
            "data", 
            "raw_subset", 
            "Classes", 
            "edible", 
            "Agaricus_bisporus", 
            "Agaricus_bisporus29.png"
        )
        try:
            with open(file_path, "rb") as img_file:
                response = self.client.post(
                    "/predict/",
                    files={"file": ("image.jpg", img_file, "image/jpeg")},
                )
        except Exception as e:
            print(e)
            raise e
