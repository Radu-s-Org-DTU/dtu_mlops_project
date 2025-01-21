import os

import reflex as rx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/") 
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:8000/") 

config = rx.Config(
    app_name="frontend", 
    backend_port=8080, 
    frontend_port=8080, 
    api_url = BACKEND_URL,
    deploy_url = FRONTEND_URL,
    cors_allowed_origins=["*"],    
)