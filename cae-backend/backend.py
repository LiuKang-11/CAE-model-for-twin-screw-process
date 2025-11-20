import shutil
import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import torch
import numpy as np
from PIL import Image
import io
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER CONSTANTS ---
AVAILABLE_MODELS = ["Default Model (v1.0)"]

PCA_CLASSES = [
  'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',
  'mL1c-50-15', 'mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',
  'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',
  'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',
  'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',
  'sL3-50-33', 'sR1-50-33', 'sR2-50-33', 'sR3-50-33'
]

# UPDATED: Colors mapped to 3 main groups (Red, Grey, Blue)
PCA_COLORS = [
  '#ef4444', '#ef4444', '#ef4444', '#ef4444', '#ef4444', '#ef4444', # Kneading (Red)
  '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', # Mixing (Grey)
  '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8',
  '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8',
  '#94a3b8', '#94a3b8', '#94a3b8', '#3b82f6', '#3b82f6', # Screw (Blue)
  '#3b82f6', '#3b82f6', '#3b82f6', '#3b82f6'
]

# --- HELPER FUNCTIONS ---

def mock_model_inference(elements: List[dict], model_name: str):
    """
    Simulates inference. Returns Temp and RTD curve.
    """
    num_elements = len(elements)
    if num_elements == 0:
        raise ValueError("Configuration cannot be empty.")
        
    print(f"Using Model: {model_name}")
    
    base_temp = 240.0 if "Default" in model_name else 255.0 

    rtd_points = []
    for i in range(20):
        val = np.exp(-(i - 10)**2 / 10)
        rtd_points.append(float(val))

    return {
        "temperature": round(random.uniform(base_temp, base_temp + 20), 1),
        "rtdCurve": rtd_points
    }

def resize_custom_element(image_bytes, target_size=(64, 64)):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize(target_size)
    return image

# --- API ENDPOINTS ---

@app.get("/api/models")
async def get_models():
    return AVAILABLE_MODELS

@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...), 
    model_name: str = Form(...),
    base_model: str = Form(...)
):
    """
    Simulates training and returns PCA, MSE, and R2 for the training set.
    """
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        df = pd.read_csv(file_location)
        print(f"Training on {len(df)} rows with new name: {model_name}...")
        
        if model_name not in AVAILABLE_MODELS:
            AVAILABLE_MODELS.append(model_name)
        
        # Generate Mock PCA Data for the Training Response
        mock_pca_data = []
        for i in range(60):
            idx = random.randint(0, len(PCA_CLASSES) - 1)
            mock_pca_data.append({
                "x": np.random.randn() * 10, 
                "y": np.random.randn() * 10, 
                "z": np.random.randn() * 10, 
                "name": PCA_CLASSES[idx],
                "color": PCA_COLORS[idx]
            })

        return {
            "status": "success", 
            "message": f"Model '{model_name}' trained successfully.",
            "model_name": model_name,
            "mse": round(random.uniform(0.001, 0.03), 4),
            "r2": round(random.uniform(0.90, 0.99), 4),
            "pca_data": mock_pca_data
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/upload-element")
async def upload_custom_element(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = resize_custom_element(contents)
    save_path = f"assets/custom_{file.filename}"
    return {"status": "success", "path": save_path}

@app.post("/api/predict")
async def run_simulation(config: dict):
    elements = config.get("elements", [])
    model_name = config.get("model_name", "Default Model (v1.0)")
    
    try:
        results = mock_model_inference(elements, model_name)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Twin Screw CAE Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)