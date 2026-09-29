"""FastAPI service around the registered Food-11 model.

The model is pulled from the mlflow Model Registry by alias, not from a file on
disk, so promoting a new version is a registry operation rather than a redeploy:

    models:/food11@champion

Run locally:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000

The tracking URI comes from MLFLOW_TRACKING_URI so the same image works against a
host server (``http://host.docker.internal:5000``) or a real one in a cluster.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# torchvision.datasets.ImageFolder assigns indices by sorting the class folder
# names, so this list must stay in sorted order to match what the model learned.
# For Food-11 the sorted order happens to equal the original numeric labels.
CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

MODEL_URI = os.environ.get("MODEL_URI", "models:/food11@champion")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Must match the preprocessing in data.py + train.py exactly.
IMAGE_SIZE = (128, 128)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup, instead of per request."""
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    state["model_uri"] = MODEL_URI
    yield
    state.clear()


app = FastAPI(title="Food-11 classifier", lifespan=lifespan)


def preprocess(raw: bytes) -> np.ndarray:
    """Bytes of an uploaded image -> normalised NCHW batch of one."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"not a readable image: {exc}") from exc

    chw = np.transpose(array, (2, 0, 1))
    return ((chw - MEAN) / STD)[np.newaxis, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict[str, object]:
    """Which model this container actually resolved, useful after a promotion."""
    return {"model_uri": state.get("model_uri"), "tracking_uri": TRACKING_URI}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    batch = preprocess(await file.read())
    logits = np.asarray(model.predict(batch), dtype=np.float32).reshape(-1)
    probabilities = softmax(logits)
    best = int(probabilities.argmax())

    ranked = sorted(
        ({"category": c, "confidence": float(p)} for c, p in zip(CATEGORIES, probabilities)),
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return {
        "filename": file.filename,
        "category": CATEGORIES[best],
        "confidence": float(probabilities[best]),
        "top3": ranked[:3],
    }
