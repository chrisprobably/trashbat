import importlib
import random
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from dataset import CLASSES, DATASET_PATH

MODELS_DIR = Path("models")
sys.path.insert(0, str(MODELS_DIR))

from model_base import TrashModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Loaded model instances, keyed by model name
_models: dict[str, TrashModel] = {}


def _get_model(name: str) -> TrashModel:
    if name not in _models:
        if not (MODELS_DIR / f"{name}.py").exists():
            raise HTTPException(status_code=404, detail="Model not found")
        mod = importlib.import_module(name)
        _models[name] = mod.Model()
    return _models[name]


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
def list_models():
    models = sorted(p.stem for p in MODELS_DIR.glob("model*.py"))
    return {"models": models}


@app.get("/api/random-images")
def random_images():
    images = {}
    for cls in CLASSES:
        folder = DATASET_PATH / cls
        files = list(folder.glob("*.jpg"))
        if files:
            chosen = random.choice(files)
            images[cls] = f"{cls}/{chosen.name}"
    return {"images": images}


@app.get("/dataset/{cls}/{filename}")
def serve_image(cls: str, filename: str):
    path = DATASET_PATH / cls / filename
    if not path.exists() or path.suffix.lower() != ".jpg":
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


class PredictRequest(BaseModel):
    model: str
    image: str  # e.g. "glass/glass42.jpg"


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.model.startswith("model"):
        raise HTTPException(status_code=400, detail="Invalid model name")

    parts = req.image.split("/")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid image path")
    cls, filename = parts
    img_path = DATASET_PATH / cls / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    model = _get_model(req.model)
    try:
        return model.predict(Image.open(img_path))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
