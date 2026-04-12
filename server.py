import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from data.dataset import CLASSES, DATASET_PATH
from lib import transforms as _transforms
from lib.model_base import TrashModel
from lib.model_loader import import_model_module

MODELS_DIR = Path("models")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Loaded model instances, keyed by model name
_models: dict[str, TrashModel] = {}


def _get_model(name: str) -> TrashModel:
    if name not in _models:
        path = MODELS_DIR / f"{name}.py"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        mod = import_model_module(path)
        _models[name] = mod.Model()
    return _models[name]


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
def list_models():
    models = sorted(
        p.relative_to(MODELS_DIR).with_suffix("").as_posix()
        for p in MODELS_DIR.rglob("*.py")
        if p.name != "__init__.py"
    )
    return {"models": models}


def _resolve_transform_name(model) -> str | None:
    transform = getattr(type(model), 'transform', None)
    if transform is None:
        return None
    for name, obj in vars(_transforms).items():
        if obj is transform:
            return name
    return None


@app.get("/api/models/{name:path}/params")
def model_params(name: str):
    if ".." in name:
        raise HTTPException(status_code=400, detail="Invalid model name")
    model = _get_model(name)
    cls = type(model)
    params = {
        k: v
        for k, v in vars(cls).items()
        if k.isupper() and not callable(v) and not isinstance(v, (classmethod, staticmethod))
    }
    transform_name = _resolve_transform_name(model)
    if transform_name:
        params["TRANSFORM"] = transform_name
    return {"params": params}


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
    if ".." in req.model:
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
