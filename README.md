# trashbat

A from-scratch PyTorch image classifier trained on the [TrashNet](https://github.com/garythung/trashnet) dataset. Classifies waste images into categories (cardboard, glass, metal, paper, plastic, trash).

Includes a FastAPI web UI for uploading images and comparing predictions across models.

## Quickstart

**Install dependencies**

```bash
uv sync
```

**Unzip the dataset**

```bash
unzip data/dataset-resized.zip -d data/
```

This produces a `data/dataset-resized/` folder containing one subfolder per class (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).

**Train all models**

```bash
uv run python train.py --all
```

**Start the web UI**

```bash
uv run uvicorn server:app --reload
```

Then open http://localhost:8000.

## Commands

```bash
# Train a specific model
uv run python train.py model1

# Train multiple models
uv run python train.py model1 model2

# Train all models
uv run python train.py --all

# Run the server (models must be trained first)
uv run uvicorn server:app --reload

# Add a dependency
uv add <package>
```

## Models

| Model | Loss | Notes |
|-------|------|-------|
| model1 | Cross-entropy | Logistic regression baseline |
| model2 | MSE on one-hot targets | For comparison with model1 |
