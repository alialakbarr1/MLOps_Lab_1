"""Seed a fresh mlflow registry with the champion Food-11 model.

The Compose stack starts with an empty named volume, so the tracking server in
the ``mlflow`` container has no experiments, no runs and no registered models -
and the ``inference`` service cannot start without ``models:/food11@champion``.

This script re-logs the model trained in lab 2 (whose artifacts are still on
disk under ./mlruns) into whichever tracking server MLFLOW_TRACKING_URI points
at, registers it as ``food11`` and assigns the ``champion`` alias.

It is idempotent: run it again and it adds a new version, then moves the alias.

    # with the stack's mlflow service published on 5000
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python scripts/seed_registry.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
from PIL import Image

# mlflow prints an emoji when a run closes; Windows consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "food11")
ALIAS = os.environ.get("MODEL_ALIAS", "champion")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "food11-serving")

# Local artifacts of the lab 2 champion run (val_accuracy 0.7436).
# LOCAL_MODEL_PATH overrides it, which is how a different version gets promoted
# without retraining - e.g. the lr=0.01 run, to prove a promotion took effect.
DEFAULT_MODEL = ROOT / "mlruns/2/models/m-a30cca875cc1477eac14ab8ec7fdc083/artifacts"
LOCAL_MODEL = Path(os.environ.get("LOCAL_MODEL_PATH", DEFAULT_MODEL))
SOURCE_RUN = os.environ.get("SOURCE_RUN_ID", "888e2d8ef879462ba98253b7c54d9a73")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def input_example() -> np.ndarray:
    """One real 128x128 image, preprocessed exactly as in training.

    mlflow 3 serialises torch models with the traced 'pt2' format, which cannot
    save without a concrete example to trace model.forward with.
    """
    sample_dir = ROOT / "data/food11_processed_mini/validation/Bread"
    candidates = sorted(sample_dir.iterdir()) if sample_dir.is_dir() else []
    if candidates:
        array = np.asarray(
            Image.open(candidates[0]).convert("RGB").resize((128, 128)), dtype=np.float32
        ) / 255.0
        chw = np.transpose(array, (2, 0, 1))
    else:  # data/ is dvc-tracked and may not be checked out
        chw = np.zeros((3, 128, 128), dtype=np.float32)
    return ((chw - MEAN) / STD)[np.newaxis, ...].astype(np.float32)


def main() -> int:
    if not LOCAL_MODEL.is_dir():
        print(f"Local model artifacts not found at {LOCAL_MODEL}", file=sys.stderr)
        print("Run lab 2 training first, or `dvc pull` the mlruns backup.", file=sys.stderr)
        return 1

    print(f"tracking server : {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    print(f"loading weights : {LOCAL_MODEL}")
    model = mlflow.pytorch.load_model(LOCAL_MODEL.as_uri())

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="seeded-champion"):
        mlflow.set_tag("source_run_id", SOURCE_RUN)
        mlflow.set_tag("note", "lab 2 champion weights, seeded into a fresh registry")
        info = mlflow.pytorch.log_model(model, "model", input_example=input_example())

    logged = client.get_logged_model(info.model_id)
    print(f"artifact uri    : {logged.artifact_location}")

    version = mlflow.register_model(f"models:/{info.model_id}", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, "source_run_id", SOURCE_RUN)

    print(f"registered      : {MODEL_NAME} v{version.version}")
    print(f"alias           : {ALIAS} -> v{version.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
