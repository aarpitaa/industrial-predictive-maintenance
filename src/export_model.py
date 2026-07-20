"""
Exports the current @champion model from the MLflow registry into a
self-contained local folder, for use by the serving container.

Run this after registering/promoting a new champion in MLflow, before
rebuilding the Docker image.
"""
import shutil
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_PATH = PROJECT_ROOT / 'models' / 'champion_model'

mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))

if EXPORT_PATH.exists():
    shutil.rmtree(EXPORT_PATH)

mlflow.artifacts.download_artifacts(
    artifact_uri="models:/turbofan-xgb-regressor@champion",
    dst_path=str(EXPORT_PATH)
)
print(f"Exported champion model to {EXPORT_PATH}")
