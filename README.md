# Industrial Predictive Maintenance

Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset, comparing a classical ML baseline (XGBoost) against a deep learning approach (LSTM), with a full MLflow-tracked, containerized serving pipeline.

## Overview

This project frames engine failure prediction two ways:
- **Regression**: predict exact Remaining Useful Life (RUL), in cycles
- **Classification**: predict whether an engine will fail within the next 30 cycles

Two model architectures were built and rigorously compared on the same held-out validation set:

|        **Model**       |    MAE (cycles)  |
|------------------------|------------------|
| **XGBoost** (deployed) |       22.18      |
|          LSTM          |       48.89      |

XGBoost significantly outperformed the LSTM (paired t-test and Wilcoxon signed-rank test, both p < 0.0001; see `notebooks/03_ltsm.ipynb` for the full statistical comparison and write-up). XGBoost was selected for deployment — both for accuracy and for its simpler, more robust serving footprint compared to a PyTorch-based model.

## Dataset

This project uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001 subset: single operating condition, single fault mode — HPC degradation). Download it from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the extracted files in `data/CMAPSSData/`.

## Setup

```
bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
data/
├── CMAPSSData/            — raw NASA dataset (not tracked in git — see Dataset section above)
└── processed/             — engineered train/val splits, XGBoost predictions cache (not tracked in git)

models/                    — trained models, fitted scaler, serving bundle (not tracked in git)

notebooks/
├── 01_eda.ipynb           — data exploration, RUL derivation, feature engineering, train/val split
├── 02_modeling.ipynb      — XGBoost regressor + classifier, feature importance, model packaging
├── 03_ltsm.ipynb          — LSTM model, aligned model comparison, hypothesis testing
└── 04_test_predict.ipynb  — smoke test for the packaged serving pipeline (direct call, API, Docker)

src/
├── data_processing.py     — shared data loading utilities
├── features.py            — feature engineering functions
├── predict.py             — inference module: raw sensor readings in, RUL prediction out
├── export_model.py        — exports the current MLflow @champion model to a self-contained local folder for serving
├── api/                   — API layer (in progress)
└── main.py                — FastAPI app: /predict and /health endpoints
└── schemas.py             — Pydantic request/response models

Dockerfile
.dockerignore
.gitignore
README.md
requirements.txt              — full project dependencies (training, notebooks, etc.)
requirements-api.txt          — minimal dependencies for the deployed serving container
```

## Approach

1. **EDA**: identified and dropped 7 flat/uninformative sensors (constant under FD001's single operating condition); computed RUL by deriving it from each engine's known failure point.

2. **Feature engineering**: rolling mean/std (5-cycle window) per sensor, computed per-engine to avoid cross-engine leakage.

3. **Train/validation split**: split by engine unit, not by row, to prevent data leakage between training and validation.

4. **XGBoost baseline**: trained both regression (RUL) and classification (fail-within-30-cycles) models; tuned classification threshold to favor recall over precision, reflecting the asymmetric cost of missed failures vs. false alarms.

5. **LSTM**: sliding-window sequence model (30-cycle windows) trained in PyTorch with early stopping.

6. **Model comparison**: aligned both models' predictions on identical validation samples; compared using paired t-test, Wilcoxon signed-rank test, and bootstrap confidence intervals.

7. **Serving**: packaged the winning model (XGBoost) with its fitted scaler and exact feature-column order into a single bundle, exposed through one `predict_rul()` function in `src/predict.py`.

8. **API**: wrapped `predict_rul()` in a FastAPI service with Pydantic-validated `/predict` and `/health` endpoints, model loaded once at startup.

9. **Experiment tracking**: all training runs (XGBoost regressor, XGBoost classifier, LSTM, and the model comparison itself) logged to MLflow, with the winning model registered and aliased `@champion`.

10. **Batch scoring**: a PySpark pipeline scores an entire fleet's current RUL in one pass, reimplementing the rolling-feature logic with Spark `Window` functions.

11. **Containerization**: the FastAPI service is Dockerized, with the champion model exported to a self-contained local folder at build time (decoupled from the live MLflow tracking store — see note below).

## Experiment tracking: MLflow

All training runs are logged with MLflow — parameters, metrics, artifacts, and (for the model comparison) hypothesis-test results (p-values, effect size, confidence intervals) as queryable run metrics rather than only prose in a notebook. The winning XGBoost regressor is registered in the MLflow Model Registry under an alias, `@champion`, so the "current best model" is a stable reference rather than a hardcoded file path.

This replaces what was, earlier in the project, manual copy-pasting of metrics into markdown cells and screenshots — MLflow gives a versioned, comparable history of every run, so re-training a model or comparing a new candidate against the current champion doesn't require re-reading old notebooks by hand.

Run `mlflow ui` from the project root to view the tracking dashboard locally.

## Serving: two inference paths

- **Real-time (`src/api/main.py`)**: FastAPI service, one prediction per request, model loaded once at startup. Suited to answering "is this specific engine about to fail?" on demand.

- **Batch (`src/batch_scoring.py`)**: PySpark pipeline, scores an entire fleet's current RUL in one run. Feature engineering (rolling sensor statistics) is computed via Spark `Window` functions rather than pandas, demonstrating the same logic implemented for distributed execution. Suited to a scheduled job scoring hundreds/thousands of engines at once, rather than one-off requests.

Note: PySpark was included primarily to demonstrate familiarity with distributed data tooling; the current dataset (a few hundred engines) doesn't itself require distributed processing to run quickly on a single machine.

## Running the API locally with Docker

```
bash
docker build -t turbofan-rul-api .
docker run -p 8000:8000 turbofan-rul-api
```

Then test:
```
bash
curl http://localhost:8000/health
```

Note: requires `models/champion_model/` to exist first — generate it by running `python src/export_model.py` after training and registering a model via MLflow. The container serves from this self-contained exported folder rather than connecting to the live MLflow tracking store, since the store's artifact paths are local to the training machine and wouldn't resolve inside a container.

## Status

✅ **Complete**: EDA, feature engineering, XGBoost baseline, LSTM, statistical model comparison, serving pipeline (`predict.py`), FastAPI service, MLflow experiment tracking + model registry, PySpark batch scoring pipeline, Docker containerization

🚧 **Next**:
1. Cloud deployment — push the image to a container registry (ECR/GCR/ACR), deploy to a managed serverless container service (Cloud Run / App Runner / Container Apps), and confirm the live endpoint matches local test results. Can be scaled to zero or torn down after demoing to control cost.

💡 **Stretch goal**: a small RAG layer — a synthetic maintenance-manual knowledge base queryable via an `/ask` endpoint, answering questions like "what does a rising vibration sensor reading usually indicate?"
