# Industrial Predictive Maintenance

Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset, comparing a classical ML baseline (XGBoost) against a deep learning approach (LSTM).

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

docker/                    — containerization setup (in progress)

models/                    — trained models, fitted scaler, serving bundle (not tracked in git)

notebooks/
├── 01_eda.ipynb           — data exploration, RUL derivation, feature engineering, train/val split
├── 02_modeling.ipynb      — XGBoost regressor + classifier, feature importance, model packaging
├── 03_ltsm.ipynb          — LSTM model, aligned model comparison, hypothesis testing
└── 04_test_predict.ipynb  — smoke test for the packaged serving pipeline

src/
├── data_processing.py     — shared data loading utilities
├── features.py            — feature engineering functions
├── predict.py             — inference module: raw sensor readings in, RUL prediction out
├── api/                   — API layer (in progress)
└── models/                — model class definitions (in progress)

.gitignore
README.md
requirements.txt
```

## Approach

1. **EDA**: identified and dropped 7 flat/uninformative sensors (constant under FD001's single operating condition); computed RUL by deriving it from each engine's known failure point.
2. **Feature engineering**: rolling mean/std (5-cycle window) per sensor, computed per-engine to avoid cross-engine leakage.
3. **Train/validation split**: split by engine unit, not by row, to prevent data leakage between training and validation.
4. **XGBoost baseline**: trained both regression (RUL) and classification (fail-within-30-cycles) models; tuned classification threshold to favor recall over precision, reflecting the asymmetric cost of missed failures vs. false alarms.
5. **LSTM**: sliding-window sequence model (30-cycle windows) trained in PyTorch with early stopping.
6. **Model comparison**: aligned both models' predictions on identical validation samples; compared using paired t-test, Wilcoxon signed-rank test, and bootstrap confidence intervals.
7. **Serving**: packaged the winning model (XGBoost) with its fitted scaler and exact feature-column order into a single bundle, exposed through one `predict_rul()` function in `src/predict.py`.

## Serving: two inference paths

- **Real-time (`src/api/main.py`)**: FastAPI service, one prediction per request, model loaded once at startup. Suited to answering "is this specific engine about to fail?" on demand.
- **Batch (`src/batch_scoring.py`)**: PySpark pipeline, scores an entire fleet's current RUL in one run. Feature engineering (rolling sensor statistics) is computed via Spark `Window` functions rather than pandas, demonstrating the same logic implemented for distributed execution. Suited to a scheduled job scoring hundreds/thousands of engines at once, rather than one-off requests.

Note: PySpark was included primarily to demonstrate familiarity with distributed data tooling; the current dataset (a few hundred engines) doesn't itself require distributed processing to run quickly on a single machine.

## Status

✅ **Complete**: EDA, feature engineering, XGBoost baseline, LSTM, statistical model comparison, serving pipeline (`predict.py`)

🚧 **In progress / planned**:
1. FastAPI service — `/predict` and `/health` endpoints around `predict.py`
2. Docker — containerize the FastAPI service
3. Cloud deployment — push to a container registry, deploy to a managed serverless container service (scaled to zero / torn down after demoing, to control cost)
4. MLflow — experiment tracking for model versions and metrics
5. PySpark — included primarily to demonstrate familiarity with distributed data tooling, not because this dataset requires it at its current scale (~20K rows)

💡 **Stretch goal**: a small RAG layer — a synthetic maintenance-manual knowledge base queryable via an `/ask` endpoint, answering questions like "what does a rising vibration sensor reading usually indicate?"
