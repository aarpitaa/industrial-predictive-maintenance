"""
Inference module for the turbofan RUL (Remaining Useful Life) model.

Usage:
    from predict import predict_rul
    rul = predict_rul(recent_cycles)

`recent_cycles` is a list of dicts, oldest cycle first, most recent cycle last.
Each dict must have keys 'sensor_1' through 'sensor_21'. Provide up to 5 cycles
(the training window) for the most accurate rolling-feature computation; fewer
is accepted (fewer than 5 cycles of engine history) and handled the same way
training handled an engine's first few cycles.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.pyfunc

_MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
_MODEL_PATH = _MODELS_DIR / 'champion_model'

_bundle = None
_model = None


def _load():
    global _bundle, _model
    if _bundle is None:
        _bundle = joblib.load(_MODELS_DIR / 'serving_bundle.joblib')
        _model = mlflow.pyfunc.load_model(str(_MODEL_PATH))
    return _bundle, _model


def _build_feature_vector(recent_cycles, bundle):
    """
    Reconstruct the exact feature set the model was trained on, from raw
    sensor history for one engine.
    """
    df = pd.DataFrame(recent_cycles)

    sensor_cols_to_use = bundle['sensor_cols_to_use']
    window = bundle['window']
    feature_cols = bundle['feature_cols']

    # Rolling mean/std, computed the SAME way as training: min_periods=1 so
    # a short history (fewer than `window` cycles) still produces a value
    # instead of NaN.
    for sensor in sensor_cols_to_use:
        df[f'{sensor}_rollmean'] = df[sensor].rolling(window=window, min_periods=1).mean()
        df[f'{sensor}_rollstd'] = df[sensor].rolling(window=window, min_periods=1).std()

    # We only care about the most recent cycle's engineered values —
    # that's the "current state" we're predicting RUL for.
    latest = df.iloc[[-1]].copy()

    # A single cycle of history gives rollstd = NaN (no spread to measure
    # from one point) — same fillna(0) convention used in training.
    latest[feature_cols] = latest[feature_cols].fillna(0)

    # CRITICAL: select columns in the exact trained order. Do not rely on
    # dict/column order matching by coincidence.
    return latest[feature_cols]


def predict_rul(recent_cycles):
    """
    Predict Remaining Useful Life (in cycles) for an engine, given its most
    recent raw sensor readings.

    Parameters
    ----------
    recent_cycles : list[dict]
        Raw sensor readings, oldest first, most recent last. Each dict needs
        keys 'sensor_1' .. 'sensor_21'. Up to 5 cycles used; fewer accepted.

    Returns
    -------
    float : predicted remaining useful life, in cycles.
    """
    if not recent_cycles:
        raise ValueError("predict_rul requires at least one cycle of sensor readings")

    bundle, model = _load()

    features = _build_feature_vector(recent_cycles, bundle)
    features_scaled = features.copy()
    features_scaled[bundle['feature_cols']] = bundle['scaler'].transform(features[bundle['feature_cols']])

    prediction = model.predict(features_scaled)
    return float(prediction[0])
