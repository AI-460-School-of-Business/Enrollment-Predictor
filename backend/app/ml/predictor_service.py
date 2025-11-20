"""
Predictor service utilities

Provides functions to load a saved model, transform raw input rows using the
project's FeatureEngineer, and produce predictions. Exposes a simple
`flask_predict` wrapper suitable for calling from a Flask view.

This file is intentionally lightweight and does not start any servers. The
Flask app should import `flask_predict` and call it with the parsed JSON body.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from data.feature_engineer import FeatureEngineer
from utils.db_config import DB_CONFIG
import psycopg2
import os
from glob import glob


def load_saved_model(model_path: str) -> Tuple[Any, Optional[List[str]]]:
    """Load a pickled model from disk.

    Returns a tuple (model, feature_columns) where feature_columns may be None
    if the saved object did not include them.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with p.open("rb") as fh:
        saved = pickle.load(fh)

    if isinstance(saved, dict):
        # Common saved layout: {"model": estimator, "feature_columns": [...]}
        model = saved.get("model") or saved.get("estimator") or next(iter(saved.values()))
        feature_columns = saved.get("feature_columns") or saved.get("feature_cols")
    else:
        model = saved
        feature_columns = getattr(model, "feature_columns", None)

    return model, feature_columns


def prepare_features(raw_rows: Iterable[Dict], features: str = "min") -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Convert an iterable of raw dict rows to model-ready feature matrix X.

    - raw_rows: iterable of dict-like objects (JSON-decoded request body rows)
    - features: feature schema name used by FeatureEngineer ("min" or "rich")

    Returns (X_df, feature_columns). X_df is a pandas DataFrame suitable for
    passing to scikit-learn estimators. feature_columns is the ordered list of
    column names used (may be None if unknown).
    """
    raw_df = pd.DataFrame(list(raw_rows))

    # If the user passed an empty list, return an empty DataFrame
    if raw_df.shape[0] == 0:
        return pd.DataFrame(), None

    fe = FeatureEngineer(features)

    # FeatureEngineer.prepare_features typically returns (X, y). We'll call it
    # and accept either (X, y) or just X depending on implementation.
    prepared = fe.prepare_features(raw_df)
    if isinstance(prepared, tuple) and len(prepared) >= 1:
        X = prepared[0]
    else:
        X = prepared

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    return X, getattr(fe, "feature_columns", None)


def predict_with_model(model_path: str, raw_rows: Iterable[Dict], features: str = "min") -> Dict[str, Any]:
    """Run predictions for the provided raw_rows using the saved model.

    Returns a dict with keys:
      - predictions: list of numeric predictions
      - feature_columns: list of columns used (if available)
      - count: number of rows predicted
    """
    model, saved_feature_cols = load_saved_model(model_path)

    X, fe_feature_cols = prepare_features(raw_rows, features)

    # If feature columns are known from the saved model and X lacks them, try to
    # reorder or select the columns expected by the model.
    expected_cols = saved_feature_cols or fe_feature_cols
    if expected_cols:
        # If X doesn't contain all expected cols, raise a helpful error
        missing = [c for c in expected_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        X = X.loc[:, expected_cols]

    # Some models expect a numpy array; scikit-learn accepts DataFrame too.
    preds = model.predict(X)

    # Convert predictions to plain Python types
    preds_list = [float(p) for p in preds]

    return {
        "predictions": preds_list,
        "feature_columns": expected_cols,
        "count": len(preds_list),
    }


def flask_predict(payload: Dict) -> Dict[str, Any]:
    """Thin wrapper to call from a Flask view.

    Expected payload shape (JSON):
      {
        "model_path": "/path/to/model.pkl",     # optional, default uses a local path
        "features": "min",                     # optional, "min" or "rich"
        "rows": [ {..}, {..} ]                   # required, list of raw input dicts
      }

    Returns a JSON-serializable dict suitable for use as a Flask response.
    """
    model_path = payload.get("model_path") or "/app/data/prediction_models/enrollment_model.pkl"
    features = payload.get("features", "min")
    rows = payload.get("rows")

    if rows is None:
        raise ValueError("Payload must include a 'rows' list of input dicts")

    return predict_with_model(model_path, rows, features)


def run_sql_query(sql: str) -> pd.DataFrame:
    """Execute a SQL query against the project's database and return a DataFrame."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(sql, conn)
        return df
    finally:
        conn.close()


def _choose_model_path(payload: Dict) -> str:
    """Choose a model file path based on payload options.

    Priority:
      1. payload['model_path'] if provided (absolute or relative)
      2. payload['model_filename'] joined to model_dir
      3. payload['model_prefix'] - newest matching file in model_dir
      4. newest .pkl in model_dir
    """
    model_path = payload.get("model_path")
    model_dir = Path(payload.get("model_dir") or os.getenv("MODEL_DIR") or "/app/data/prediction_models")

    if model_path:
        return str(Path(model_path))

    if payload.get("model_filename"):
        return str(model_dir / payload["model_filename"])

    prefix = payload.get("model_prefix")
    if prefix:
        matches = sorted(model_dir.glob(f"{prefix}*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return str(matches[0])

    # Fallback: newest .pkl
    all_models = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if all_models:
        return str(all_models[0])

    raise FileNotFoundError(f"No model files found in {model_dir}")


def sql_predict(payload: Dict) -> Dict[str, Any]:
    """Run a SQL query from the payload, convert results to raw rows, choose a model, and predict.

    Expected payload keys (JSON):
      - sql: required, SQL string to select course rows
      - model_path | model_filename | model_prefix: optional to select model
      - features: optional, 'min' (default) or 'rich'

    Returns the same shape as `predict_with_model`.
    """
    sql = payload.get("sql")
    if not sql:
        raise ValueError("Payload must include 'sql' with the SELECT statement to retrieve courses")

    # Execute SQL and get rows
    df = run_sql_query(sql)
    rows = df.to_dict(orient="records")

    # Choose model
    model_path = _choose_model_path(payload)
    features = payload.get("features", "min")

    return predict_with_model(model_path, rows, features)
