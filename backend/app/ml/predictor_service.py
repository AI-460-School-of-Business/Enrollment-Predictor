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


# Default model to use when none provided / found in MODEL_DIR
DEFAULT_MODEL_PATH = "backend/data/prediction_models/enrollment_tree_min_20251203_185724.pkl"


def load_saved_model(model_path: str) -> Tuple[Any, Optional[List[str]], Optional[Any], Optional[Dict]]:
    """Load a pickled model from disk.

    Returns a tuple (model, feature_columns, scaler, label_encoders) where some may be None
    if the saved object did not include them.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with p.open("rb") as fh:
        saved = pickle.load(fh)

    if isinstance(saved, dict):
        # Common saved layout: {"model": estimator, "feature_columns": [...], "scaler": ..., "label_encoders": ...}
        model = saved.get("model") or saved.get("estimator") or next(iter(saved.values()))
        feature_columns = saved.get("feature_columns") or saved.get("feature_cols")
        scaler = saved.get("scaler")
        label_encoders = saved.get("label_encoders", {})
    else:
        model = saved
        feature_columns = getattr(model, "feature_columns", None)
        scaler = getattr(model, "scaler", None)
        label_encoders = getattr(model, "label_encoders", {})

    return model, feature_columns, scaler, label_encoders


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
    model, saved_feature_cols, scaler, label_encoders = load_saved_model(model_path)

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

    # Apply the same preprocessing as during training
    X_processed = X.copy()
    
    # Apply label encoding to categorical features
    if label_encoders:
        for col, le in label_encoders.items():
            if col in X_processed.columns:
                # Convert to string and handle unknown categories
                col_values = X_processed[col].astype(str)
                unique_values = set(col_values)
                known_values = set(le.classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    # Replace unknown values with the most frequent known value
                    most_frequent = le.classes_[0]
                    col_values = col_values.replace(list(unknown_values), most_frequent)
                
                X_processed[col] = le.transform(col_values)
    
    # Apply scaling if scaler was saved
    if scaler:
        X_processed = scaler.transform(X_processed)

    # Make predictions
    preds = model.predict(X_processed)

    # Convert predictions to plain Python types
    preds_list = [float(p) for p in preds]

    return {
        "predictions": preds_list,
        "features": X.to_dict(orient='records'),
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
    model_path = payload.get("model_path") or DEFAULT_MODEL_PATH
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
    model_dir = Path(payload.get("model_dir") or os.getenv("MODEL_DIR") or "/backend/data/prediction_models")

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

    # Final fallback: explicit default model path (if it exists)
    if Path(DEFAULT_MODEL_PATH).exists():
        return str(Path(DEFAULT_MODEL_PATH))

    raise FileNotFoundError(f"No model files found in {model_dir} and default model not present: {DEFAULT_MODEL_PATH}")

def read_prediction_models(model_dir: Optional[str] = None, pattern: str = "*.pkl") -> List[Dict[str, str]]:
        """Return available prediction model files.
 
        Args:
            - model_dir: optional path to search. If not provided the function will
                consult the `MODEL_DIR` environment variable and finally fall back to
                the repository's `backend/data/prediction_models` directory which is
                convenient for local development.
            - pattern: glob pattern to match files (default: "*.pkl").
 
        Returns a list of dicts with keys `name` and `path`, ordered newest-first.
        """
        model_dir = Path(model_dir or os.getenv("MODEL_DIR") or "/workspaces/Enrollment-Predictor/backend/data/prediction_models")
 
        if not model_dir.exists():
                return []
 
        files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
 
        return [{"name": p.name, "path": str(p)} for p in files]

def sql_predict(payload):
    """
    Execute SQL query, load model, and make predictions.
    
    Args:
        payload (dict): Request payload containing:
            - sql (str): SQL query to execute
            - model_path (str, optional): Direct path to model file
            - model_filename (str, optional): Model filename in model_dir
            - model_prefix (str, optional): Prefix to find latest model
            - model_dir (str, optional): Directory containing models
            - features (str, optional): Feature schema to use (default: "min")
    
    Returns:
        list: Prediction results
    """
    # Extract parameters
    sql_query = payload.get("sql")
    if not sql_query:
        raise ValueError("Missing required parameter: sql")
    
    model_path = payload.get("model_path")
    model_filename = payload.get("model_filename")
    model_prefix = payload.get("model_prefix")
    model_dir = payload.get("model_dir", "backend/data/prediction_models")
    features = payload.get("features", "min")
    
    # Determine model path
    if model_path:
        # Use direct path
        pass
    elif model_filename:
        # Use filename in model_dir
        model_path = os.path.join(model_dir, model_filename)
    elif model_prefix:
        # Find latest model matching prefix
        model_path = find_latest_model(model_dir, model_prefix)
    else:
        # Default: find latest model with default prefix
        model_path = find_latest_model(model_dir, f"enrollment_tree_{features}_")
    
    # Load the model using shared loader (returns estimator and feature columns)
    model_obj, saved_feature_cols = load_saved_model(model_path)

    # Also inspect the raw pickle to retrieve auxiliary metadata if present
    with open(model_path, 'rb') as f:
        raw_saved = pickle.load(f)

    # If the raw saved object is a dict, prefer its metadata keys
    if isinstance(raw_saved, dict):
        model_data = raw_saved
    else:
        # Construct a minimal metadata dict from available attributes
        model_data = {
            "model": model_obj,
            "feature_columns": saved_feature_cols or getattr(model_obj, "feature_columns", None),
            "feature_schema": getattr(raw_saved, "feature_schema", None) or getattr(model_obj, "feature_schema", None),
            "label_encoders": getattr(raw_saved, "label_encoders", None) or getattr(model_obj, "label_encoders", None),
            "scaler": getattr(raw_saved, "scaler", None) or getattr(model_obj, "scaler", None),
        }

    # Verify feature schema matches when available
    if model_data.get("feature_schema") is not None and model_data.get("feature_schema") != features:
        raise ValueError(
            f"Model feature schema '{model_data.get('feature_schema')}' "
            f"does not match requested features '{features}'"
        )

    # Execute SQL query
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()
    
    if df.empty:
        return []
    
    # Extract and prepare features
    feature_columns = model_data["feature_columns"]
    
    # Verify all required columns are present
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in query result: {missing_cols}")
    
    # Prepare features
    X = df[feature_columns].copy()
    
    # Apply label encoding for categorical columns
    label_encoders = model_data["label_encoders"]
    for col, encoder in label_encoders.items():
        if col in X.columns:
            # Handle unknown categories by using a default value
            X[col] = X[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    # Scale features
    scaler = model_data["scaler"]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    model = model_data["model"]
    predictions = model.predict(X_scaled)
    
    # Prepare results
    results = []
    for idx, pred in enumerate(predictions):
        result = {
            "index": int(idx),
            "prediction": float(pred),
        }
        # Include original row data for context
        for col in df.columns:
            result[col] = df.iloc[idx][col].item() if hasattr(df.iloc[idx][col], 'item') else df.iloc[idx][col]
        results.append(result)
    
    return results