"""
app.py
-------------------------------------------------------------------------------
Flask API for the Course Sense / Enrollment Predictor application.

What this service does:
- Provides reference data endpoints:
  - GET /api/semesters     → distinct (term, term_desc)
  - GET /api/departments   → distinct subject codes mapped to human-readable names
- Provides prediction endpoint:
  - POST /api/predict/sql  → execute a SQL query, choose/load an ML model, return predictions
- Provides a debug SQL endpoint:
  - GET /sql?sql=...       → run ad-hoc SQL and return rows (use with caution)

Key implementation notes:
- Uses Postgres via psycopg2.
- Uses pandas to run SQL and shape results.
- Uses `ml.predictor_service.predict_with_model` for feature engineering + inference.
- Converts numpy/pandas objects into JSON-safe native Python types before returning.

Security note:
- `/sql` and `/api/predict/sql` can execute arbitrary SQL if exposed publicly.
  Restrict or remove these routes in production, or add authentication + validation.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, current_app, jsonify, request, send_from_directory
from flask_cors import CORS

from ml.predictor_service import _choose_model_path, predict_with_model


# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = Flask(__name__)

# Allow frontend (e.g., Vite dev server) to call this API.
# In production, strongly consider restricting `origins` to your domain(s).
CORS(app, resources={r"/*": {"origins": "*"}})


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _to_native(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas objects into JSON-serializable native types.

    Handles:
    - dict, list, tuple
    - numpy scalars (np.generic)
    - numpy arrays (np.ndarray)

    Note:
    - pandas scalars generally show up as numpy scalars.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_native(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------------------
# Subject → Department mapping
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent  # .../backend/app

# Candidate locations to support different working directories (app folder vs backend root).
CANDIDATE_MAP_PATHS = [
    BASE_DIR / "subjectDepartmentMap.json",        # .../backend/app/subjectDepartmentMap.json
    BASE_DIR.parent / "subjectDepartmentMap.json", # .../backend/subjectDepartmentMap.json
]

SUBJECT_DEPT_MAP: dict[str, str] = {}
_loaded_map_path: Optional[Path] = None

for p in CANDIDATE_MAP_PATHS:
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            SUBJECT_DEPT_MAP = json.load(f) or {}
        _loaded_map_path = p
        app.logger.info("Loaded subjectDepartmentMap.json from %s", p)
        break

if not SUBJECT_DEPT_MAP:
    app.logger.error(
        "subjectDepartmentMap.json not found or empty. Checked: %s",
        ", ".join(str(p) for p in CANDIDATE_MAP_PATHS),
    )


# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------

def get_db_connection():
    """
    Create and return a new Postgres connection.

    Environment variables (with defaults):
    - DB_HOST            (default: "db")
    - DB_PORT            (default: "5432")
    - POSTGRES_DB        (default: "enrollprdctDB")
    - POSTGRES_USER      (default: "DBUser")
    - POSTGRES_PASSWORD  (default: "DBPassword")
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "enrollprdctDB"),
        user=os.getenv("POSTGRES_USER", "DBUser"),
        password=os.getenv("POSTGRES_PASSWORD", "DBPassword"),
    )


def _find_table_with_column(conn, column_name: str) -> Optional[str]:
    """
    Return the first public table name that contains the specified column, or None.

    This is a convenience method for environments where the specific table name
    may differ, but the schema includes known columns (e.g., 'subj').
    """
    q = """
        SELECT table_name
        FROM information_schema.columns
        WHERE column_name = %s
          AND table_schema = 'public'
        ORDER BY table_name
        LIMIT 1;
    """
    df = pd.read_sql_query(q, conn, params=(column_name,))
    return df["table_name"].iloc[0] if not df.empty else None


# ------------------------------------------------------------------------------
# Model helpers (mostly legacy; prediction flow primarily uses _choose_model_path)
# ------------------------------------------------------------------------------

def find_latest_model(model_dir: str, prefix: str) -> str:
    """
    Find the latest model file in `model_dir` matching `prefix*.pkl`.

    NOTE:
    - Current code path uses `_choose_model_path` instead, but this helper is kept
      for compatibility / future use.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pattern = f"{prefix}*.pkl"
    matching_files = list(model_path.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(f"No model files found matching pattern: {pattern}")

    # If filenames include timestamps, reverse sort often works; otherwise consider mtime.
    matching_files.sort(reverse=True)
    return str(matching_files[0])


def load_model(model_path: str):
    """
    Load a pickled model from disk.

    NOTE:
    - Not currently used in sql_predict since `predict_with_model` handles loading.
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------
# Routes: reference data
# ------------------------------------------------------------------------------

@app.route("/api/semesters", methods=["GET"])
def get_semesters():
    """
    Return distinct (term, term_desc) values from the first public table that
    contains BOTH columns.

    Response:
      {
        "ok": true,
        "semesters": [
          { "term": 202420, "term_desc": "Spring 2024" },
          ...
        ]
      }
    """
    conn = None
    try:
        conn = get_db_connection()

        # Find a public table that has both term and term_desc.
        q_table = """
            SELECT table_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name IN ('term', 'term_desc')
            GROUP BY table_name
            HAVING COUNT(DISTINCT column_name) = 2
            ORDER BY table_name
            LIMIT 1;
        """
        df_table = pd.read_sql_query(q_table, conn)
        if df_table.empty:
            return (
                jsonify(
                    {
                        "ok": False,
                        "semesters": [],
                        "error": "No table with both 'term' and 'term_desc' found",
                    }
                ),
                404,
            )

        table = df_table["table_name"].iloc[0]

        # Fetch distinct semesters and sort ascending by term.
        q_semesters = f"""
            SELECT DISTINCT term, term_desc
            FROM {table}
            WHERE term IS NOT NULL AND term_desc IS NOT NULL
            ORDER BY term ASC;
        """
        df_semesters = pd.read_sql_query(q_semesters, conn)

        semesters: list[dict[str, Any]] = []
        for _, row in df_semesters.iterrows():
            term_val = row["term"]
            term_desc_val = row["term_desc"]
            if pd.isna(term_val) or pd.isna(term_desc_val):
                continue
            semesters.append({"term": int(term_val), "term_desc": str(term_desc_val)})

        return jsonify({"ok": True, "semesters": semesters})

    except Exception as exc:
        current_app.logger.exception("Failed to fetch semesters")
        return jsonify({"ok": False, "error": str(exc), "semesters": []}), 500
    finally:
        if conn:
            conn.close()


@app.route("/api/departments", methods=["GET"])
def get_departments():
    """
    Return distinct department subject codes from the DB, mapped to readable names.

    Logic:
    - Find a public table containing `subj`.
    - Get DISTINCT subj values.
    - Keep only codes that exist in subjectDepartmentMap.json.
    """
    conn = None
    try:
        conn = get_db_connection()

        table = _find_table_with_column(conn, "subj")
        if not table:
            return (
                jsonify(
                    {
                        "ok": False,
                        "departments": [],
                        "error": "No table with column 'subj' found",
                    }
                ),
                404,
            )

        q_subj = f"""
            SELECT DISTINCT subj
            FROM {table}
            WHERE subj IS NOT NULL;
        """
        df_subj = pd.read_sql_query(q_subj, conn)

        if df_subj.empty:
            return jsonify({"ok": True, "departments": []})

        # Keep only codes present in the JSON mapping.
        dept_dict: dict[str, str] = {}
        for _, row in df_subj.iterrows():
            code = row["subj"]
            if pd.isna(code):
                continue

            code_str = str(code).strip().upper()
            if code_str in SUBJECT_DEPT_MAP:
                dept_dict[code_str] = SUBJECT_DEPT_MAP[code_str]

        departments = [{"code": c, "name": n} for c, n in dept_dict.items()]
        departments.sort(key=lambda d: d["name"])  # stable UX ordering by name

        return jsonify({"ok": True, "departments": departments})

    except Exception as exc:
        current_app.logger.exception("Failed to fetch departments")
        return jsonify({"ok": False, "error": str(exc), "departments": []}), 500
    finally:
        if conn:
            conn.close()


# ------------------------------------------------------------------------------
# Prediction (SQL → model inference)
# ------------------------------------------------------------------------------

def sql_predict(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Execute a SQL query, choose a model path, and run predictions using `predict_with_model`.

    Payload fields:
    - sql (str) [required]
    - model_path (str) [optional]      → absolute/relative model file path
    - model_filename (str) [optional]  → filename within model_dir
    - model_prefix (str) [optional]    → used to find the latest matching model
    - model_dir (str) [optional]       → directory containing models
    - features (str) [optional]        → feature schema to use (default: "min")

    Returns:
    - list of dicts, each containing:
      {
        "index": 0,
        "prediction": 42.0,
        ...original SQL columns...
      }
    """
    sql_query = payload.get("sql")
    if not sql_query:
        raise ValueError("Missing required parameter: sql")

    model_path = payload.get("model_path")
    model_filename = payload.get("model_filename")
    model_prefix = payload.get("model_prefix")
    model_dir = payload.get("model_dir", "backend/data/prediction_models")
    features = payload.get("features", "min")

    # --- Choose the model path ---
    if model_path:
        # Use as provided.
        pass
    elif model_filename:
        model_path = os.path.join(model_dir, model_filename)
    else:
        # Delegate to the shared chooser logic used by the CLI.
        chooser_payload: dict[str, Any] = {"model_dir": model_dir}
        chooser_payload["model_prefix"] = model_prefix or f"enrollment_tree_{features}_"

        try:
            model_path = _choose_model_path(chooser_payload)
        except Exception as e:
            raise ValueError(f"Error finding model: {e}") from e

    # --- Execute SQL query ---
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()

    if df.empty:
        return []

    # Normalize column casing to match predictor expectations.
    df.columns = [c.lower() for c in df.columns]

    # Ensure defaults exist (matches prior CLI behavior).
    if "act" not in df.columns:
        df["act"] = 0
    if "credits" not in df.columns:
        df["credits"] = 3

    # Convert dataframe rows into list-of-dicts for the predictor service.
    rows = df.to_dict(orient="records")

    # `predict_with_model` handles loading, preprocessing, and inference.
    result = predict_with_model(model_path, rows, features=features)

    predictions = result.get("predictions") or []

    # Build response rows containing prediction + original columns.
    out: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        record: dict[str, Any] = {
            "index": int(idx),
            "prediction": float(pred),
        }

        # Attach original row columns for context
        for col in df.columns:
            val = df.iloc[idx][col]

            # pandas often stores scalars as numpy types (need JSON-safe conversion)
            if pd.isna(val):
                record[col] = None
            elif hasattr(val, "item"):
                try:
                    record[col] = val.item()
                except Exception:
                    record[col] = val
            else:
                record[col] = val

        out.append(record)

    return out


@app.route("/api/predict/sql", methods=["POST"])
def predict_sql_route():
    """
    POST /api/predict/sql

    Body:
      { "sql": "...", "features": "min", "model_prefix": "...", ... }

    Response:
      - On success: JSON list of rows with predictions.
      - On failure: { "error": "..." }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "sql" not in data:
            return jsonify({"error": "Request body must contain 'sql' field"}), 400

        result = sql_predict(data)

        # Ensure JSON-safe output (numpy/pandas types → native Python types)
        safe_result = _to_native(result)
        return jsonify(safe_result)

    except Exception as e:
        current_app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# Misc / debug routes
# ------------------------------------------------------------------------------

@app.route("/api/hello")
def hello():
    """Simple health-ish check route."""
    return jsonify({"message": "Hello from the backend!"})


@app.route("/")
def serve_index():
    """
    Serve frontend index.html (only works if app.static_folder is configured).
    If you're serving Vite separately in dev, you may not use this route.
    """
    return send_from_directory(app.static_folder, "index.html")


@app.route("/sql", methods=["GET"])
def sql_request():
    """
    GET /sql?sql=...

    Executes arbitrary SQL and returns rows.

    WARNING:
    - This is dangerous if exposed publicly.
    - Keep dev-only or add auth + allowlists.
    """
    sql = request.args.get("sql")
    if not sql:
        return jsonify({"ok": False, "error": "Missing required parameter: sql"}), 400

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as exc:
        current_app.logger.exception("SQL request failed")
        return jsonify({"ok": False, "error": str(exc)}), 400
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Convert dataframe to JSON-safe rows.
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row: dict[str, Any] = {}
        for col in df.columns:
            val = df.iloc[idx][col]

            if pd.isna(val):
                row[col] = None
                continue

            if hasattr(val, "item"):
                try:
                    row[col] = val.item()
                except Exception:
                    row[col] = val
            else:
                row[col] = val

        rows.append(row)

    return jsonify({"ok": True, "sql": sql, "row_count": len(rows), "rows": rows})


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Debug=True is convenient for dev; use a real WSGI server (gunicorn/uwsgi)
    # and debug=False in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
