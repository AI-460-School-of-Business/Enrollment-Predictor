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
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, current_app, jsonify, request, send_from_directory, send_file, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ml.predictor_service import _choose_model_path, predict_with_model
from ml.data.data_loader import DataLoader
from ml.data.feature_engineer import FeatureEngineer
from ml.models.linear_predictor import LinearRegressionPredictor
from ml.models.tree_predictor import TreePredictor
from ml.models.neural_predictor import NeuralNetworkPredictor


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
REPO_ROOT = BASE_DIR.parent.parent
REPORT_SCRIPT = (REPO_ROOT / "frontend" / "src" / "generate_report.py").resolve()

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

# --------------------------------------------------------------------------
# Model storage
# --------------------------------------------------------------------------

MODEL_DIR = (Path(__file__).resolve().parent.parent / "data" / "prediction_models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CSV_DIR = (Path(__file__).resolve().parent.parent / "data" / "csv").resolve()
TRAIN_CSV_DIR.mkdir(parents=True, exist_ok=True)


def _list_available_models() -> list[dict[str, Any]]:
    """Return metadata for .pkl models in MODEL_DIR, newest first."""
    models: list[dict[str, Any]] = []
    for path in sorted(MODEL_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        models.append(
            {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    return models


def _build_predictor(model_type: str, feature_schema: str, custom_query: Optional[str]):
    """Factory mirroring train_model.py for API usage."""
    model_type = (model_type or "tree").lower()
    feature_schema = (feature_schema or "min").lower()

    if model_type == "linear":
        return LinearRegressionPredictor(feature_schema, custom_query=custom_query)
    if model_type == "tree":
        return TreePredictor(feature_schema, custom_query=custom_query)
    if model_type == "neural":
        return NeuralNetworkPredictor(feature_schema, custom_query=custom_query)

    raise ValueError(f"Unsupported model type: {model_type}")


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

# ------------------------------------------------------------------------------
# Models: list and upload
# ------------------------------------------------------------------------------


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available model files (.pkl) from the prediction_models directory."""
    try:
        return jsonify({"ok": True, "models": _list_available_models()})
    except Exception as exc:
        current_app.logger.exception("Failed to list models")
        return jsonify({"ok": False, "error": str(exc), "models": []}), 500


@app.route("/api/models", methods=["POST"])
def upload_model():
    """
    Save an uploaded .pkl model into the prediction_models directory.

    Expects multipart/form-data with field name 'model'.
    """
    try:
        if "model" not in request.files:
            return jsonify({"ok": False, "error": "Missing uploaded file 'model'"}), 400

        file = request.files["model"]
        if not file or not file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".pkl"):
            return jsonify({"ok": False, "error": "Only .pkl files are allowed"}), 400

        target = MODEL_DIR / filename
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = MODEL_DIR / f"{target.stem}_{timestamp}{target.suffix}"
            filename = target.name

        file.save(target)
        models = _list_available_models()

        return jsonify({"ok": True, "filename": filename, "models": models})
    except Exception as exc:
        current_app.logger.exception("Failed to upload model")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/train/upload-file", methods=["POST"])
def upload_training_file():
    """
    Upload training CSVs to the container's data/csv directory.

    Expects multipart/form-data with field name 'file'.
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "Missing uploaded file 'file'"}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".csv"):
            return jsonify({"ok": False, "error": "Only .csv files are allowed"}), 400

        target = TRAIN_CSV_DIR / filename
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = TRAIN_CSV_DIR / f"{target.stem}_{timestamp}{target.suffix}"
            filename = target.name

        file.save(target)
        return jsonify({"ok": True, "filename": filename, "path": str(target)})
    except Exception as exc:
        current_app.logger.exception("Failed to upload training file")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/train", methods=["POST"])
def train_route():
    """
    Train a model using the train_model.py pipeline with selected semesters.
    Body (JSON):
      {
        "model": "tree" | "linear" | "neural",
        "features": "min" | "rich",
        "terms": [202430, 202440]   # required
      }
    """
    try:
        payload = request.get_json(silent=True) or {}
        model_type = payload.get("model", "tree")
        feature_schema = payload.get("features", "min")
        terms_raw = payload.get("terms") or []

        try:
            term_values = [int(t) for t in terms_raw if str(t).strip() != ""]
        except Exception:
            return jsonify({"ok": False, "error": "Invalid term values"}), 400

        if not term_values:
            return jsonify({"ok": False, "error": "At least one term is required"}), 400

        act_threshold = 10
        term_list = ", ".join(str(t) for t in term_values)
        custom_query = (
            "SELECT * FROM section_detail_report_sbussection_detail_report_sbus "
            f"WHERE term IN ({term_list}) AND act > {act_threshold}"
        )

        predictor = _build_predictor(model_type, feature_schema, custom_query=custom_query)

        data_loader = DataLoader(custom_query)
        raw_data = data_loader.extract_training_data()

        fe = FeatureEngineer(feature_schema)
        X, y = fe.prepare_features(raw_data)
        predictor.feature_columns = fe.feature_columns

        predictor.train(X, y)

        sorted_terms = sorted(term_values)

        def _season_from_term(term_value: int) -> str:
            suffix = term_value % 100
            mapping = {
                10: "fall",   
                40: "spring",
            }
            return mapping.get(suffix, f"term{suffix}")

        start_term = sorted_terms[0]
        end_term = sorted_terms[-1]
        start_year = start_term // 100
        end_year = end_term // 100
        start_season = _season_from_term(start_term)
        end_season = _season_from_term(end_term)

        model_filename = (
            f"enrollment_tree_{feature_schema}_{model_type}_"
            f"{start_season}_{start_year}_to_{end_season}_{end_year}.pkl"
        )
        model_path = MODEL_DIR / model_filename
        predictor.save_model(str(model_path))

        return jsonify(
            {
                "ok": True,
                "model_filename": model_filename,
                "saved_path": str(model_path),
                "terms": term_values,
                "row_count": int(getattr(X, "shape", [0])[0]),
            }
        )
    except Exception as exc:
        current_app.logger.exception("Training failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/reports/export", methods=["POST"])
def export_report():
    """Generate and return an Excel report using generate_report.py."""
    try:
        data = request.get_json(silent=True) or {}
        rows = data.get("rows")
        accuracy_csv = data.get("accuracy_csv")
        model_info = data.get("model_info") or {}

        if not isinstance(rows, list) or not rows:
            return jsonify({"ok": False, "error": "rows must be a non-empty list"}), 400

        if not REPORT_SCRIPT.exists():
            return jsonify({"ok": False, "error": f"Report script not found at {REPORT_SCRIPT}"}), 500

        temp_dir = Path(tempfile.mkdtemp(prefix="report_export_"))
        predictions_path = temp_dir / "predictions.json"
        output_path = temp_dir / f"enrollment_report_{uuid4().hex}.xlsx"

        predictions_path.write_text(json.dumps(rows), encoding="utf-8")

        cmd = [
            "python",
            str(REPORT_SCRIPT),
            "--json",
            str(predictions_path),
            "--output",
            str(output_path),
        ]

        if accuracy_csv:
            cmd += ["--accuracy", accuracy_csv]
        if model_info.get("model_type"):
            cmd += ["--model-type", model_info["model_type"]]
        if model_info.get("model_name"):
            cmd += ["--model-name", model_info["model_name"]]
        if model_info.get("feature_schema"):
            cmd += ["--feature-schema", model_info["feature_schema"]]

        subprocess.run(cmd, check=True)

        if not output_path.exists():
            raise FileNotFoundError("Report generation did not produce output file")

        @after_this_request
        def cleanup(response):
            try:
                for path in (predictions_path, output_path):
                    if path.exists():
                        path.unlink()
                temp_dir.rmdir()
            except Exception:
                pass
            return response

        return send_file(
            str(output_path),
            as_attachment=True,
            download_name="enrollment_report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except subprocess.CalledProcessError as exc:
        current_app.logger.exception("Report generation failed")
        return jsonify({"ok": False, "error": f"Report generation failed: {exc}"}), 500
    except Exception as exc:
        current_app.logger.exception("Report export failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


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
    model_dir = payload.get("model_dir", str(MODEL_DIR))
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
