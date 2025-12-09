# server/app.py
from flask import Flask, jsonify, send_from_directory, request, current_app
from flask_cors import CORS
import psycopg2
import pickle
import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from ml.predictor_service import predict_with_model, _choose_model_path


def _to_native(obj):
    """Recursively convert numpy/pandas scalar/arrays to native Python types."""
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

app = Flask(__name__)

# Load subject → department mapping
BASE_DIR = Path(__file__).resolve().parent  # .../backend/app
CANDIDATE_PATHS = [
    BASE_DIR / "subjectDepartmentMap.json",        # .../backend/app/subjectDepartmentMap.json
    BASE_DIR.parent / "subjectDepartmentMap.json"  # .../backend/subjectDepartmentMap.json
]

SUBJECT_DEPT_MAP: dict[str, str] = {}
_loaded_path = None

for path in CANDIDATE_PATHS:
    if path.exists():
        with open(path, "r") as f:
            SUBJECT_DEPT_MAP = json.load(f)
        _loaded_path = path
        app.logger.info("Loaded subjectDepartmentMap.json from %s", path)
        break

if not SUBJECT_DEPT_MAP:
    app.logger.error(
        "subjectDepartmentMap.json not found or empty. Checked: %s",
        ", ".join(str(p) for p in CANDIDATE_PATHS),
    )
# Allow frontend (Vite dev server) to call this API
CORS(app, resources={r"/*": {"origins": "*"}})

# Database Connection
def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "enrollprdctDB"),
        user=os.getenv("POSTGRES_USER", "DBUser"),
        password=os.getenv("POSTGRES_PASSWORD", "DBPassword")
    )

def _find_table_with_column(conn, column_name):
    """Return the first public table name that contains column_name, or None."""
    q = """
        SELECT table_name
        FROM information_schema.columns
        WHERE column_name = %s
          AND table_schema = 'public'
        ORDER BY table_name
        LIMIT 1;
    """
    df = pd.read_sql_query(q, conn, params=(column_name,))
    return df['table_name'].iloc[0] if not df.empty else None

# Model Loading
def find_latest_model(model_dir, prefix):
    """Find the latest model file matching the given prefix."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find all files matching the prefix pattern
    pattern = f"{prefix}*.pkl"
    matching_files = list(model_path.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
    
    # Sort by modification time (newest first) or by name if timestamps are in filename
    matching_files.sort(reverse=True)
    return str(matching_files[0])


def load_model(model_path):
    """Load a pickled model from the given path."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@app.route("/api/semesters", methods=["GET"])
def get_semesters():
    """
    Return distinct (term, term_desc) values from a table that has them.
    Response:
      {
        "ok": True,
        "semesters": [
          { "term": 202420, "term_desc": "Spring 2024" },
          { "term": 202430, "term_desc": "Summer 2024" },
          ...
        ]
      }
    """
    conn = None
    try:
        conn = get_db_connection()

        # Find a table that has BOTH term and term_desc
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
            return jsonify({
                "ok": False,
                "semesters": [],
                "error": "No table with both 'term' and 'term_desc' found"
            }), 404

        table = df_table["table_name"].iloc[0]

        # Get distinct terms + descriptions, sorted by term (lowest → highest)
        q_semesters = f"""
            SELECT DISTINCT term, term_desc
            FROM {table}
            WHERE term IS NOT NULL AND term_desc IS NOT NULL
            ORDER BY term ASC;
        """
        df_semesters = pd.read_sql_query(q_semesters, conn)

        semesters = []
        for _, row in df_semesters.iterrows():
            term_val = row["term"]
            term_desc_val = row["term_desc"]
            if pd.isna(term_val) or pd.isna(term_desc_val):
                continue
            semesters.append({
                "term": int(term_val),
                "term_desc": str(term_desc_val),
            })

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
    Return departments whose subject codes appear in the database's `subj` column,
    mapped via subjectDepartmentMap.json.

    Response:
    {
      "ok": true,
      "departments": [
        { "code": "AC",  "name": "Accounting" },
        { "code": "FIN", "name": "Finance" },
        ...
      ]
    }
    """
    conn = None
    try:
        conn = get_db_connection()

        # Find a table that has a 'subj' column
        table = _find_table_with_column(conn, "subj")
        if not table:
            return jsonify({
                "ok": False,
                "departments": [],
                "error": "No table with column 'subj' found"
            }), 404

        # Get distinct subject codes from that table
        q_subj = f"""
            SELECT DISTINCT subj
            FROM {table}
            WHERE subj IS NOT NULL;
        """
        df_subj = pd.read_sql_query(q_subj, conn)

        if df_subj.empty:
            return jsonify({"ok": True, "departments": []})

        # Map subj codes to department names using SUBJECT_DEPT_MAP
        dept_dict = {}  # code -> name (dedup)
        for _, row in df_subj.iterrows():
            code = row["subj"]
            if pd.isna(code):
                continue
            code_str = str(code).strip().upper()
            if code_str in SUBJECT_DEPT_MAP:
                dept_dict[code_str] = SUBJECT_DEPT_MAP[code_str]

        departments = [
            {"code": code, "name": name}
            for code, name in dept_dict.items()
        ]

        # Sort alphabetically by department name
        departments.sort(key=lambda d: d["name"])

        return jsonify({"ok": True, "departments": departments})

    except Exception as exc:
        current_app.logger.exception("Failed to fetch departments")
        return jsonify({"ok": False, "error": str(exc), "departments": []}), 500
    finally:
        if conn:
            conn.close()

# Prediction Logic (SQL → Model)
def sql_predict(payload):
    """
    Execute SQL query, load/choose model, and make predictions using predict_with_model.

    Args:
        payload (dict): Request payload containing:
            - sql (str): SQL query to execute
            - model_path (str, optional): Direct path to model file
            - model_filename (str, optional): Model filename in model_dir
            - model_prefix (str, optional): Prefix to find latest model
            - model_dir (str, optional): Directory containing models
            - features (str, optional): Feature schema to use (default: "min")

    Returns:
        list: Prediction results (each item includes index, prediction, and original row columns)
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
        # use as given
        pass
    elif model_filename:
        model_path = os.path.join(model_dir, model_filename)
    else:
        # Use _choose_model_path helper to pick latest model (respecting provided prefix if any)
        chooser_payload = {"model_dir": model_dir}
        if model_prefix:
            chooser_payload["model_prefix"] = model_prefix
        else:
            chooser_payload["model_prefix"] = f"enrollment_tree_{features}_"
        try:
            model_path = _choose_model_path(chooser_payload)
        except Exception as e:
            raise ValueError(f"Error finding model: {e}")

    # Execute SQL query
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()

    if df.empty:
        return []

    # Normalize columns to lowercase (match predict.py behavior)
    df.columns = [c.lower() for c in df.columns]

    # Ensure required defaults match CLI behavior in predict.py
    if 'act' not in df.columns:
        df['act'] = 0
    if 'credits' not in df.columns:
        df['credits'] = 3

    # Convert rows to the list-of-dicts shape expected by predict_with_model
    rows = df.to_dict(orient='records')

    # Delegate loading, feature engineering, encoding, scaling, prediction to predict_with_model
    result = predict_with_model(model_path, rows, features=features)

    # predict_with_model is expected to return a dict with at least:
    #  - 'predictions': sequence of prediction values (aligned with 'features' returned)
    #  - optionally 'features' or other metadata
    predictions = result.get('predictions', [])
    if predictions is None:
        predictions = []

    # Prepare output list similar to the original function: include index, prediction, and original row data
    results = []
    for idx, pred in enumerate(predictions):
        res = {
            "index": int(idx),
            "prediction": float(pred),
        }
        # include original row columns for context
        for col in df.columns:
            val = df.iloc[idx][col]
            try:
                # convert numpy scalars to python primitives when possible
                res[col] = val.item() if hasattr(val, "item") else val
            except Exception:
                res[col] = val
        results.append(res)

    return results

@app.route("/api/hello")
def hello():
    return jsonify({"message": "Hello from the backend!"})

@app.route("/api/predict/sql", methods=['POST'])
def predict_sql_route():
    try:
        data = request.get_json()
        if not data or 'sql' not in data:
            return jsonify({"error": "Request body must contain 'sql' field"}), 400
        
        # Run prediction
        # sql_predict returns a dict with 'predictions', 'features', etc.
        result = sql_predict(data)

        # Convert numpy/pandas types to native Python types for JSON
        safe_result = _to_native(result)

        return jsonify(safe_result)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# Serve frontend (index.html)
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/sql", methods=["GET"])
def sql_request():
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

    # JSON conversion
    rows = []
    for idx in range(len(df)):
        row = {}
        for col in df.columns:
            val = df.iloc[idx][col]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            if pd.isna(val):
                val = None
            row[col] = val
        rows.append(row)

    return jsonify({
        "ok": True,
        "sql": sql,
        "row_count": len(rows),
        "rows": rows
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
