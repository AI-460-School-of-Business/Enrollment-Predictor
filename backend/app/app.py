# server/app.py
from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
import psycopg2
import pickle
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

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


@app.route("/api/semesters", methods=["GET"])
def get_semesters():
    """
    Discover a table with `term_desc` and return distinct term_desc values.
    Response: { ok: True, semesters: ["Spring 2025", "Fall 2024", ...] }
    """
    conn = None
    try:
        conn = get_db_connection()
        table = _find_table_with_column(conn, "term_desc")
        if not table:
            return jsonify({"ok": False, "semesters": [], "error": "No table with 'term_desc' found"}), 404

        # If the table also has a numeric 'term' column we can order by it (newest first)
        has_term = not pd.read_sql_query(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'term' AND table_schema = 'public' LIMIT 1
            """,
            conn, params=(table,)
        ).empty

        if has_term:
            q = f"SELECT DISTINCT term_desc FROM {table} ORDER BY term DESC NULLS LAST;"
            df = pd.read_sql_query(q, conn)
        else:
            q = f"SELECT DISTINCT term_desc FROM {table} ORDER BY term_desc DESC;"
            df = pd.read_sql_query(q, conn)

        semesters = df['term_desc'].dropna().astype(str).tolist()
        return jsonify({"ok": True, "semesters": semesters})
    except Exception as exc:
        current_app.logger.exception("Failed to fetch semesters")
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if conn:
            conn.close()

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


# Prediction Logic (SQL → Model)
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
    model_dir = payload.get("model_dir", "/app/data/prediction_models")
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
    
    # Load the model
    model_data = load_model(model_path)
    
    # Verify feature schema matches
    if model_data.get("feature_schema") != features:
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


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from the backend!"})


@app.route("/api/predict/sql", methods=["POST"])
def predict_by_sql():
    """
    POST JSON payload:
    {
      "sql": "SELECT subj, crse, term, ... FROM section_detail_report_sbussection_detail_report_sbus WHERE (term / 100) = 2025;",
      ...
    }
    """
    try:
        payload = request.get_json(force=True)
        result = sql_predict(payload)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    
    return jsonify({
        "ok": False,
        "message": "Prediction endpoint available: use /api/predict/sql for SQL+model predictions."
    }), 400


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

    return jsonify({"ok": True, "sql": sql, "row_count": len(rows), "rows": rows})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
