from flask import Flask, jsonify, send_from_directory, request
import sys
import os

# Add 'ml' directory to path so internal imports in predictor_service work (e.g. 'from data...')
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from predictor_service import sql_predict
import numpy as np


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

app = Flask(__name__, static_folder="static")

# API route (test)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)