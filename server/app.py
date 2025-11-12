from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "*"}})

# API route (test)
@app.route("/api/hello")
def hello():
    return jsonify({"message": "Hello from the backend!"})

# Serve frontend (index.html)
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=True)