from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# API route (test)
@app.route("/api/hello")
def hello():
    return jsonify({"message": "Hello from the backend!"})

# Serve frontend (index.html)
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)