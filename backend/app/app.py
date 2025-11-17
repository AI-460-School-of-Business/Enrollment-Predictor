# server/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# Allow frontend (Vite dev server) to call this API
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from the backend!"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    This is the endpoint App.tsx will call when the user
    clicks "Generate Report".

    For now this returns dummy data.
    """

    data = request.get_json() or {}

    # You can inspect filters if you want:
    model = data.get("model")
    semesters = data.get("semesters", [])
    department = data.get("department")
    course_identifier = data.get("courseIdentifier")

    print("Received request:", {
        "model": model,
        "semesters": semesters,
        "department": department,
        "course_identifier": course_identifier,
    })

    results = [
        {
            "id": "1",
            "confidenceLevel": 92.5,
            "seatsNeeded": 45,
            "courseNumber": "BUS 101",
            "courseTitle": "Introduction to Business",
        },
        {
            "id": "2",
            "confidenceLevel": 87.3,
            "seatsNeeded": 38,
            "courseNumber": "BUS 205",
            "courseTitle": "Marketing Fundamentals",
        },
        {
            "id": "3",
            "confidenceLevel": 94.1,
            "seatsNeeded": 52,
            "courseNumber": "BUS 310",
            "courseTitle": "Financial Accounting",
        },
        {
            "id": "4",
            "confidenceLevel": 89.8,
            "seatsNeeded": 41,
            "courseNumber": "BUS 415",
            "courseTitle": "Strategic Management",
        },
        {
            "id": "5",
            "confidenceLevel": 91.2,
            "seatsNeeded": 47,
            "courseNumber": "BUS 220",
            "courseTitle": "Business Statistics",
        },
    ]

    # Return as JSON array; TS side expects ResultData[]
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
