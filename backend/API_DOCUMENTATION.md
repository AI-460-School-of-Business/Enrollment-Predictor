# API Documentation: /api/predict/sql

## Endpoint: POST /api/predict/sql

This endpoint accepts SQL queries to fetch data from the database and makes enrollment predictions using pre-trained machine learning models.

### Request Format

```json
{
  "sql": "SELECT subj, crse, term, sec, credits FROM section_detail_report WHERE (term / 100) = 2025;",
  "model_path": "/app/data/prediction_models/enrollment_tree_min_20251120_173605.pkl",   // optional
  "model_filename": "enrollment_tree_min_20251120_173605.pkl",                        // optional
  "model_prefix": "enrollment_tree_min_",                                              // optional, chooses newest match
  "model_dir": "/app/data/prediction_models",                                         // optional override
  "features": "min"                                                                   // optional, default "min"
}
```

### Required Parameters

- **sql** (string): SQL query to execute against the database. The query must return columns matching the model's feature requirements.

### Optional Parameters

- **model_path** (string): Direct path to a specific model file. Takes precedence over other model selection methods.
- **model_filename** (string): Name of the model file in the model directory. Used if `model_path` is not provided.
- **model_prefix** (string): Prefix to match model files. The newest matching file will be selected.
- **model_dir** (string): Directory containing model files. Default: `/app/data/prediction_models`
- **features** (string): Feature schema to use. Default: `"min"`. Must match the model's feature schema.

### Model Selection Priority

1. If `model_path` is provided, use it directly
2. Else if `model_filename` is provided, use it with `model_dir`
3. Else if `model_prefix` is provided, find the latest matching model
4. Else use default prefix: `enrollment_tree_{features}_`

### Feature Requirements

For the "min" feature schema, the SQL query must return these columns:
- `term` (integer): Term code (e.g., 202501)
- `subj` (string): Subject code (e.g., "BUS", "CS")
- `crse` (string): Course number (e.g., "101", "205")
- `sec` (string): Section number (e.g., "01", "02")
- `credits` (integer): Number of credits

### Response Format

#### Success Response (200 OK)

```json
{
  "ok": true,
  "result": [
    {
      "index": 0,
      "prediction": 12.48,
      "term": 202501,
      "subj": "BUS",
      "crse": "101",
      "sec": "01",
      "credits": 3
    },
    {
      "index": 1,
      "prediction": 15.32,
      "term": 202501,
      "subj": "BUS",
      "crse": "205",
      "sec": "02",
      "credits": 3
    }
  ]
}
```

#### Error Response (400 Bad Request)

```json
{
  "ok": false,
  "error": "Missing required parameter: sql"
}
```

### Common Error Messages

- `"Missing required parameter: sql"` - The SQL query is required
- `"Model directory not found: {path}"` - The specified model directory doesn't exist
- `"No model files found matching pattern: {pattern}"` - No models match the prefix
- `"Model feature schema '{schema1}' does not match requested features '{schema2}'"` - Schema mismatch
- `"Missing required columns in query result: {columns}"` - SQL query doesn't return all required columns

### Example Usage

```bash
curl -X POST http://localhost:5000/api/predict/sql \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT term, subj, crse, sec, credits FROM courses WHERE term = 202501 LIMIT 10",
    "features": "min"
  }'
```

### Notes

- The endpoint automatically handles database connection management
- Label encoding is applied to categorical features
- Feature scaling is applied using the model's trained scaler
- Unknown categories in encoded fields are assigned a value of -1
- The endpoint logs errors server-side for debugging
