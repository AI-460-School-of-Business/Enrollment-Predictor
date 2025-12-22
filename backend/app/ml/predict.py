"""
CLI: Enrollment Prediction Runner
-------------------------------------------------------------------------------
This script runs enrollment predictions from the command line using the shared
`ml.predictor_service` utilities.

Supported input modes (mutually exclusive):
1) --single
   Predict a single (term, subj, crse) record.

2) --sql "..."
   Execute a SQL query that returns rows to predict.

3) --sql-file path/to/query.sql
   Same as --sql, but loads the query from a file.

Model selection:
- If --model is provided, that model file is used.
- Otherwise, the script selects the latest model in `--model-dir` using
  `_choose_model_path()`.

Output:
- Prints a compact table: Subject, Course, Term, Prediction

Notes:
- The predictor pipeline expects certain columns for feature engineering.
  We ensure the following exist:
  - act (defaults to 0)
  - credits (defaults to 3)
- SQL results are normalized to lowercase column names to match the backend
  predict route behavior.

Example:
  python backend/app/ml/predict.py --single --subj FIN --crse 400 --term 202640
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ------------------------------------------------------------------------------
# Import setup
# ------------------------------------------------------------------------------

# Ensure the backend/app directory is importable when running from this file.
# (Keeps this script runnable both locally and inside a container.)
THIS_DIR = Path(__file__).resolve().parent
SERVER_ROOT = (THIS_DIR / "..").resolve()  # backend/app
sys.path.insert(0, str(SERVER_ROOT))

from ml.predictor_service import predict_with_model, run_sql_query, _choose_model_path  # noqa: E402


# ------------------------------------------------------------------------------
# CLI parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build and return an ArgumentParser for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Generate enrollment predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Predict for a single course
              python backend/app/ml/predict.py --single --subj FIN --crse 400 --term 202640

              # Predict for courses using a SQL query
              docker-compose exec web python backend/app/ml/predict.py --sql "
                SELECT term, subj, crse
                FROM section_detail_report_sbussection_detail_report_sbus
                WHERE subj='MIS' AND term=202540
              "
            """
        ).strip(),
    )

    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--single",
        action="store_true",
        help="Predict for a single course (requires --subj, --crse, --term)",
    )
    input_group.add_argument(
        "--sql",
        type=str,
        help="SQL query used to select rows to predict",
    )
    input_group.add_argument(
        "--sql-file",
        type=str,
        help="Path to a .sql file containing the query",
    )

    # Single prediction args
    parser.add_argument("--subj", type=str, help="Subject code (e.g., FIN)")
    parser.add_argument("--crse", type=str, help="Course number (e.g., 400)")
    parser.add_argument("--term", type=int, help="Term code (e.g., 202640)")
    parser.add_argument("--credits", type=int, default=3, help="Credits (default: 3)")

    # Model args
    parser.add_argument("--model", type=str, help="Path to a specific model file (.pkl)")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/backend/data/prediction_models",
        help="Directory to search for the latest model (default: /backend/data/prediction_models)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="min",
        choices=["min", "rich"],
        help="Feature schema used by the model pipeline (default: min)",
    )

    return parser


# ------------------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------------------

def resolve_model_path(model_arg: Optional[str], model_dir: str) -> str:
    """
    Determine which model path to use.

    Priority:
    1) If the user provided --model, use it directly.
    2) Otherwise, use _choose_model_path() to select the newest/most appropriate model.

    Raises:
        RuntimeError if no model can be resolved.
    """
    if model_arg:
        return model_arg

    try:
        payload = {"model_dir": model_dir}
        model_path = _choose_model_path(payload)
        print(f"Using latest model: {model_path}")
        return model_path
    except Exception as e:
        raise RuntimeError(f"Error finding model in '{model_dir}': {e}") from e


def read_sql_from_args(sql_text: Optional[str], sql_file: Optional[str]) -> str:
    """
    Return SQL query text from --sql or --sql-file.

    Precedence:
    - If sql_file is provided, it overrides sql_text (because it's explicit).
    """
    if sql_file:
        with open(sql_file, "r", encoding="utf-8") as f:
            return f.read()
    return sql_text or ""


def normalize_prediction_df(df: pd.DataFrame, default_credits: int = 3) -> pd.DataFrame:
    """
    Normalize a DataFrame returned from SQL before passing it to `predict_with_model`.

    - Lowercase all column names (matches backend behavior).
    - Ensure required columns exist:
      - act: required for some feature engineering pipelines → default 0
      - credits: some pipelines use credits → default `default_credits`
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "act" not in df.columns:
        df["act"] = 0

    if "credits" not in df.columns:
        df["credits"] = default_credits

    return df


def make_single_row(term: int, subj: str, crse: str, credits: int) -> List[Dict[str, Any]]:
    """
    Build a single-row payload for prediction.

    Note:
    - `act` is included as a dummy value so feature engineering succeeds.
    """
    return [
        {
            "term": term,
            "subj": subj,
            "crse": crse,
            "credits": credits,
            "act": 0,
        }
    ]


def print_predictions_table(feature_rows: List[Dict[str, Any]], predictions: List[float]) -> None:
    """
    Print a compact, human-readable table of predictions to stdout.
    """
    print("\n" + "=" * 60)
    print(f"{'Subject':<8} {'Course':<8} {'Term':<10} {'Prediction':<12}")
    print("-" * 60)

    for i, row in enumerate(feature_rows):
        pred = float(predictions[i])
        subj = row.get("subj", "N/A")
        crse = row.get("crse", "N/A")
        term = row.get("term", "N/A")
        print(f"{str(subj):<8} {str(crse):<8} {str(term):<10} {pred:<12.2f}")

    print("=" * 60 + "\n")


# ------------------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------------------

def main() -> int:
    """
    CLI entrypoint.

    Returns:
        Exit code (0 = success, non-zero = failure).
    """
    parser = build_parser()
    args = parser.parse_args()

    # Resolve model path first, so we fail early if models are missing.
    try:
        model_path = resolve_model_path(args.model, args.model_dir)
    except RuntimeError as e:
        print(e)
        return 1

    rows: List[Dict[str, Any]] = []

    # ---- Mode 1: Single-row prediction ----
    if args.single:
        if not (args.subj and args.crse and args.term):
            print("Error: --single requires --subj, --crse, and --term")
            return 1

        rows = make_single_row(args.term, args.subj, args.crse, args.credits)
        print(f"Predicting for single course: {args.subj} {args.crse} Term {args.term}")

    # ---- Mode 2/3: SQL-based prediction ----
    else:
        query = read_sql_from_args(args.sql, args.sql_file)
        if not query.strip():
            print("Error: empty SQL query.")
            return 1

        print("Executing SQL query...")
        try:
            df = run_sql_query(query)
            df = normalize_prediction_df(df, default_credits=3)
            rows = df.to_dict(orient="records")
            print(f"Found {len(rows)} rows to predict.")
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return 1

    if not rows:
        print("No data to predict.")
        return 0

    # Run predictions
    try:
        result = predict_with_model(model_path, rows, features=args.features)

        predictions = result.get("predictions", [])
        feature_rows = result.get("features", [])

        if not predictions or not feature_rows:
            print("Prediction succeeded but no output was returned by predictor_service.")
            return 1

        print_predictions_table(feature_rows, predictions)
        return 0

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
