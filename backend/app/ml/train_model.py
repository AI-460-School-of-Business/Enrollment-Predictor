"""
Enrollment Prediction Model Training CLI
-------------------------------------------------------------------------------
Train an enrollment prediction model from the command line.

Overview
- Trains a model to predict enrollment counts using course identity
  (subj + crse) and term-related features (semester/year, etc.).
- Supports optional custom SQL queries to control which rows are used for training.

Usage
  python server/ml/train_model.py --model [linear|tree|neural] --features [min|rich]

Examples
  # Train latest tree model with minimal features
  python server/ml/train_model.py --model tree --features min

  # Train with a custom query from file
  python server/ml/train_model.py --model tree --features rich --data-query-file ./train_query.sql

  # Train with interactive query editing
  python server/ml/train_model.py --model neural --interactive-query

Notes / Assumptions
- `DataLoader(custom_query)` is responsible for pulling training data from the DB.
- `FeatureEngineer(schema)` performs feature creation / target extraction.
- Each predictor type implements:
  - train(X, y) -> results dict
  - save_model(path)
  - feature_columns attribute (set here after feature engineering)
"""

from __future__ import annotations

import argparse
import sys
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Sequence

# ------------------------------------------------------------------------------
# Import setup
# ------------------------------------------------------------------------------

# Add server directory to path so this script can be executed directly.
THIS_DIR = Path(__file__).resolve().parent
SERVER_DIR = (THIS_DIR / "..").resolve()  # server/
sys.path.insert(0, str(SERVER_DIR))

from models.linear_predictor import LinearRegressionPredictor  # noqa: E402
from models.tree_predictor import TreePredictor  # noqa: E402
from models.neural_predictor import NeuralNetworkPredictor  # noqa: E402
from data.data_loader import DataLoader  # noqa: E402
from data.feature_engineer import FeatureEngineer  # noqa: E402


# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------

DEFAULT_CUSTOM_QUERY = textwrap.dedent(
    """
    -- Enrollment Predictor Custom Data Query Template
    -- Update the WHERE clause or add joins to shape the dataset.
    -- Ensure the query returns all columns required by your chosen feature schema.
    -- Example below limits data to terms with a year of 2023 or later.

    SELECT *
    FROM section_detail_report_sbussection_detail_report_sbus
    WHERE (term / 100) >= 2023;
    """
).strip()


# ------------------------------------------------------------------------------
# Query handling
# ------------------------------------------------------------------------------

def prompt_for_custom_query(default_query: Optional[str]) -> Optional[str]:
    """
    Interactive prompt for selecting or providing a custom training SQL query.

    Options:
    - Press Enter: use the default query (if provided)
    - Type 'custom': paste your own multi-line SQL query
    - Type 'skip': do not use a custom query (DataLoader uses its default discovery)
    """
    print("\n=== Custom SQL Query Mode ===")
    print("Provide a SELECT statement that returns the rows you want to train on.")
    print("The query must include the columns required by the selected feature schema.\n")

    template_content = (default_query or "").strip()
    if template_content:
        print("Default query preview:\n")
        print(textwrap.indent(template_content, "    "))
    else:
        print("No default query available. You can still enter a custom query manually.\n")

    print("\nOptions:")
    print("  • Press Enter to use the query above (if available).")
    print("  • Type 'custom' to paste your own SQL query.")
    print("  • Type 'skip' to fall back to automatic table discovery.\n")

    choice = input("Select option [Enter/custom/skip]: ").strip().lower()

    # Use default query
    if choice in {"", "enter"}:
        if template_content:
            print("Using default query.\n")
            return template_content
        print("No default query available. Skipping custom query.\n")
        return None

    # No custom query
    if choice == "skip":
        print("Skipping custom query. Using automatic extraction.\n")
        return None

    # User pastes their own query
    if choice == "custom":
        print("\nEnter your SQL query. Submit an empty line to finish:")
        lines: list[str] = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        query = "\n".join(lines).strip()
        if not query:
            print("No query entered. Using automatic extraction.\n")
            return None

        print("Custom query captured.\n")
        return query

    print("Unrecognized option. Using automatic extraction.\n")
    return None


def load_custom_query(
    data_query_file: Optional[str],
    data_query_inline: Optional[str],
    interactive_query: bool,
) -> Optional[str]:
    """
    Resolve which custom SQL query to use based on CLI flags.

    Precedence:
      1) --data-query-file
      2) --data-query
      3) --interactive-query
      4) None (automatic extraction)
    """
    if data_query_file:
        query_path = Path(data_query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found at: {query_path}")
        content = query_path.read_text(encoding="utf-8").strip()
        return content or None

    if data_query_inline:
        content = data_query_inline.strip()
        return content or None

    if interactive_query:
        return prompt_for_custom_query(DEFAULT_CUSTOM_QUERY)

    return None


# ------------------------------------------------------------------------------
# Model creation / saving helpers
# ------------------------------------------------------------------------------

ModelType = Literal["linear", "tree", "neural"]
FeatureSchema = Literal["min", "rich"]


def build_predictor(model_type: ModelType, feature_schema: FeatureSchema, custom_query: Optional[str]):
    """
    Factory to construct the appropriate predictor.

    Note:
    - Predictors accept (feature_schema, custom_query=...) per your existing API.
    """
    if model_type == "linear":
        return LinearRegressionPredictor(feature_schema, custom_query=custom_query)
    if model_type == "tree":
        return TreePredictor(feature_schema, custom_query=custom_query)
    if model_type == "neural":
        return NeuralNetworkPredictor(feature_schema, custom_query=custom_query)

    # Defensive: argparse should prevent this.
    raise ValueError(f"Unknown model type: {model_type}")


def resolve_model_dir() -> Path:
    """
    Determine where to save trained models.

    Preference:
    1) Container path used elsewhere in your app: /app/data/prediction_models
    2) Fallback to repo-relative: <repo>/data/prediction_models
    """
    container_dir = Path("/app/data/prediction_models")
    if container_dir.exists():
        return container_dir

    # train_model.py is server/ml/train_model.py -> go up to repo root-ish
    return (Path(__file__).resolve().parent.parent.parent / "data" / "prediction_models").resolve()


def maybe_report_trimmed_mape(results: dict) -> None:
    """
    Print average per-course MAPE excluding the bottom 10 (worst) courses, if present.

    This reporting is best-effort and should never fail training.
    """
    try:
        per_course = results.get("per_course_accuracy")
        if per_course is None or not hasattr(per_course, "shape") or per_course.shape[0] == 0:
            return

        total_courses = len(per_course)

        # Assumes per_course is sorted ascending by mape (best -> worst)
        if total_courses > 10:
            trimmed = per_course.iloc[:-10]
        else:
            trimmed = per_course

        trimmed_count = len(trimmed)
        mean_mape = float(trimmed["mape"].dropna().mean()) if trimmed_count > 0 else float("nan")

        print(
            f"\nAverage per-course MAPE (excluding bottom 10): {mean_mape:.2f}%"
            f"\n  Based on {trimmed_count} courses out of {total_courses} total"
        )
    except Exception:
        # Reporting should not affect training flow.
        return


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train enrollment prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Notes:
              - Use --data-query / --data-query-file / --interactive-query to control
                what rows are used for training.
              - If no custom query is supplied, DataLoader will use its default discovery.

            Examples:
              python server/ml/train_model.py --model tree --features min
              python server/ml/train_model.py --model tree --features rich --data-query-file ./train.sql
              python server/ml/train_model.py --model neural --interactive-query
            """
        ).strip(),
    )

    parser.add_argument(
        "--model",
        choices=["linear", "tree", "neural"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--features",
        choices=["min", "rich"],
        default="min",
        help="Feature schema to use (default: min)",
    )

    # Custom query options
    parser.add_argument("--data-query-file", type=str, help="Path to a SQL file for custom data retrieval")
    parser.add_argument("--data-query", type=str, help="Inline SQL query to retrieve training data")
    parser.add_argument("--interactive-query", action="store_true", help="Interactively preview and edit SQL query")

    return parser


def main() -> int:
    """
    Main training routine.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = build_parser()
    args = parser.parse_args()

    model_type: ModelType = args.model
    feature_schema: FeatureSchema = args.features

    print(f"=== Training {model_type} model with {feature_schema} features ===")

    # Resolve custom query (optional)
    try:
        custom_query = load_custom_query(args.data_query_file, args.data_query, args.interactive_query)
    except Exception as e:
        print(f"✗ Failed to load custom query: {e}")
        return 1

    # Construct predictor
    predictor = build_predictor(model_type, feature_schema, custom_query=custom_query)

    try:
        # ------------------------------------------------------------------
        # 1) Load data
        # ------------------------------------------------------------------
        print("\n=== Loading Data ===")
        data_loader = DataLoader(custom_query)
        raw_data = data_loader.extract_training_data()

        # ------------------------------------------------------------------
        # 2) Prepare features and target
        # ------------------------------------------------------------------
        print("\n=== Preparing Features ===")
        feature_engineer = FeatureEngineer(feature_schema)
        X, y = feature_engineer.prepare_features(raw_data)

        # Ensure the predictor is aware of the columns used during training.
        predictor.feature_columns = feature_engineer.feature_columns

        # ------------------------------------------------------------------
        # 3) Train
        # ------------------------------------------------------------------
        print("\n=== Training Model ===")
        results = predictor.train(X, y)
        print("\n=== Training Complete ===")

        # Best-effort reporting
        maybe_report_trimmed_mape(results)

        # ------------------------------------------------------------------
        # 4) Save (optional)
        # ------------------------------------------------------------------
        save_model = input("\nWould you like to save this model? (y/n): ").strip().lower()
        if save_model not in {"y", "yes"}:
            print("\nModel was not saved.")
            return 0

        model_dir = resolve_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"enrollment_{model_type}_{feature_schema}_{timestamp}.pkl"
        model_path = model_dir / model_filename

        predictor.save_model(str(model_path))
        print(f"\n✓ Model saved: {model_path}")
        return 0

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
