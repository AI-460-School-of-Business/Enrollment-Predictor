"""
Enrollment Prediction Model Training CLI

Trains models to predict enrollment counts using Subject+Course, Semester, Year data.
Course identity is determined by the combination of subj (subject) and crse (course number).

Usage: python server/ml/train_model.py --model [linear|tree|neural] --features [min|rich]
"""

import argparse
import sys
import os
import textwrap
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.linear_predictor import LinearRegressionPredictor
from models.tree_predictor import TreePredictor
from models.neural_predictor import NeuralNetworkPredictor
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer

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


def prompt_for_custom_query(default_query: Optional[str]) -> Optional[str]:
    """Interactively prompt the user for a custom SQL query."""
    print("\n=== Custom SQL Query Mode ===")
    print("Provide a SELECT statement that returns the rows you want to train on.")
    print("The query must include the columns required by the selected feature schema.\n")

    template_content = default_query.strip() if default_query else ""
    if template_content:
        print("Default query preview:\n")
        preview = textwrap.indent(template_content, "    ")
        print(preview)
    else:
        print("No default query available. You can still enter a custom query manually.\n")
    
    print("\nOptions:")
    print("  • Press Enter to use the query above (if available).")
    print("  • Type 'custom' to paste your own SQL query.")
    print("  • Type 'skip' to fall back to automatic table discovery.\n")
    
    choice = input("Select option [Enter/custom/skip]: ").strip().lower()
    if choice in {"", "enter"}:
        if template_content:
            print("Using default query.\n")
            return template_content
        print("No default query available. Skipping custom query.\n")
        return None
    if choice == "skip":
        print("Skipping custom query. Using automatic extraction.\n")
        return None
    if choice == "custom":
        print("\nEnter your SQL query. Submit an empty line to finish:")
        lines = []
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train enrollment prediction models")
    parser.add_argument("--model", choices=["linear", "tree", "neural"], required=True,
                       help="Model type to train")
    parser.add_argument("--features", choices=["min", "rich"], default="min",
                       help="Feature schema to use")
    parser.add_argument("--data-query-file", type=str,
                       help="Path to a SQL file for custom data retrieval")
    parser.add_argument("--data-query", type=str,
                       help="Inline SQL query to retrieve training data")
    parser.add_argument("--interactive-query", action="store_true",
                       help="Interactively preview and edit SQL query")
    
    args = parser.parse_args()
    
    print(f"=== Training {args.model} model with {args.features} features ===")

    # Handle custom query
    custom_query: Optional[str] = None
    if args.data_query_file:
        query_path = Path(args.data_query_file)
        if not query_path.exists():
            print(f"Error: query file not found at {query_path}")
            sys.exit(1)
        custom_query = query_path.read_text().strip() or None
    elif args.data_query:
        custom_query = args.data_query.strip() or None
    elif args.interactive_query:
        custom_query = prompt_for_custom_query(DEFAULT_CUSTOM_QUERY)
    
    # Create predictor
    if args.model == "linear":
        predictor = LinearRegressionPredictor(args.features, custom_query=custom_query)
    elif args.model == "tree":
        predictor = TreePredictor(args.features, custom_query=custom_query)
    elif args.model == "neural":
        predictor = NeuralNetworkPredictor(args.features, custom_query=custom_query)
    else:
        print(f"Unknown model type: {args.model}")
        sys.exit(1)
    
    try:
        # Load data
        print("\n=== Loading Data ===")
        data_loader = DataLoader(custom_query)
        raw_data = data_loader.extract_training_data()
        
        # Prepare features
        print("\n=== Preparing Features ===")
        feature_engineer = FeatureEngineer(args.features)
        X, y = feature_engineer.prepare_features(raw_data)
        
        # Store feature columns in predictor
        predictor.feature_columns = feature_engineer.feature_columns
        
        # Train model
        print("\n=== Training Model ===")
        results = predictor.train(X, y)
        
        print(f"\n=== Training Complete ===")

        # If per-course accuracy was computed, report average MAPE excluding bottom 10 courses
        try:
            per_course = results.get('per_course_accuracy')
            if per_course is not None and hasattr(per_course, 'shape') and per_course.shape[0] > 0:
                # per_course is expected to be a DataFrame sorted by 'mape' ascending
                total_courses = len(per_course)
                if total_courses > 10:
                    trimmed = per_course.iloc[:-10]  # remove bottom 10 (highest MAPE)
                    trimmed_count = len(trimmed)
                else:
                    trimmed = per_course
                    trimmed_count = len(trimmed)

                # Compute mean MAPE ignoring possible NaN values
                mean_mape = float(trimmed['mape'].dropna().mean()) if trimmed_count > 0 else float('nan')
                print(f"\nAverage per-course MAPE (excluding bottom 10): {mean_mape:.2f}%\n  Based on {trimmed_count} courses out of {total_courses} total")

        except Exception:
            # Don't fail training just because of this reporting step
            pass
        
        # Prompt to save
        save_model = input("\nWould you like to save this model? (y/n): ").strip().lower()
        
        if save_model in ['y', 'yes']:
            # Determine save location
            model_dir = Path("/app/data/prediction_models")
            if not model_dir.exists():
                # Fallback to local path
                model_dir = Path(__file__).parent.parent.parent / "data" / "prediction_models"
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"enrollment_{args.model}_{args.features}_{timestamp}.pkl"
            model_path = model_dir / model_filename
            
            predictor.save_model(str(model_path))
            print(f"\n✓ Model saved: {model_path}")
        else:
            print("\nModel was not saved.")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
