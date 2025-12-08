import argparse
import sys
import os
import pandas as pd
from pathlib import Path
import textwrap

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.predictor_service import predict_with_model, run_sql_query, _choose_model_path

def main():
    parser = argparse.ArgumentParser(description="Generate enrollment predictions", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--single", action="store_true", help="Predict for a single course (requires --subj, --crse, --term)")
    input_group.add_argument("--sql", type=str, help="SQL query to select courses")
    input_group.add_argument("--sql-file", type=str, help="Path to SQL file")
    
    # Single prediction args
    parser.add_argument("--subj", type=str, help="Subject code (e.g. FIN)")
    parser.add_argument("--crse", type=str, help="Course number (e.g. 400)")
    parser.add_argument("--term", type=int, help="Term code (e.g. 202640)")
    parser.add_argument("--credits", type=int, default=3, help="Credits (default: 3)")
    
    # Model args
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--features", type=str, default="min", choices=["min", "rich"], help="Feature schema")
    
    parser.epilog = textwrap.dedent("""
    Examples:
      # Predict for a single course
      python backend/app/ml/predict.py --single --subj FIN --crse 400 --term 202640
      
      # Predict for courses using a SQL query
      docker-compose exec web python backend/app/ml/predict.py --sql "SELECT term, subj, crse FROM section_detail_report_sbussection_detail_report_sbus WHERE subj='MIS' AND term=202540"
    """)
    
    args = parser.parse_args()
    
    # Resolve model path
    model_dir = "/backend/data/prediction_models"
    if args.model:
        model_path = args.model
    else:
        # Find latest model
        try:
            payload = {"model_dir": model_dir}
            model_path = _choose_model_path(payload)
            print(f"Using latest model: {model_path}")
        except Exception as e:
            print(f"Error finding model: {e}")
            sys.exit(1)
            
    rows = []
    
    if args.single:
        if not (args.subj and args.crse and args.term):
            print("Error: --single requires --subj, --crse, and --term")
            sys.exit(1)
            
        rows = [{
            "term": args.term,
            "subj": args.subj,
            "crse": args.crse,
            "credits": args.credits,
            "act": 0 # Dummy for feature engineer
        }]
        print(f"Predicting for single course: {args.subj} {args.crse} Term {args.term}")
        
    elif args.sql or args.sql_file:
        query = args.sql
        if args.sql_file:
            with open(args.sql_file, 'r') as f:
                query = f.read()
        
        print("Executing SQL query...")
        try:
            df = run_sql_query(query)
            
            # Normalize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure 'act' column exists for FeatureEngineer
            if 'act' not in df.columns:
                df['act'] = 0
            
            # Ensure 'credits' column exists if not selected
            if 'credits' not in df.columns:
                df['credits'] = 3
            
            rows = df.to_dict(orient='records')
            print(f"Found {len(rows)} rows to predict.")
        except Exception as e:
            print(f"Error executing SQL: {e}")
            sys.exit(1)

    if not rows:
        print("No data to predict.")
        sys.exit(0)

    try:
        result = predict_with_model(model_path, rows, features=args.features)
        predictions = result['predictions']
        feature_rows = result['features'] # These are the aggregated rows used for prediction
        
        # Display results
        print("\n" + "="*60)
        print(f"{'Subject':<8} {'Course':<8} {'Term':<8} {'Prediction':<12}")
        print("-" * 60)
        
        for i, row in enumerate(feature_rows):
            pred = predictions[i]
            subj = row.get('subj', 'N/A')
            crse = row.get('crse', 'N/A')
            term = row.get('term', 'N/A')
            print(f"{subj:<8} {crse:<8} {term:<8} {pred:<12.2f}")
            
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
