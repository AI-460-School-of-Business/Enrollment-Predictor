"""
Command-line model testing tool

Test a trained enrollment prediction model interactively or with SQL queries.

Usage examples:
  # Predict specific course for FUTURE term (finds historical sections and predicts)
  docker compose exec web python server/ml/test_model.py --course AC 300 202501

  # Predict specific course with custom model
  docker compose exec web python server/ml/test_model.py --course BUS 301 202501 --model data/prediction_models/enrollment_tree_min_20251202_234630.pkl

  # Predict all courses for a subject in a future term
  docker compose exec web python server/ml/test_model.py --subject BUS --term 202501

  # Test against HISTORICAL data (has actual enrollments)
  docker compose exec web python server/ml/test_model.py --course AC 300 202410

  # Interactive mode - input course details manually
  docker compose exec web python server/ml/test_model.py --interactive

  # SQL mode - predict for courses from custom database query
  docker compose exec web python server/ml/test_model.py --sql "SELECT * FROM section_detail_report_sbussection_detail_report_sbus WHERE term = 202410 LIMIT 5"

  # Batch mode - test multiple courses at once
  docker compose exec web python server/ml/test_model.py --batch

Term codes: YYYYSS where YYYY=year, SS=01 (Fall) or 04 (Spring)
Examples: 202501=Fall 2025, 202504=Spring 2026, 202410=Fall 2024
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import json

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predictor_service import load_saved_model, prepare_features, predict_with_model, run_sql_query


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("  ENROLLMENT PREDICTION MODEL TESTER")
    print("="*70 + "\n")


def print_model_info(model_path: str, features: str):
    """Print information about the loaded model."""
    print(f"Model: {model_path}")
    print(f"Feature Schema: {features}")
    
    try:
        model, feature_cols, scaler, label_encoders = load_saved_model(model_path)
        print(f"Model Type: {type(model).__name__}")
        if feature_cols:
            print(f"Features ({len(feature_cols)}): {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
        if label_encoders:
            print(f"Label Encoders: {', '.join(label_encoders.keys())}")
        print()
    except Exception as e:
        print(f"Error loading model: {e}\n")
        sys.exit(1)


def interactive_single_course(model_path: str, features: str):
    """Interactively input single course details."""
    print("=== Interactive Single Course Prediction ===\n")
    print("Enter course details (press Enter for defaults shown in brackets):\n")
    
    # Get user input
    term = input("Term (e.g., 202501 for Fall 2025) [202501]: ").strip() or "202501"
    subj = input("Subject Code (e.g., CS, MATH, BUS) [CS]: ").strip() or "CS"
    crse = input("Course Number (e.g., 111, 250, 410) [111]: ").strip() or "111"
    credits = input("Credits [3]: ").strip() or "3"
    
    # Build the input row
    row = {
        "term": term,
        "subj": subj,
        "crse": crse,
        "credits": credits,
        "act": "0"
    }
    
    print("\n" + "-"*70)
    print("Input Course Data:")
    print(json.dumps(row, indent=2))
    print("-"*70 + "\n")
    
    # Make prediction
    try:
        result = predict_with_model(model_path, [row], features)
        
        if result['predictions']:
            print("=== PREDICTION RESULT ===\n")
            print(f"Predicted Enrollment: {result['predictions'][0]:.1f} students")
        else:
            print("No prediction returned.")
        
        print()
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def interactive_batch_courses(model_path: str, features: str):
    """Interactively input multiple courses and predict enrollments."""
    print("=== Batch Course Prediction ===\n")
    print("Enter multiple courses. For each course, provide comma-separated values:")
    print("Format: term,subj,crse,credits")
    print("Example: 202501,CS,111,3")
    print("\nPress Enter with empty line when done.\n")
    
    rows = []
    course_num = 1
    
    while True:
        line = input(f"Course {course_num} (or Enter to finish): ").strip()
        if not line:
            break
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            print("  ✗ Invalid format. Need at least: term,subj,crse")
            continue
        
        row = {
            "term": parts[0],
            "subj": parts[1],
            "crse": parts[2],
            "credits": parts[3] if len(parts) > 3 else "3",
            "act": "0"
        }
        
        rows.append(row)
        print(f"  ✓ Added: {row['subj']} {row['crse']}")
        course_num += 1
    
    if not rows:
        print("No courses entered.\n")
        return
    
    print(f"\n{len(rows)} courses entered. Making predictions...\n")
    
    # Make predictions
    try:
        result = predict_with_model(model_path, rows, features)
        
        print("=== PREDICTION RESULTS ===\n")
        print(f"{'Course':<15} {'Predicted Enrollment':<25}")
        print("-"*50)
        
        # Note: result['predictions'] might be shorter than rows if aggregation happened
        # But since we are inputting unique courses, it should be 1-to-1
        
        preds = result['predictions']
        
        if len(preds) == len(rows):
            for i, (row, pred) in enumerate(zip(rows, preds)):
                course_id = f"{row['subj']} {row['crse']}"
                print(f"{course_id:<15} {pred:>20.1f} students")
        else:
            # If aggregation happened, we need to match by features
            # But for now, let's just print what we have
            print(f"Returned {len(preds)} predictions for {len(rows)} inputs (aggregation occurred).")
            for pred in preds:
                print(f"Prediction: {pred:.1f}")
        
        print()
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def sql_mode(model_path: str, features: str, sql: str):
    """Run SQL query and predict for returned courses."""
    print("=== SQL Query Mode ===\n")
    print(f"Executing SQL:\n{sql}\n")
    
    try:
        # Execute query
        df = run_sql_query(sql)
        print(f"Query returned {len(df)} rows\n")
        
        if len(df) == 0:
            print("No data returned from query.\n")
            return
        
        # Convert to dict rows
        rows = df.to_dict(orient='records')
        
        # Make predictions
        result = predict_with_model(model_path, rows, features)
        preds = result['predictions']
        
        print("=== PREDICTION RESULTS (Aggregated by Course) ===\n")
        
        if 'features' in result:
            # We have the features used for prediction (raw values before encoding)
            feat_rows = result['features']
            print(f"{'Term':<10} {'Course':<15} {'Predicted':<12}")
            print("-"*40)
            
            for feat, pred in zip(feat_rows, preds):
                term = feat.get('term', 'N/A')
                subj = feat.get('subj', '?')
                crse = feat.get('crse', '?')
                course_id = f"{subj} {crse}"
                
                print(f"{str(term):<10} {course_id:<15} {pred:>10.1f}")
        else:
            # Fallback
            print(f"Generated {len(preds)} predictions.")
            print(preds)
            
        print()
        
    except Exception as e:
        print(f"\n✗ SQL prediction failed: {e}")
        import traceback
        traceback.print_exc()


def predict_course(model_path: str, features: str, subj: str, crse: str, term: str):
    """Predict enrollment for a specific course (future or historical)."""
    print(f"=== Predicting Enrollment for {subj} {crse} in Term {term} ===\n")
    
    # Try to get credits from DB if possible, else default to 3
    credits = 3
    try:
        sql = f"SELECT credits FROM section_detail_report_sbussection_detail_report_sbus WHERE subj='{subj}' AND crse='{crse}' LIMIT 1"
        df = run_sql_query(sql)
        if not df.empty and 'credits' in df.columns:
            val = df.iloc[0]['credits']
            if val:
                credits = val
    except:
        pass
        
    row = {
        'term': int(term),
        'subj': subj,
        'crse': crse,
        'credits': credits,
        'act': 0 # Placeholder
    }
    
    print(f"Input: {json.dumps(row, indent=2)}")
    
    try:
        # Make prediction
        result = predict_with_model(model_path, [row], features)
        
        if result['predictions']:
            pred = result['predictions'][0]
            print("\n=== PREDICTION RESULT ===\n")
            print(f"Course: {subj} {crse}")
            print(f"Term: {term}")
            print(f"Credits: {credits}")
            print(f"Predicted Total Enrollment: {pred:.1f}")
            
            # Check for actuals if historical
            try:
                check_sql = f"""
                    SELECT SUM(act) as total_act 
                    FROM section_detail_report_sbussection_detail_report_sbus 
                    WHERE subj='{subj}' AND crse='{crse}' AND term={term}
                """
                check_df = run_sql_query(check_sql)
                if not check_df.empty and check_df.iloc[0]['total_act'] is not None:
                    actual = float(check_df.iloc[0]['total_act'])
                    if actual > 0:
                        error = abs(pred - actual)
                        pct = (error / actual * 100)
                        print(f"Actual Enrollment: {actual:.0f}")
                        print(f"Error: {error:.1f} ({pct:.1f}%)")
            except:
                pass
        else:
            print("No prediction returned.")
            
        print()
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def predict_subject_term(model_path: str, features: str, subj: str, term: str):
    """Predict enrollment for all courses in a subject for a specific term."""
    print(f"=== Predicting Enrollment for All {subj} Courses in Term {term} ===\n")
    
    # Query database for all courses in this subject for the specified term
    sql = f"""
        SELECT * 
        FROM section_detail_report_sbussection_detail_report_sbus 
        WHERE subj = '{subj}' 
          AND term = {term}
        ORDER BY crse, sec
    """
    
    print(f"Querying database for all {subj} courses in term {term}...\n")
    sql_mode(model_path, features, sql)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test enrollment prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", type=str,
                       help="Path to model file (.pkl)")
    parser.add_argument("--features", choices=["min", "rich"], default="min",
                       help="Feature schema to use (default: min)")
    parser.add_argument("--course", nargs=3, metavar=('SUBJ', 'CRSE', 'TERM'),
                       help="Predict for specific course: SUBJ CRSE TERM (e.g., BUS 301 202501)")
    parser.add_argument("--subject", type=str,
                       help="Predict for all courses in a subject (use with --term)")
    parser.add_argument("--term", type=str,
                       help="Term to predict for (use with --subject)")
    parser.add_argument("--sql", type=str,
                       help="SQL query to fetch courses for prediction")
    parser.add_argument("--batch", action="store_true",
                       help="Batch mode - enter multiple courses")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode - manually input course details")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Use default model
        model_path = "/app/data/prediction_models/enrollment_tree_min_20251202_234630.pkl"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"✗ Model file not found: {model_path}")
        print("\nAvailable models in /app/data/prediction_models/:")
        model_dir = Path("/app/data/prediction_models")
        if model_dir.exists():
            for f in sorted(model_dir.glob("*.pkl")):
                print(f"  - {f.name}")
        sys.exit(1)
    
    print_banner()
    print_model_info(model_path, args.features)
    
    # Choose mode based on arguments
    if args.course:
        # Predict specific course
        subj, crse, term = args.course
        predict_course(model_path, args.features, subj, crse, term)
    elif args.subject and args.term:
        # Predict all courses in subject for a term
        predict_subject_term(model_path, args.features, args.subject, args.term)
    elif args.subject or args.term:
        print("✗ Error: --subject and --term must be used together\n")
        sys.exit(1)
    elif args.sql:
        sql_mode(model_path, args.features, args.sql)
    elif args.batch:
        interactive_batch_courses(model_path, args.features)
    elif args.interactive:
        interactive_single_course(model_path, args.features)
    else:
        # Default to interactive mode if no arguments provided
        interactive_single_course(model_path, args.features)
    
    print("="*70)
    print("Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
