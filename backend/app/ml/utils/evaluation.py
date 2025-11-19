"""Model evaluation utilities."""
from typing import Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def analyze_per_course_accuracy(
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    y_pred: np.ndarray,
    model_type: str
) -> Optional[pd.DataFrame]:
    """Analyze prediction accuracy per course (subj+crse combination).
    
    Args:
        X_val: Original unprocessed validation features DataFrame
        y_val: Validation target values
        y_pred: Predicted values
        model_type: Model type name for saving results
    
    Returns:
        DataFrame with per-course accuracy statistics, or None if subj/crse not in features
    """
    # Check if we have subj and crse in our features
    has_subj = any('subj' in col.lower() for col in X_val.columns)
    has_crse = any('crse' in col.lower() for col in X_val.columns)
    
    if not (has_subj and has_crse):
        print("\nCannot analyze per-course accuracy: subj or crse not in features")
        return None
    
    # Find the actual column names
    subj_col = next((col for col in X_val.columns if 'subj' in col.lower()), None)
    crse_col = next((col for col in X_val.columns if 'crse' in col.lower()), None)
    
    if not subj_col or not crse_col:
        return None
    
    # Reset index to ensure alignment
    X_val_reset = X_val.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    
    # Create a dataframe with predictions and actuals
    analysis_df = pd.DataFrame({
        'subj': X_val_reset[subj_col],
        'crse': X_val_reset[crse_col],
        'actual': y_val_reset.values,
        'predicted': y_pred
    })
    
    # Calculate per-course metrics
    analysis_df['abs_error'] = np.abs(analysis_df['actual'] - analysis_df['predicted'])
    analysis_df['pct_error'] = np.where(
        analysis_df['actual'] > 0,
        (analysis_df['abs_error'] / analysis_df['actual']) * 100,
        np.nan
    )
    
    # Group by course
    course_stats = analysis_df.groupby(['subj', 'crse']).agg({
        'actual': ['mean', 'count'],
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).round(2)
    
    course_stats.columns = ['avg_enrollment', 'predictions_count', 'mae', 'mape']
    course_stats = course_stats.sort_values('mape')
    
    # Print report
    _print_accuracy_report(course_stats)
    
    # Save to CSV
    _save_accuracy_report(course_stats, model_type)
    
    return course_stats


def _print_accuracy_report(course_stats: pd.DataFrame):
    """Print per-course accuracy report."""
    print("\n" + "="*80)
    print("PER-COURSE PREDICTION ACCURACY REPORT")
    print("="*80)
    print(f"Total courses in validation set: {len(course_stats)}")
    print(f"\nTop 10 Most Predictable Courses (Lowest MAPE):")
    print("-"*80)
    print(f"{'Course':<12} {'Avg Enroll':<12} {'Predictions':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*80)
    
    for (subj, crse), row in course_stats.head(10).iterrows():
        course_name = f"{subj} {crse}"
        print(f"{course_name:<12} {row['avg_enrollment']:<12.1f} {int(row['predictions_count']):<12} "
              f"{row['mae']:<12.2f} {row['mape']:<12.1f}%")
    
    print(f"\nBottom 10 Least Predictable Courses (Highest MAPE):")
    print("-"*80)
    print(f"{'Course':<12} {'Avg Enroll':<12} {'Predictions':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*80)
    
    for (subj, crse), row in course_stats.tail(10).iterrows():
        course_name = f"{subj} {crse}"
        print(f"{course_name:<12} {row['avg_enrollment']:<12.1f} {int(row['predictions_count']):<12} "
              f"{row['mae']:<12.2f} {row['mape']:<12.1f}%")
    
    print("\n" + "="*80)
    
    # Summary statistics
    print(f"\nPrediction Accuracy Summary:")
    print(f"  Courses with MAPE < 20% (Good):        {(course_stats['mape'] < 20).sum()} courses")
    print(f"  Courses with MAPE 20-40% (Moderate):   {((course_stats['mape'] >= 20) & (course_stats['mape'] < 40)).sum()} courses")
    print(f"  Courses with MAPE > 40% (Poor):        {(course_stats['mape'] >= 40).sum()} courses")
    print("="*80 + "\n")


def _save_accuracy_report(course_stats: pd.DataFrame, model_type: str):
    """Save accuracy report to CSV."""
    try:
        # Use a path relative to the script location (works in Docker)
        output_dir = Path(__file__).parent.parent / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"per_course_accuracy_{model_type}_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        
        # Prepare DataFrame for export with better column names
        export_df = course_stats.reset_index()
        export_df.columns = ['Subject', 'Course', 'Avg_Enrollment', 'Num_Predictions', 'MAE', 'MAPE']
        export_df['Rank'] = range(1, len(export_df) + 1)
        
        # Reorder columns
        export_df = export_df[['Rank', 'Subject', 'Course', 'Avg_Enrollment', 'Num_Predictions', 'MAE', 'MAPE']]
        
        # Save to CSV
        export_df.to_csv(csv_path, index=False)
        print(f"\n✓ Per-course accuracy report saved to: {csv_path}")
        print(f"  Total courses ranked: {len(export_df)}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: Could not save CSV file: {e}")
        import traceback
        traceback.print_exc()
