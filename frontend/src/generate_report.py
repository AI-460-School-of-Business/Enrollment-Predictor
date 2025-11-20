"""
Enrollment Prediction Report Generator

Generates comprehensive Excel reports with:
1. ML Model Overview & Synopsis
2. Visualization Graphs
3. Enrollment Predictions Table
"""

import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import os
import sys

# Database
try:
    import psycopg2
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}


def get_db_connection():
    """Create database connection."""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        sys.exit(1)


def get_course_identifiers(custom_query: Optional[str] = None) -> pd.DataFrame:
    """
    Get course identifiers from the database.
    
    Args:
        custom_query: Optional SQL query to get specific courses.
                     If None, queries all unique course identifiers from the main table.
    
    Returns:
        DataFrame with columns: subj, crse, and other course details
    """
    conn = get_db_connection()
    
    try:
        if custom_query:
            print(f"Fetching course identifiers with custom query...")
            courses_df = pd.read_sql(custom_query, conn)
        else:
            # Auto-discover the main table
            print("Auto-discovering course identifiers from database...")
            
            # Find the main enrollment table
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """
            tables_df = pd.read_sql(tables_query, conn)
            available_tables = tables_df['table_name'].tolist()
            
            # Look for the section detail report table (our main table)
            main_table = None
            for table in available_tables:
                if 'section' in table.lower() and 'detail' in table.lower():
                    main_table = table
                    break
            
            if not main_table:
                # Fallback to first table
                main_table = available_tables[0] if available_tables else None
            
            if not main_table:
                raise Exception("No tables found in database")
            
            print(f"Using table: {main_table}")
            
            # Get unique course identifiers
            # Query for distinct subj+crse combinations with latest term info
            query = f"""
                SELECT DISTINCT
                    subj,
                    crse,
                    title,
                    credits
                FROM {main_table}
                WHERE subj IS NOT NULL 
                  AND crse IS NOT NULL
                ORDER BY subj, crse;
            """
            
            courses_df = pd.read_sql(query, conn)
        
        print(f"Found {len(courses_df)} unique courses")
        print(f"Columns: {list(courses_df.columns)}")
        
        # Verify we have subj and crse
        has_subj = any('subj' in col.lower() for col in courses_df.columns)
        has_crse = any('crse' in col.lower() for col in courses_df.columns)
        
        if not (has_subj and has_crse):
            raise Exception("Query result must include 'subj' and 'crse' columns")
        
        return courses_df
        
    finally:
        conn.close()


def run_predictions(model_path: str, courses_df: pd.DataFrame, target_term: int) -> pd.DataFrame:
    """
    Run enrollment predictions for the given courses.
    
    Args:
        model_path: Path to the trained model pickle file
        courses_df: DataFrame with course identifiers (subj, crse, etc.)
        target_term: The term to predict for (e.g., 202501 for Spring 2025)
    
    Returns:
        DataFrame with predictions including: subj, crse, term, predicted_enrollment
    """
    print(f"\nLoading model from {model_path}...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    label_encoders = model_data.get('label_encoders', {})
    feature_columns = model_data['feature_columns']
    model_type = model_data.get('model_type', 'unknown')
    
    print(f"Model type: {model_type}")
    print(f"Required features: {feature_columns}")
    
    # Create prediction dataset
    # For each course, create entries for different sections (typically 1-5)
    prediction_records = []
    
    for _, course in courses_df.iterrows():
        # Create predictions for sections 1-3 (common section numbers)
        for section in [1, 2, 3]:
            record = {
                'term': target_term,
                'subj': course['subj'],
                'crse': course['crse'],
                'sec': section,
                'credits': course.get('credits', 3.0)  # Default to 3 credits if not available
            }
            prediction_records.append(record)
    
    X_pred = pd.DataFrame(prediction_records)
    print(f"\nCreated {len(X_pred)} prediction records")
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in X_pred.columns:
            print(f"Warning: Feature '{col}' not in prediction data, adding with default value 0")
            X_pred[col] = 0
    
    # Select only the features used by the model
    X_pred_features = X_pred[feature_columns].copy()
    
    # Preprocess features (same as in training)
    # Handle categorical variables
    categorical_columns = X_pred_features.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col in label_encoders:
            # Use existing encoder
            le = label_encoders[col]
            # Handle unknown categories
            X_pred_features[col] = X_pred_features[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            # Simple encoding for new categorical columns
            X_pred_features[col] = pd.factorize(X_pred_features[col])[0]
    
    # Scale numerical features
    if scaler:
        X_pred_scaled = scaler.transform(X_pred_features)
    else:
        X_pred_scaled = X_pred_features.values
    
    # Make predictions
    print("Running predictions...")
    predictions = model.predict(X_pred_scaled)
    
    # Round predictions to nearest integer (can't have fractional students)
    predictions = np.round(predictions).astype(int)
    
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    
    # Add predictions to the dataframe
    X_pred['Predicted_Enrollment'] = predictions
    
    # Rename columns for report
    X_pred = X_pred.rename(columns={
        'subj': 'Subject',
        'crse': 'Course',
        'sec': 'Section',
        'term': 'Term',
        'credits': 'Credits'
    })
    
    print(f"Predictions complete!")
    print(f"Total predicted enrollment: {predictions.sum()}")
    print(f"Average predicted enrollment: {predictions.mean():.1f}")
    
    return X_pred


class EnrollmentReportGenerator:
    """Generate comprehensive Excel reports for enrollment predictions."""
    
    def __init__(self, model_path: str, predictions_df: pd.DataFrame, accuracy_df: Optional[pd.DataFrame] = None):
        """
        Initialize the report generator.
        
        Args:
            model_path: Path to the trained model pickle file
            predictions_df: DataFrame with enrollment predictions
            accuracy_df: Optional DataFrame with per-course accuracy metrics
        """
        self.model_path = model_path
        self.predictions_df = predictions_df
        self.accuracy_df = accuracy_df
        self.model_metadata = self._load_model_metadata()
        
        # Excel styling
        self.header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        self.header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        self.title_font = Font(name="Calibri", size=16, bold=True, color="366092")
        self.subtitle_font = Font(name="Calibri", size=12, bold=True, color="44546A")
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
    def _load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata from pickle file."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            return {
                'model_type': model_data.get('model_type', 'Unknown'),
                'feature_schema': model_data.get('feature_schema', 'Unknown'),
                'feature_columns': model_data.get('feature_columns', []),
                'model_file': Path(self.model_path).name
            }
        except Exception as e:
            print(f"Warning: Could not load model metadata: {e}")
            return {
                'model_type': 'Unknown',
                'feature_schema': 'Unknown',
                'feature_columns': [],
                'model_file': Path(self.model_path).name
            }
    
    def generate_report(self, output_path: str):
        """Generate the complete Excel report."""
        print(f"Generating enrollment prediction report...")
        
        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # Remove default sheet
        
        # Create sheets
        print("  Creating Overview sheet...")
        self._create_overview_sheet(wb)
        
        print("  Creating Visualizations sheet...")
        self._create_visualizations_sheet(wb)
        
        print("  Creating Predictions sheet...")
        self._create_predictions_sheet(wb)
        
        if self.accuracy_df is not None:
            print("  Creating Accuracy Analysis sheet...")
            self._create_accuracy_sheet(wb)
        
        # Save workbook
        wb.save(output_path)
        print(f"✓ Report saved to: {output_path}")
        
    def _create_overview_sheet(self, wb: openpyxl.Workbook):
        """Create the overview and synopsis sheet."""
        ws = wb.create_sheet("Overview & Synopsis")
        
        # Title
        ws['A1'] = "Enrollment Prediction Report"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:D1')
        
        # Report metadata
        ws['A3'] = "Report Generated:"
        ws['B3'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        ws['A4'] = "Model Type:"
        ws['B4'] = self.model_metadata['model_type'].title()
        ws['A5'] = "Feature Schema:"
        ws['B5'] = self.model_metadata['feature_schema'].upper()
        ws['A6'] = "Model File:"
        ws['B6'] = self.model_metadata['model_file']
        
        # Make labels bold
        for row in range(3, 7):
            ws[f'A{row}'].font = Font(bold=True)
        
        # Model Synopsis Section
        ws['A8'] = "Model Synopsis"
        ws['A8'].font = self.subtitle_font
        ws.merge_cells('A8:D8')
        
        synopsis_text = self._generate_synopsis()
        row = 10
        for line in synopsis_text.split('\n'):
            ws[f'A{row}'] = line
            ws.merge_cells(f'A{row}:D{row}')
            ws[f'A{row}'].alignment = Alignment(wrap_text=True, vertical='top')
            row += 1
        
        # Key Metrics Section
        if self.accuracy_df is not None:
            ws[f'A{row + 1}'] = "Key Performance Metrics"
            ws[f'A{row + 1}'].font = self.subtitle_font
            ws.merge_cells(f'A{row + 1}:D{row + 1}')
            
            metrics_row = row + 3
            self._add_key_metrics(ws, metrics_row)
        
        # Prediction Summary
        summary_row = row + 10
        ws[f'A{summary_row}'] = "Prediction Summary"
        ws[f'A{summary_row}'].font = self.subtitle_font
        ws.merge_cells(f'A{summary_row}:D{summary_row}')
        
        summary_data = self._get_prediction_summary()
        summary_row += 2
        for key, value in summary_data.items():
            ws[f'A{summary_row}'] = key
            ws[f'B{summary_row}'] = value
            ws[f'A{summary_row}'].font = Font(bold=True)
            summary_row += 1
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 40
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 20
        
    def _generate_synopsis(self) -> str:
        """Generate a text synopsis of the model."""
        model_type = self.model_metadata['model_type']
        features = len(self.model_metadata['feature_columns'])
        
        synopsis = f"""This report presents enrollment predictions generated using a {model_type} machine learning model. """
        synopsis += f"""The model was trained on historical enrollment data using {features} key features including course identifiers, """
        synopsis += f"""term information, section details, and credit hours.\n\n"""
        
        if self.accuracy_df is not None:
            good_courses = (self.accuracy_df['MAPE'] < 20).sum()
            total_courses = len(self.accuracy_df)
            pct_good = (good_courses / total_courses * 100) if total_courses > 0 else 0
            
            synopsis += f"""Model Performance: The model demonstrates excellent predictive accuracy for {good_courses} out of {total_courses} courses """
            synopsis += f"""({pct_good:.1f}%), with a Mean Absolute Percentage Error (MAPE) below 20%. """
            synopsis += f"""These courses represent stable, predictable enrollment patterns suitable for operational planning.\n\n"""
        
        synopsis += f"""Usage Recommendations: Predictions should be used as a planning tool alongside institutional knowledge. """
        synopsis += f"""Courses with lower prediction counts or unusual patterns should be reviewed manually. """
        synopsis += f"""The model is most accurate for core required courses with consistent historical enrollment."""
        
        return synopsis
    
    def _add_key_metrics(self, ws: openpyxl.Workbook.active, start_row: int):
        """Add key performance metrics to the overview sheet."""
        if self.accuracy_df is None:
            return
        
        # Calculate metrics
        total_courses = len(self.accuracy_df)
        excellent = (self.accuracy_df['MAPE'] < 20).sum()
        good = ((self.accuracy_df['MAPE'] >= 20) & (self.accuracy_df['MAPE'] < 40)).sum()
        poor = (self.accuracy_df['MAPE'] >= 40).sum()
        avg_mape = self.accuracy_df['MAPE'].mean()
        median_mape = self.accuracy_df['MAPE'].median()
        
        # Headers
        ws[f'A{start_row}'] = "Metric"
        ws[f'B{start_row}'] = "Value"
        ws[f'A{start_row}'].font = self.header_font
        ws[f'B{start_row}'].font = self.header_font
        ws[f'A{start_row}'].fill = self.header_fill
        ws[f'B{start_row}'].fill = self.header_fill
        
        # Data
        metrics = [
            ("Total Courses Analyzed", total_courses),
            ("Excellent Predictions (MAPE < 20%)", f"{excellent} ({excellent/total_courses*100:.1f}%)"),
            ("Good Predictions (MAPE 20-40%)", f"{good} ({good/total_courses*100:.1f}%)"),
            ("Poor Predictions (MAPE > 40%)", f"{poor} ({poor/total_courses*100:.1f}%)"),
            ("Average MAPE", f"{avg_mape:.2f}%"),
            ("Median MAPE", f"{median_mape:.2f}%"),
        ]
        
        for i, (metric, value) in enumerate(metrics, start=1):
            ws[f'A{start_row + i}'] = metric
            ws[f'B{start_row + i}'] = value
            ws[f'A{start_row + i}'].border = self.border
            ws[f'B{start_row + i}'].border = self.border
    
    def _get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary statistics for predictions."""
        return {
            "Total Predictions": len(self.predictions_df),
            "Unique Courses": self.predictions_df[['Subject', 'Course']].drop_duplicates().shape[0] if 'Subject' in self.predictions_df.columns else "N/A",
            "Predicted Total Enrollment": int(self.predictions_df['Predicted_Enrollment'].sum()) if 'Predicted_Enrollment' in self.predictions_df.columns else "N/A",
            "Average Predicted Enrollment": f"{self.predictions_df['Predicted_Enrollment'].mean():.1f}" if 'Predicted_Enrollment' in self.predictions_df.columns else "N/A",
        }
    
    def _create_visualizations_sheet(self, wb: openpyxl.Workbook):
        """Create sheet with charts and visualizations."""
        ws = wb.create_sheet("Visualizations")
        
        ws['A1'] = "Enrollment Prediction Visualizations"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:F1')
        
        if self.accuracy_df is not None:
            # Chart 1: Accuracy Distribution (Pie Chart)
            self._add_accuracy_pie_chart(ws, start_row=3)
            
            # Chart 2: Top 10 Predictable Courses (Bar Chart)
            self._add_top_courses_chart(ws, start_row=20)
            
            # Chart 3: MAPE Distribution (Histogram-style)
            self._add_mape_distribution(ws, start_row=37)
    
    def _add_accuracy_pie_chart(self, ws: openpyxl.Workbook.active, start_row: int):
        """Add pie chart showing accuracy distribution."""
        ws[f'A{start_row}'] = "Model Accuracy Distribution"
        ws[f'A{start_row}'].font = self.subtitle_font
        
        # Data for pie chart
        excellent = (self.accuracy_df['MAPE'] < 20).sum()
        good = ((self.accuracy_df['MAPE'] >= 20) & (self.accuracy_df['MAPE'] < 40)).sum()
        poor = (self.accuracy_df['MAPE'] >= 40).sum()
        
        data_row = start_row + 2
        ws[f'A{data_row}'] = "Category"
        ws[f'B{data_row}'] = "Count"
        ws[f'A{data_row}'].font = Font(bold=True)
        ws[f'B{data_row}'].font = Font(bold=True)
        
        ws[f'A{data_row + 1}'] = "Excellent (< 20%)"
        ws[f'B{data_row + 1}'] = excellent
        ws[f'A{data_row + 2}'] = "Good (20-40%)"
        ws[f'B{data_row + 2}'] = good
        ws[f'A{data_row + 3}'] = "Poor (> 40%)"
        ws[f'B{data_row + 3}'] = poor
        
        # Create pie chart
        pie = PieChart()
        labels = Reference(ws, min_col=1, min_row=data_row + 1, max_row=data_row + 3)
        data = Reference(ws, min_col=2, min_row=data_row, max_row=data_row + 3)
        pie.add_data(data, titles_from_data=True)
        pie.set_categories(labels)
        pie.title = "Prediction Accuracy by Category"
        pie.height = 10
        pie.width = 15
        
        ws.add_chart(pie, f'D{start_row}')
    
    def _add_top_courses_chart(self, ws: openpyxl.Workbook.active, start_row: int):
        """Add bar chart showing top 10 most predictable courses."""
        ws[f'A{start_row}'] = "Top 10 Most Predictable Courses"
        ws[f'A{start_row}'].font = self.subtitle_font
        
        # Get top 10 courses
        top_10 = self.accuracy_df.nsmallest(10, 'MAPE')
        
        data_row = start_row + 2
        ws[f'A{data_row}'] = "Course"
        ws[f'B{data_row}'] = "MAPE %"
        ws[f'A{data_row}'].font = Font(bold=True)
        ws[f'B{data_row}'].font = Font(bold=True)
        
        for i, (idx, row) in enumerate(top_10.iterrows(), start=1):
            course_name = f"{row['Subject']} {row['Course']}"
            ws[f'A{data_row + i}'] = course_name
            ws[f'B{data_row + i}'] = round(row['MAPE'], 2)
        
        # Create bar chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Top 10 Most Predictable Courses (Lowest MAPE)"
        chart.y_axis.title = 'MAPE %'
        chart.x_axis.title = 'Course'
        
        data = Reference(ws, min_col=2, min_row=data_row, max_row=data_row + 10)
        cats = Reference(ws, min_col=1, min_row=data_row + 1, max_row=data_row + 10)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 10
        chart.width = 15
        
        ws.add_chart(chart, f'D{start_row}')
    
    def _add_mape_distribution(self, ws: openpyxl.Workbook.active, start_row: int):
        """Add histogram-style distribution of MAPE values."""
        ws[f'A{start_row}'] = "MAPE Distribution Across All Courses"
        ws[f'A{start_row}'].font = self.subtitle_font
        
        # Create bins
        bins = [0, 10, 20, 30, 40, 50, 100, 500, 3000]
        labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-100%', '100-500%', '500%+']
        
        # Filter out NaN values
        mape_values = self.accuracy_df['MAPE'].dropna()
        counts, _ = np.histogram(mape_values, bins=bins)
        
        data_row = start_row + 2
        ws[f'A{data_row}'] = "MAPE Range"
        ws[f'B{data_row}'] = "Course Count"
        ws[f'A{data_row}'].font = Font(bold=True)
        ws[f'B{data_row}'].font = Font(bold=True)
        
        for i, (label, count) in enumerate(zip(labels, counts), start=1):
            ws[f'A{data_row + i}'] = label
            ws[f'B{data_row + i}'] = int(count)
        
        # Create bar chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 11
        chart.title = "Distribution of Prediction Accuracy"
        chart.y_axis.title = 'Number of Courses'
        chart.x_axis.title = 'MAPE Range'
        
        data = Reference(ws, min_col=2, min_row=data_row, max_row=data_row + len(labels))
        cats = Reference(ws, min_col=1, min_row=data_row + 1, max_row=data_row + len(labels))
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 10
        chart.width = 15
        
        ws.add_chart(chart, f'D{start_row}')
    
    def _create_predictions_sheet(self, wb: openpyxl.Workbook):
        """Create sheet with enrollment predictions table in hierarchical format."""
        ws = wb.create_sheet("Enrollment Predictions")
        
        # Header Section
        ws['A1'] = "Enrollment Prediction Report"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:G1')
        
        row = 2
        ws[f'A{row}'] = "Report Generated:"
        ws[f'B{row}'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        row += 1
        
        ws[f'A{row}'] = "Version:"
        ws[f'B{row}'] = "0.01"
        row += 2
        
        # Model Card Section
        ws[f'A{row}'] = "Model Card"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        ws[f'A{row}'] = "Model Type:"
        ws[f'B{row}'] = self.model_metadata['model_type'].title()
        row += 1
        
        ws[f'A{row}'] = "Feature Schema:"
        ws[f'B{row}'] = self.model_metadata['feature_schema'].upper()
        row += 1
        
        ws[f'A{row}'] = "Model File:"
        ws[f'B{row}'] = self.model_metadata['model_file']
        row += 2
        
        # Prediction Query Section
        ws[f'A{row}'] = "Prediction Query"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        ws[f'A{row}'] = "Semester(s):"
        # Extract term from predictions
        if 'Term' in self.predictions_df.columns and len(self.predictions_df) > 0:
            terms = self.predictions_df['Term'].unique()
            ws[f'B{row}'] = ", ".join([str(t) for t in sorted(terms)])
        row += 1
        
        ws[f'A{row}'] = "Filter:"
        row += 1
        
        ws[f'A{row}'] = "Prediction Summary"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        ws[f'A{row}'] = "Total Predictions"
        ws[f'B{row}'] = len(self.predictions_df)
        row += 1
        
        ws[f'A{row}'] = "MAPE"
        if self.accuracy_df is not None:
            ws[f'B{row}'] = f"{self.accuracy_df['MAPE'].mean():.0f}"
        row += 2
        
        # Predictions Section
        ws[f'A{row}'] = "Predictions"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        # Institution header
        institution_row = row
        ws[f'A{institution_row}'] = "Central Connecticut State University"
        ws[f'A{institution_row}'].font = Font(bold=True, size=11)
        ws[f'A{institution_row}'].fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        ws.merge_cells(f'A{institution_row}:G{institution_row}')
        row += 1
        
        # School header
        school_row = row
        ws[f'A{school_row}'] = "School of Business"
        ws[f'A{school_row}'].font = Font(bold=True, size=10, italic=True)
        ws[f'A{school_row}'].fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
        ws.merge_cells(f'A{school_row}:G{school_row}')
        row += 1
        
        # Group predictions by subject (department)
        if 'Subject' in self.predictions_df.columns and 'Course' in self.predictions_df.columns:
            grouped = self.predictions_df.groupby('Subject')
            
            for subject, group_df in grouped:
                # Department header
                dept_row = row
                dept_name = self._get_department_name(subject)
                ws[f'B{dept_row}'] = dept_name
                ws[f'B{dept_row}'].font = Font(bold=True, size=10)
                ws[f'B{dept_row}'].fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
                ws.merge_cells(f'B{dept_row}:G{dept_row}')
                row += 1
                
                # Get unique courses in this department
                courses = group_df[['Course', 'Subject']].drop_duplicates()
                
                for _, course_row_data in courses.iterrows():
                    course_num = course_row_data['Course']
                    
                    # Course identifier
                    course_row = row
                    ws[f'A{course_row}'] = "Course Identifier"
                    row += 1
                    
                    # Course code and historical data header
                    ws[f'A{row}'] = f"{subject} {course_num}"
                    ws[f'B{row}'] = "Historical Enrollment Count"
                    ws[f'B{row}'].font = Font(bold=True)
                    
                    # Add year headers (2022, 2023, 2024, Predicted)
                    ws[f'C{row}'] = "2022"
                    ws[f'D{row}'] = "2023"
                    ws[f'E{row}'] = "2024"
                    ws[f'F{row}'] = "Predicted"
                    ws[f'F{row}'].fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                    
                    for col in ['C', 'D', 'E', 'F']:
                        ws[f'{col}{row}'].font = Font(bold=True)
                        ws[f'{col}{row}'].alignment = Alignment(horizontal='center')
                    
                    row += 1
                    
                    # Get prediction for this course (average across sections)
                    course_predictions = group_df[group_df['Course'] == course_num]
                    if len(course_predictions) > 0 and 'Predicted_Enrollment' in course_predictions.columns:
                        predicted_val = int(course_predictions['Predicted_Enrollment'].sum())
                        
                        # Data row (leave historical blank for now, show predicted)
                        ws[f'C{row}'] = ""  # Would need historical data
                        ws[f'D{row}'] = ""
                        ws[f'E{row}'] = ""
                        ws[f'F{row}'] = predicted_val
                        ws[f'F{row}'].alignment = Alignment(horizontal='center')
                        row += 1
                    
                    row += 1  # Add spacing between courses
        
        # Set column widths
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 35
        ws.column_dimensions['C'].width = 12
        ws.column_dimensions['D'].width = 12
        ws.column_dimensions['E'].width = 12
        ws.column_dimensions['F'].width = 12
        ws.column_dimensions['G'].width = 12
    
    def _get_department_name(self, subject_code: str) -> str:
        """Map subject codes to department names."""
        dept_mapping = {
            'MIS': 'Management Information Systems',
            'FIN': 'Finance Department',
            'AC': 'Accounting Department',
            'BUS': 'Business Administration',
            'MGT': 'Management Department',
            'MKT': 'Marketing Department',
            'ECON': 'Economics Department',
        }
        return dept_mapping.get(subject_code, f"{subject_code} Department")
    
    def _create_accuracy_sheet(self, wb: openpyxl.Workbook):
        """Create sheet with per-course accuracy analysis."""
        ws = wb.create_sheet("Accuracy Analysis")
        
        ws['A1'] = "Per-Course Accuracy Analysis"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:H1')
        
        # Add accuracy table
        start_row = 3
        for r_idx, row in enumerate(dataframe_to_rows(self.accuracy_df, index=False, header=True), start=start_row):
            for c_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                
                # Style header row
                if r_idx == start_row:
                    cell.font = self.header_font
                    cell.fill = self.header_fill
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                else:
                    # Color code MAPE values
                    if c_idx == self.accuracy_df.columns.get_loc('MAPE') + 1:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            if value < 20:
                                cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Light green
                            elif value < 40:
                                cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # Light yellow
                            else:
                                cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # Light red
                
                cell.border = self.border
        
        # Auto-fit columns
        for col_idx in range(1, len(self.accuracy_df.columns) + 1):
            max_length = 0
            column_letter = openpyxl.utils.get_column_letter(col_idx)
            for row_idx in range(start_row, start_row + len(self.accuracy_df) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            adjusted_width = min(max_length + 2, 30)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Add filters
        ws.auto_filter.ref = f"A{start_row}:{ws.cell(row=start_row, column=len(self.accuracy_df.columns)).column_letter}{start_row + len(self.accuracy_df)}"


def generate_enrollment_report(
    model_path: str,
    predictions_csv: Optional[str] = None,
    accuracy_csv: Optional[str] = None,
    output_path: Optional[str] = None,
    target_term: Optional[int] = None,
    courses_query: Optional[str] = None
):
    """
    Generate an enrollment prediction Excel report.
    
    Args:
        model_path: Path to trained model pickle file
        predictions_csv: Optional path to CSV file with predictions (if None, will run predictions)
        accuracy_csv: Optional path to CSV file with per-course accuracy
        output_path: Optional output path for Excel file (auto-generated if not provided)
        target_term: Optional term to predict for (required if predictions_csv is None)
        courses_query: Optional SQL query to get specific courses (if None, gets all courses)
    """
    # If predictions CSV not provided, run predictions
    if predictions_csv is None:
        if target_term is None:
            raise ValueError("target_term must be provided when predictions_csv is None")
        
        print("No predictions CSV provided, running predictions...")
        
        # Get course identifiers
        courses_df = get_course_identifiers(courses_query)
        
        # Run predictions
        predictions_df = run_predictions(model_path, courses_df, target_term)
        
        # Optionally save predictions to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_csv_path = Path(__file__).parent.parent / "reports" / f"predictions_{timestamp}.csv"
        predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions saved to: {predictions_csv_path}")
    else:
        # Load predictions from CSV
        print(f"Loading predictions from {predictions_csv}...")
        predictions_df = pd.read_csv(predictions_csv)
    
    # Load accuracy data if provided
    accuracy_df = pd.read_csv(accuracy_csv) if accuracy_csv else None
    
    # Generate default output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"enrollment_report_{timestamp}.xlsx")
    
    # Create report
    print(f"\nGenerating Excel report...")
    generator = EnrollmentReportGenerator(model_path, predictions_df, accuracy_df)
    generator.generate_report(output_path)
    
    # Print both container and Windows paths
    print(f"\n" + "="*80)
    print(f"✓ Report saved successfully!")
    print(f"="*80)
    print(f"Container path: {output_path}")
    
    # Convert to Windows path
    if output_path.startswith('/app/'):
        windows_path = output_path.replace('/app/', 'C:\\Development\\ccsu_enrollment\\Enrollment-Predictor\\').replace('/', '\\')
        print(f"Windows path:   {windows_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate enrollment prediction Excel reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with existing predictions CSV
  python generate_report.py --model data/prediction_models/enrollment_tree_min_20251112_024130.pkl --predictions predictions.csv --accuracy server/ml/test_results/per_course_accuracy_tree_20251109_013953.csv
  
  # Generate report by running predictions for Spring 2025
  python generate_report.py --model data/prediction_models/enrollment_tree_min_20251112_024130.pkl --term 202501
  
  # Generate report with accuracy data
  python generate_report.py --model data/prediction_models/enrollment_tree_min_20251112_024130.pkl --term 202501 --accuracy server/ml/test_results/per_course_accuracy_tree_20251109_013953.csv
  
  # Generate report with custom course query (only BUS and AC courses)
  python generate_report.py --model data/prediction_models/enrollment_tree_min_20251112_024130.pkl --term 202501 --query "SELECT DISTINCT subj, crse, title, credits FROM section_detail_report_sbussection_detail_report_sbus WHERE subj IN ('BUS', 'AC')"
        """
    )
    
    parser.add_argument("--model", required=True, help="Path to trained model pickle file (default location: data/prediction_models/)")
    parser.add_argument("--predictions", help="Path to CSV file with predictions (if not provided, will run predictions)")
    parser.add_argument("--accuracy", help="Path to CSV file with per-course accuracy metrics (default location: server/ml/test_results/)")
    parser.add_argument("--output", help="Output path for Excel report (auto-generated if not provided, saves to server/web/reports/)")
    parser.add_argument("--term", type=int, help="Term to predict for (e.g., 202501 for Spring 2025, required if --predictions not provided)")
    parser.add_argument("--query", help="SQL query to get specific courses (if not provided, gets all courses)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.predictions is None and args.term is None:
        parser.error("Either --predictions or --term must be provided")
    
    try:
        result = generate_enrollment_report(
            model_path=args.model,
            predictions_csv=args.predictions,
            accuracy_csv=args.accuracy,
            output_path=args.output,
            target_term=args.term,
            courses_query=args.query
        )
        print(f"\n✓ Report generation complete: {result}")
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
