"""
Enrollment Prediction Model Training

Trains models to predict enrollment counts using Subject+Course, Semester, Year data.
Course identity is determined by the combination of subj (subject) and crse (course number).
Usage: python server/ml/train_model.py --model [linear|tree|neural] --features [min|rich]
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import argparse
import pickle
import joblib
import textwrap

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Database
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}

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
    
    print("Options:")
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
        lines: List[str] = []
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


class EnrollmentPredictor:
    """Base class for enrollment prediction models."""
    
    def __init__(self, model_type: str, feature_schema: str = "min", custom_query: Optional[str] = None):
        self.model_type = model_type
        self.feature_schema = feature_schema
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = "act_enrollment"
        self.custom_query = custom_query
        
        # Load feature schema
        self.schema = self._load_feature_schema()
        
    def _load_feature_schema(self) -> Dict:
        """Load the feature schema configuration."""
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent / "feature_schema" / f"enrollment_features_{self.feature_schema}.json",  # New location
            Path(__file__).parent.parent / "ai" / "feature_schema" / f"enrollment_features_{self.feature_schema}.json",
            Path("/app/server/ml/feature_schema") / f"enrollment_features_{self.feature_schema}.json",  # Container path
            Path("/app/server/ai/feature_schema") / f"enrollment_features_{self.feature_schema}.json",
            Path("server/ai/feature_schema") / f"enrollment_features_{self.feature_schema}.json"
        ]
        
        for schema_path in possible_paths:
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    return json.load(f)
        
        # If no schema file found, create a basic one
        print(f"No feature schema found, using basic schema for {self.feature_schema}")
        if self.feature_schema == "min":
            return {"Subject": "string", "Course": "string", "Semester": "string", "EnrollmentCount": "integer"}
        else:
            return {"Subject": "string", "Course": "string", "Semester": "string", "EnrollmentCount": "integer", "additional_features": "mixed"}
    
    def get_db_connection(self):
        """Create database connection."""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            raise
    
    def extract_training_data(self) -> pd.DataFrame:
        """Extract training data from database and join related tables."""
        print(f"Extracting training data using {self.feature_schema} schema...")
        
        conn = self.get_db_connection()
        try:
            if self.custom_query:
                print("Custom SQL query provided. Skipping automatic table discovery.")
                print("Executing custom query...\n")
                custom_data = pd.read_sql(self.custom_query, conn)
                print(f"Custom query returned {custom_data.shape[0]} rows and {custom_data.shape[1]} columns")
                return custom_data
            
            table_metadata = self._analyze_table_structures(conn)
            
            primary_table = self._identify_primary_table(table_metadata)
            if not primary_table:
                raise ValueError("Could not find primary table with Subject, Course, Semester, Year, and Enrollment")
            
            print(f"Primary table: {primary_table['name']}")
            
            joinable_tables = self._identify_joinable_tables(table_metadata, primary_table)
            print(f"Found {len(joinable_tables)} joinable tables: {[t['name'] for t in joinable_tables]}")
            combined_data = self._extract_and_join_data(conn, primary_table, joinable_tables)
            
            return combined_data
            
        finally:
            conn.close()
    
    def _analyze_table_structures(self, conn) -> List[Dict[str, Any]]:
        """Analyze all tables and their column structures."""
        print("Analyzing table structures...")
        
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """
        
        tables_df = pd.read_sql(tables_query, conn)
        available_tables = tables_df['table_name'].tolist()
        print(f"Available tables: {available_tables}")
        
        table_metadata = []
        
        for table_name in available_tables:
            # Get detailed column info
            columns_query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
            
            columns_df = pd.read_sql(columns_query, conn)
            column_info = columns_df.to_dict('records')
            
            column_names = [col['column_name'] for col in column_info]
            column_names_lower = [col.lower() for col in column_names]
            
            has_subj = any('subj' in col for col in column_names_lower)
            has_crse = any('crse' in col or 'course' in col for col in column_names_lower)
            has_term = any(col == 'term' for col in column_names_lower)
            has_semester = any('semester' in col for col in column_names_lower) or has_term
            has_year = any('year' in col for col in column_names_lower) or has_term
            has_enrollment = any('enrollment' in col or 'headcount' in col or col == 'act' for col in column_names_lower)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count = pd.read_sql(count_query, conn).iloc[0]['row_count']
            
            metadata = {
                'name': table_name,
                'columns': column_info,
                'column_names': column_names,
                'column_names_lower': column_names_lower,
                'has_subj': has_subj,
                'has_crse': has_crse,
                'has_term': has_term,
                'has_semester': has_semester,
                'has_year': has_year,
                'has_enrollment': has_enrollment,
                'row_count': row_count
            }
            
            table_metadata.append(metadata)
            
            print(f"Table {table_name}: {row_count} rows, Subj={has_subj}, Crse={has_crse}, Term={has_term}, Semester={has_semester}, Year={has_year}, Enrollment={has_enrollment}")
            print(f"  Columns: {column_names[:10]}...")
        
        return table_metadata
    
    def _identify_primary_table(self, table_metadata: List[Dict]) -> Optional[Dict]:
        """Identify the primary table that has Subject, Course, Semester, Year, and Enrollment."""
        candidates = []
        
        for table in table_metadata:
            # Must have Subject, Course, Semester, and Enrollment at minimum
            if table['has_subj'] and table['has_crse'] and table['has_semester'] and table['has_enrollment']:
                score = 0
                score += 10 if table['has_subj'] else 0
                score += 10 if table['has_crse'] else 0
                score += 10 if table['has_semester'] else 0
                score += 10 if table['has_year'] else 0
                score += 10 if table['has_enrollment'] else 0
                score += min(table['row_count'] / 1000, 10)
                
                candidates.append((score, table))
        
        if not candidates:
            return None
        
        # Return the highest scoring table
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def _identify_joinable_tables(self, table_metadata: List[Dict], primary_table: Dict) -> List[Dict]:
        """Identify tables that can be joined to the primary table via Year/Semester."""
        joinable = []
        
        for table in table_metadata:
            if table['name'] == primary_table['name']:
                continue
            
            # Must have at least Term, Year or Semester to be joinable
            if table.get('has_term') or table['has_year'] or table['has_semester']:
                # Determine join strategy
                join_strategy = []
                if table.get('has_term'):
                    join_strategy.append('term')
                if table['has_year']:
                    join_strategy.append('year')
                if table['has_semester']:
                    join_strategy.append('semester')
                
                table['join_strategy'] = join_strategy
                joinable.append(table)
                
                print(f"  {table['name']}: joinable via {join_strategy}")
        
        return joinable
    
    def _extract_and_join_data(self, conn, primary_table: Dict, joinable_tables: List[Dict]) -> pd.DataFrame:
        """Extract data from primary table and join with related tables."""
        print("Extracting and joining data...")
        
        print(f"Loading primary table: {primary_table['name']} (filtered to 2023 onward)")
        
        year_col = None
        if 'year' in [col.lower() for col in primary_table['column_names']]:
            for col in primary_table['column_names']:
                if col.lower() == 'year':
                    year_col = col
                    break        # Build the query with year filter
        if year_col:
            print(f"Filtering by year column: {year_col}")
            primary_query = f"SELECT * FROM {primary_table['name']} WHERE {year_col} >= 2023"
        elif any('term' == col.lower() for col in primary_table['column_names']):
            term_col = next(col for col in primary_table['column_names'] if col.lower() == 'term')
            print(f"Filtering by term column: {term_col}")
            primary_query = f"SELECT * FROM {primary_table['name']} WHERE ({term_col} / 100) >= 2023"
        else:
            print("WARNING: Could not find year or term column for filtering. Using all data.")
            primary_query = f"SELECT * FROM {primary_table['name']}"
        
        print(f"Query: {primary_query}")
        base_data = pd.read_sql(primary_query, conn)
        
        print(f"Primary data shape after 2023 filtering: {base_data.shape}")
        
        # Identify key columns in primary table
        primary_cols = self._identify_key_columns(base_data.columns)
        print(f"Primary table key columns: {primary_cols}")
        
        # Join with each related table
        for related_table in joinable_tables:
            try:
                print(f"\nJoining with: {related_table['name']}")
                
                # Load related table data from 2023 onward
                # Determine which column to use for year filtering in related table
                year_col = None
                term_col = None
                
                for col in related_table['column_names']:
                    if col.lower() == 'year':
                        year_col = col
                        break
                    elif col.lower() == 'term':
                        term_col = col
                
                # Build the query with year filter
                if year_col:
                    print(f"Filtering related table by year column: {year_col}")
                    related_query = f"SELECT * FROM {related_table['name']} WHERE {year_col} >= 2023"
                elif term_col:
                    print(f"Filtering related table by term column: {term_col}")
                    related_query = f"SELECT * FROM {related_table['name']} WHERE ({term_col} / 100) >= 2023"
                else:
                    print(f"WARNING: Could not find year or term column for filtering in {related_table['name']}. Using all data.")
                    related_query = f"SELECT * FROM {related_table['name']}"
                
                print(f"Related query: {related_query}")
                related_data = pd.read_sql(related_query, conn)
                
                print(f"Related data shape (2023+ only): {related_data.shape}")
                
                # Identify key columns in related table
                related_cols = self._identify_key_columns(related_data.columns)
                print(f"Related table key columns: {related_cols}")
                
                # Determine join keys: prefer 'term' if present in both; else fall back to year/semester
                join_keys = []
                if primary_cols.get('term') and related_cols.get('term'):
                    join_keys = [('term', 'term')]
                else:
                    if primary_cols['year'] and related_cols['year']:
                        join_keys.append(('year', 'year'))
                    if primary_cols['semester'] and related_cols['semester']:
                        join_keys.append(('semester', 'semester'))
                
                if not join_keys:
                    print(f"  Cannot join {related_table['name']} - no common keys")
                    continue
                
                # Perform the join
                left_keys = [primary_cols[k[0]] for k in join_keys if primary_cols[k[0]]]
                right_keys = [related_cols[k[1]] for k in join_keys if related_cols[k[1]]]
                
                if left_keys and right_keys:
                    print(f"  Joining on: {list(zip(left_keys, right_keys))}")
                    
                    # Add table prefix to avoid column name conflicts
                    related_data_prefixed = related_data.copy()
                    for col in related_data_prefixed.columns:
                        if col not in right_keys:  # Don't prefix join keys
                            related_data_prefixed = related_data_prefixed.rename(columns={col: f"{related_table['name']}_{col}"})
                    
                    # Perform left join
                    base_data = pd.merge(
                        base_data, 
                        related_data_prefixed, 
                        left_on=left_keys, 
                        right_on=right_keys, 
                        how='left',
                        suffixes=('', f'_{related_table["name"]}')
                    )
                    
                    print(f"  After join shape: {base_data.shape}")
                else:
                    print(f"  Missing join keys for {related_table['name']}")
                    
            except Exception as e:
                print(f"  Error joining {related_table['name']}: {e}")
                continue
        
        print(f"\nFinal combined data shape: {base_data.shape}")
        print(f"Final columns: {len(base_data.columns)} total")
        
        # Show data distribution by year to confirm our filtering worked
        print("\n=== Confirming data is from 2023 onward ===")
        
        # Check if we have a year column
        year_col = None
        term_col = None
        
        for col in base_data.columns:
            if col.lower() == 'year':
                year_col = col
                break
            elif col.lower() == 'term':
                term_col = col
        
        # Display year distribution
        if year_col:
            year_counts = base_data[year_col].value_counts().sort_index()
            print(f"\nData distribution by year column '{year_col}':")
            for year, count in year_counts.items():
                print(f"  Year {year}: {count} records")
                
            # Validate no records before 2023
            if any(year < 2023 for year in year_counts.index if isinstance(year, (int, float))):
                print("WARNING: Some records are from before 2023!")
                
        elif term_col:
            # Extract year from term and count - updated to handle integer terms consistently
            base_data['extracted_year'] = (base_data[term_col] / 100).astype(int)
                
            year_counts = base_data['extracted_year'].value_counts().sort_index()
            print(f"\nData distribution by year (extracted from '{term_col}'):")
            for year, count in year_counts.items():
                if isinstance(year, (int, float)):  # Skip non-numeric values
                    print(f"  Year {year}: {count} records")
                    
            # Validate no records before 2023
            if any(year < 2023 for year in year_counts.index if isinstance(year, (int, float))):
                print("WARNING: Some records are from before 2023!")
        
        # Show title counts from the data
        if 'title' in base_data.columns:
            title_counts = base_data['title'].value_counts()
            print(f"\nCourse Title Distribution ({len(title_counts)} unique titles):")
            for title, count in title_counts.items():
                print(f"  {title}: {count} rows")
            
            # Also show enrollment statistics by title
            if 'act' in base_data.columns:
                print("\nAverage enrollment (act) by title:")
                title_enrollments = base_data.groupby('title')['act'].agg(['mean', 'min', 'max', 'count'])
                for title, stats in title_enrollments.iterrows():
                    print(f"  {title}: avg={stats['mean']:.1f}, min={stats['min']}, max={stats['max']}, count={stats['count']}")
        
        return base_data
    
    def _identify_key_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """Identify key columns (Subject, Course, Term/Semester/Year, Enrollment) in a list of column names."""
        key_cols = {
            'subj': None,
            'crse': None,
            'term': None,
            'semester': None, 
            'year': None,
            'enrollment': None
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if 'subj' in col_lower and key_cols['subj'] is None:
                key_cols['subj'] = col
            elif ('crse' in col_lower or (col_lower == 'course' and 'crse' not in [c.lower() for c in columns])) and key_cols['crse'] is None:
                key_cols['crse'] = col
            elif (col_lower == 'term') and key_cols['term'] is None:
                key_cols['term'] = col
            elif ('semester' in col_lower) and key_cols['semester'] is None:
                key_cols['semester'] = col
            elif ('year' in col_lower) and key_cols['year'] is None:
                key_cols['year'] = col
            elif ('enrollment' in col_lower or 'headcount' in col_lower or col_lower == 'act') and key_cols['enrollment'] is None:
                key_cols['enrollment'] = col
        
        return key_cols
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from the joined data."""
        print("Preparing features and target variables...")
        
        # Print basic info about the data
        print(f"Raw data shape: {data.shape}")
        print(f"Total columns: {len(data.columns)}")
        
        # Identify key columns
        key_cols = self._identify_key_columns(data.columns)
        
        print(f"\nIdentified key columns:")
        print(f"  Subject: {key_cols['subj']}")
        print(f"  Course: {key_cols['crse']}")
        print(f"  Term: {key_cols['term']}")
        print(f"  Semester: {key_cols['semester']}")
        print(f"  Year: {key_cols['year']}")
        print(f"  Enrollment (act): {key_cols['enrollment']}")
        
        # Require Subject + Course + (Term or Semester) + Enrollment
        if not (key_cols['subj'] and key_cols['crse'] and (key_cols['term'] or key_cols['semester']) and key_cols['enrollment']):
            raise ValueError("Could not identify required columns (Subject, Course, Term/Semester, Enrollment/act)")
        
        # Create working copy
        features_data = data.copy()
        
        # Prepare target variable
        enrollment_col = key_cols['enrollment']
        print(f"\nUsing '{enrollment_col}' as our target enrollment prediction column")
        features_data[self.target_column] = pd.to_numeric(
            features_data[enrollment_col], errors='coerce'
        )
        
        # Remove rows with missing target
        initial_rows = len(features_data)
        features_data = features_data.dropna(subset=[self.target_column])
        rows_removed = initial_rows - len(features_data)
        
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing enrollment data")
        
        print(f"Data after cleaning: {features_data.shape}")
        print(f"Target statistics:")
        print(features_data[self.target_column].describe())
        
        # Select features based on schema type
        feature_cols = self._select_features(features_data, key_cols)
        
        # Store the feature columns for later use
        self.feature_columns = feature_cols
        
        X = features_data[feature_cols]
        y = features_data[self.target_column]
        
        print(f"\nSelected {len(feature_cols)} feature columns")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Show feature distribution
        self._analyze_features(X)
        
        return X, y
    
    def _select_features(self, data: pd.DataFrame, key_cols: Dict[str, str]) -> List[str]:
        """Select appropriate features based on the schema type."""
        
        # Define specific features we want to use for enrollment prediction
        desired_features = ['term', 'subj', 'crse', 'sec', 'credits']
        
        if self.feature_schema == "min":
            # Find the available desired features in our dataset
            feature_cols = []
            columns_lower = {col.lower(): col for col in data.columns}
            
            # Always include term (required)
            if key_cols['term']:
                feature_cols.append(key_cols['term'])
            elif key_cols['semester']: # Fallback
                feature_cols.append(key_cols['semester'])
                
            # Add other desired features if available (case insensitive)
            for feature in desired_features:
                if feature.lower() in columns_lower and columns_lower[feature.lower()] not in feature_cols:
                    feature_cols.append(columns_lower[feature.lower()])
            
            print(f"Using specific features for enrollment prediction: {feature_cols}")
            
        else:
            # Rich features: include all available columns plus our specific desired columns
            feature_cols = []
            excluded_cols = {self.target_column, key_cols['enrollment']}
            
            # Include our specific desired features first (if available)
            desired_features = ['term', 'subj', 'crse', 'sec', 'credits']
            columns_lower = {col.lower(): col for col in data.columns}
            
            print("\nIncluding specifically desired features:")
            for feature in desired_features:
                if feature.lower() in columns_lower:
                    col = columns_lower[feature.lower()]
                    if col not in excluded_cols:
                        feature_cols.append(col)
                        print(f"  Added desired feature: {col}")
                else:
                    print(f"  Missing desired feature: {feature}")
            
            # Categorize remaining columns
            categorical_cols = []
            numerical_cols = []
            id_cols = []
            
            for col in data.columns:
                if col in excluded_cols or col in feature_cols:
                    continue
                
                col_lower = col.lower()
                
                # Skip obvious ID columns that aren't useful features
                if any(id_word in col_lower for id_word in ['id', 'guid', 'uuid']) and col not in [key_cols['crn']]:
                    id_cols.append(col)
                    continue
                
                # Check data type and content
                if data[col].dtype in ['object', 'category']:
                    # Categorical - but only if not too many unique values
                    unique_count = data[col].nunique()
                    if unique_count <= min(100, len(data) * 0.1):
                        categorical_cols.append(col)
                    else:
                        print(f"Skipping high-cardinality categorical column: {col} ({unique_count} unique values)")
                elif data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numerical
                    numerical_cols.append(col)
                else:
                    # Other types - convert if possible
                    try:
                        pd.to_numeric(data[col], errors='raise')
                        numerical_cols.append(col)
                    except:
                        if data[col].nunique() <= 50:
                            categorical_cols.append(col)
            
            # Add remaining columns to feature list
            feature_cols.extend(categorical_cols)
            feature_cols.extend(numerical_cols)
            
            print(f"Rich feature selection:")
            print(f"  Categorical features: {len(categorical_cols)}")
            print(f"  Numerical features: {len(numerical_cols)}")
            print(f"  Excluded ID columns: {len(id_cols)}")
            print(f"  Total selected: {len(feature_cols)}")
            
            if len(categorical_cols) > 0:
                print(f"  Sample categorical: {categorical_cols[:5]}")
            if len(numerical_cols) > 0:
                print(f"  Sample numerical: {numerical_cols[:5]}")
        
        return feature_cols
    
    def _analyze_features(self, X: pd.DataFrame):
        """Analyze the selected features for quality and completeness."""
        print(f"\nFeature Analysis:")
        
        # Missing value analysis
        missing_stats = X.isnull().sum()
        missing_pct = (missing_stats / len(X) * 100).round(2)
        
        columns_with_missing = missing_stats[missing_stats > 0]
        if len(columns_with_missing) > 0:
            print(f"Columns with missing values:")
            for col in columns_with_missing.index:
                print(f"  {col}: {missing_stats[col]} ({missing_pct[col]}%)")
        else:
            print("No missing values found")
        
        # Data type distribution
        dtype_counts = X.dtypes.value_counts()
        print(f"\nData type distribution:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Show sample values for first few columns
        print(f"\nSample values from first few features:")
        for col in X.columns[:5]:
            unique_vals = X[col].unique()
            sample_vals = unique_vals[:5] if len(unique_vals) > 5 else unique_vals
            print(f"  {col}: {sample_vals} (total unique: {len(unique_vals)})")
    
    def preprocess_features(self, X: pd.DataFrame, fit_transform: bool = True) -> np.ndarray:
        """
        Preprocess features for ML consumption.
        Handle categorical variables, scaling, etc.
        """
        print("Preprocessing features...")
        
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        numerical_columns = X_processed.select_dtypes(include=[np.number]).columns
        
        print(f"Categorical columns: {list(categorical_columns)}")
        print(f"Numerical columns: {list(numerical_columns)}")
        
        # Ensure 'subj', 'sec', and 'crse' are treated as categorical features even if they appear numeric
        for special_cat in ['subj', 'sec', 'crse']:
            if special_cat in X_processed.columns and special_cat not in categorical_columns:
                print(f"Converting {special_cat} to categorical feature")
                X_processed[special_cat] = X_processed[special_cat].astype(str)
                categorical_columns = list(categorical_columns) + [special_cat]
        
        # Encode categorical variables
        for col in categorical_columns:
            if fit_transform:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(X_processed[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        print(f"Warning: Unknown categories in {col}: {unknown_values}")
                        # Replace unknown values with the most frequent known value
                        most_frequent = self.label_encoders[col].classes_[0]
                        X_processed[col] = X_processed[col].astype(str).replace(list(unknown_values), most_frequent)
                    
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Scale numerical features
        if fit_transform:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Filter out zero values to avoid division by zero
        mask = y_test > 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan  # Set to NaN if all values are zero
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mape
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'feature_schema': self.feature_schema
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def _analyze_per_course_accuracy(self, X_val: pd.DataFrame, y_val: pd.Series, y_pred: np.ndarray) -> Optional[pd.DataFrame]:
        """Analyze prediction accuracy per course (subj+crse combination).
        
        Args:
            X_val: Original unprocessed validation features DataFrame
            y_val: Validation target values
            y_pred: Predicted values
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
        
        # Save complete ranked list to CSV
        try:
            # Use a path relative to the script location (works in Docker)
            output_dir = Path(__file__).parent / "test_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"per_course_accuracy_{self.model_type}_{timestamp}.csv"
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
        
        return course_stats


class LinearRegressionPredictor(EnrollmentPredictor):
    """Linear Regression implementation for enrollment prediction."""
    
    def __init__(self, feature_schema: str = "min", custom_query: Optional[str] = None):
        super().__init__("linear", feature_schema, custom_query=custom_query)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train linear regression model with hyperparameter tuning."""
        print("Training Linear Regression model...")
        
        # Split data BEFORE preprocessing to preserve original DataFrames
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess features
        X_train = self.preprocess_features(X_train_raw, fit_transform=True)
        X_val = self.preprocess_features(X_val_raw, fit_transform=False)
        
        # Hyperparameter tuning
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        }
        
        ridge = Ridge()
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Predict on validation set
        y_val_pred = self.model.predict(X_val)
        
        # Calculate MAPE
        # Filter out zero values to avoid division by zero
        mask = y_val > 0
        if mask.any():
            val_mape = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
        else:
            val_mape = np.nan  # Set to NaN if all values are zero
        
        results = {
            'best_params': grid_search.best_params_,
            'train_r2': train_score,
            'val_r2': val_score,
            'val_mape': val_mape,
            'cv_scores': grid_search.cv_results_['mean_test_score']
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Train R²: {train_score:.4f}")
        print(f"Validation R²: {val_score:.4f}")
        print(f"Validation MAPE: {val_mape:.2f}%")
        
        # Report per-course accuracy if we have subj and crse features (use raw unprocessed data)
        results['per_course_accuracy'] = self._analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained linear model."""
        X_processed = self.preprocess_features(X, fit_transform=False)
        return self.model.predict(X_processed)


class TreePredictor(EnrollmentPredictor):
    """Tree-based model (Random Forest) for enrollment prediction."""
    
    def __init__(self, feature_schema: str = "min", custom_query: Optional[str] = None):
        super().__init__("tree", feature_schema, custom_query=custom_query)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter tuning."""
        print("Training Random Forest model...")
        
        # Split data BEFORE preprocessing to preserve original DataFrames
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess features
        X_train = self.preprocess_features(X_train_raw, fit_transform=True)
        X_val = self.preprocess_features(X_val_raw, fit_transform=False)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Predict on validation set
        y_val_pred = self.model.predict(X_val)
        
        # Calculate MAPE
        # Filter out zero values to avoid division by zero
        mask = y_val > 0
        if mask.any():
            val_mape = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
        else:
            val_mape = np.nan  # Set to NaN if all values are zero
        
        results = {
            'best_params': grid_search.best_params_,
            'train_r2': train_score,
            'val_r2': val_score,
            'val_mape': val_mape,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Train R²: {train_score:.4f}")
        print(f"Validation R²: {val_score:.4f}")
        print(f"Validation MAPE: {val_mape:.2f}%")
        print("Top 5 important features:")
        sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feat, importance in sorted_features[:5]:
            print(f"  {feat}: {importance:.4f}")
        
        # Report per-course accuracy if we have subj and crse features (use raw unprocessed data)
        results['per_course_accuracy'] = self._analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained tree model."""
        X_processed = self.preprocess_features(X, fit_transform=False)
        return self.model.predict(X_processed)


class NeuralNetworkPredictor(EnrollmentPredictor):
    """Neural Network implementation for enrollment prediction."""
    
    def __init__(self, feature_schema: str = "min", custom_query: Optional[str] = None):
        super().__init__("neural", feature_schema, custom_query=custom_query)
        self.model_path = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train neural network model."""
        print("Training Neural Network model...")
        
        # Split data BEFORE preprocessing to preserve original DataFrames
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess features
        X_train = self.preprocess_features(X_train_raw, fit_transform=True)
        X_val = self.preprocess_features(X_val_raw, fit_transform=False)
        
        # Build neural network
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Evaluate
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Predict on validation set
        y_val_pred = self.model.predict(X_val)
        
        # Calculate MAPE
        # Filter out zero values to avoid division by zero
        mask = y_val > 0
        if mask.any():
            val_mape = np.mean(np.abs((y_val[mask] - y_val_pred.flatten()[mask]) / y_val[mask])) * 100
        else:
            val_mape = np.nan  # Set to NaN if all values are zero
        
        results = {
            'train_loss': train_loss[0],
            'train_mae': train_loss[1],
            'val_loss': val_loss[0],
            'val_mae': val_loss[1],
            'val_mape': val_mape,
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"Train Loss: {train_loss[0]:.4f}, MAE: {train_loss[1]:.4f}")
        print(f"Val Loss: {val_loss[0]:.4f}, MAE: {val_loss[1]:.4f}")
        print(f"Validation MAPE: {val_mape:.2f}%")
        print(f"Epochs trained: {results['epochs_trained']}")
        
        # Report per-course accuracy if we have subj and crse features (use raw unprocessed data)
        results['per_course_accuracy'] = self._analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred.flatten())
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained neural network."""
        X_processed = self.preprocess_features(X, fit_transform=False)
        return self.model.predict(X_processed).flatten()
    
    def save_model(self, filepath: str):
        """Save neural network model."""
        # Save the Keras model with proper extension
        keras_dir = Path(filepath).parent
        keras_filename = f"{Path(filepath).stem}_keras.keras"
        keras_path = keras_dir / keras_filename
        self.model.save(str(keras_path))
        self.model_path = str(keras_path)
        
        print(f"Neural network saved to {keras_path}")
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'feature_schema': self.feature_schema,
            'model_path': self.model_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model metadata saved to {filepath}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train enrollment prediction models")
    parser.add_argument("--model", choices=["linear", "tree", "neural"], required=True,
                       help="Model type to train")
    parser.add_argument("--features", choices=["min", "rich", "auto"], default="min",
                       help="Feature schema to use (min=specific features, rich=all features, auto=auto-generated)")
    parser.add_argument("--data-query-file", type=str,
                       help="Path to a SQL file whose contents will be executed to retrieve training data")
    parser.add_argument("--data-query", type=str,
                       help="Inline SQL query to retrieve training data")
    parser.add_argument("--interactive-query", action="store_true",
                       help="Interactively preview and edit a SQL query template before training")
    
    args = parser.parse_args()
    
    print(f"=== Training {args.model} model with {args.features} features ===")
    if args.features == "min":
        print("Using specific features: term, subj, crse, sec, credits to predict act enrollment")

    custom_query: Optional[str] = None
    if args.data_query_file:
        query_path = Path(args.data_query_file)
        if not query_path.exists():
            print(f"Error: query file not found at {query_path}")
            sys.exit(1)
        custom_query = query_path.read_text().strip() or None
        print(f"Loaded custom query from {query_path}")
    elif args.data_query:
        custom_query = args.data_query.strip() or None
        print("Using custom query provided via --data-query")
    elif args.interactive_query:
        custom_query = prompt_for_custom_query(DEFAULT_CUSTOM_QUERY)
    
    if custom_query:
        preview_lines = custom_query.strip().splitlines()
        preview_display = preview_lines[:5]
        if len(preview_lines) > 5:
            preview_display.append("...")
        print("Custom query preview:")
        print(textwrap.indent("\n".join(preview_display), "    "))
        print()
    
    # Create predictor based on model type
    if args.model == "linear":
        predictor = LinearRegressionPredictor(args.features, custom_query=custom_query)
    elif args.model == "tree":
        predictor = TreePredictor(args.features, custom_query=custom_query)
    elif args.model == "neural":
        predictor = NeuralNetworkPredictor(args.features, custom_query=custom_query)
    
    try:
        # Extract and prepare data
        raw_data = predictor.extract_training_data()
        X, y = predictor.prepare_features(raw_data)
        
        # Train model
        results = predictor.train(X, y)
        
        print(f"\n=== Training Complete ===")
        print(f"Results: {results}")
        
        # Prompt user to save model
        save_model = input("\nWould you like to save this model? (y/n): ").strip().lower()
        
        if save_model == 'y' or save_model == 'yes':
            # Save model to persistent storage
            model_dir = Path("/app/data/prediction_models")
            model_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"enrollment_{args.model}_{args.features}_{timestamp}.pkl"
            model_path = model_dir / model_filename
            
            predictor.save_model(str(model_path))
            print(f"Model saved: {model_path}")
        else:
            print("Model was not saved.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
