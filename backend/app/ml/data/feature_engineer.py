"""Feature engineering and preparation."""
from typing import Dict, List, Tuple
import pandas as pd
from .data_loader import DataLoader


class FeatureEngineer:
    """Prepare features for model training."""
    
    def __init__(self, feature_schema: str = "min"):
        self.feature_schema = feature_schema
        self.target_column = "act_enrollment"
        self.feature_columns = []
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from the joined data."""
        print("Preparing features and target variables...")
        
        # Print basic info about the data
        print(f"Raw data shape: {data.shape}")
        print(f"Total columns: {len(data.columns)}")
        
        # Identify key columns
        key_cols = DataLoader._identify_key_columns(data.columns)
        
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
            
        # Aggregate data by Course and Term (summing enrollment)
        print("\nAggregating data by Subject, Course, and Term...")
        
        # Determine grouping columns
        group_cols = []
        if key_cols['term']:
            group_cols.append(key_cols['term'])
        else:
            if key_cols['year']: group_cols.append(key_cols['year'])
            if key_cols['semester']: group_cols.append(key_cols['semester'])
            
        group_cols.append(key_cols['subj'])
        group_cols.append(key_cols['crse'])
        
        # Determine aggregation rules
        agg_dict = {self.target_column: 'sum'}
        
        # Handle credits if present (take max to be safe)
        credits_col = None
        for col in features_data.columns:
            if col.lower() == 'credits':
                credits_col = col
                break
        
        if credits_col:
            agg_dict[credits_col] = 'max'
            
        # Perform aggregation
        try:
            features_data = features_data.groupby(group_cols, as_index=False).agg(agg_dict)
            print(f"Data after aggregation: {features_data.shape}")
        except Exception as e:
            print(f"Warning: Aggregation failed ({e}). Using unaggregated data.")
        
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
        # Note: 'sec' (Section) is excluded as we predict at the Course level
        desired_features = ['term', 'subj', 'crse', 'credits']
        
        if self.feature_schema == "min":
            # Find the available desired features in our dataset
            feature_cols = []
            columns_lower = {col.lower(): col for col in data.columns}
            
            # Always include term (required)
            if key_cols['term']:
                feature_cols.append(key_cols['term'])
            elif key_cols['semester']:  # Fallback
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
                if any(id_word in col_lower for id_word in ['id', 'guid', 'uuid']):
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
        missing = X.isnull().sum()
        if missing.any():
            print(f"\nMissing values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(X)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("  No missing values found")
        
        # Data type distribution
        print(f"\nData types:")
        print(X.dtypes.value_counts())
        
        # Sample of feature values
        print(f"\nSample feature values:")
        for col in X.columns[:5]:
            unique_vals = X[col].unique()
            sample_vals = unique_vals[:5] if len(unique_vals) > 5 else unique_vals
            print(f"  {col}: {sample_vals} (total unique: {len(unique_vals)})")
