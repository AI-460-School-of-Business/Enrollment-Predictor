"""Base class for enrollment prediction models."""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
            Path(__file__).parent.parent / "feature_schema" / f"enrollment_features_{self.feature_schema}.json",
            Path("/app/server/ml/feature_schema") / f"enrollment_features_{self.feature_schema}.json",
            Path("server/ml/feature_schema") / f"enrollment_features_{self.feature_schema}.json"
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
