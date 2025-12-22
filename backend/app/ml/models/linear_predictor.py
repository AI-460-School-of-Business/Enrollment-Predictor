"""Linear Regression implementation for enrollment prediction."""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from .base_predictor import EnrollmentPredictor
from ..utils.evaluation import analyze_per_course_accuracy


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
        mask = y_val > 0
        if mask.any():
            val_mape = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
        else:
            val_mape = np.nan
        
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
        
        # Report per-course accuracy if we have subj and crse features
        results['per_course_accuracy'] = analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred, self.model_type)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained linear model."""
        X_processed = self.preprocess_features(X, fit_transform=False)
        return self.model.predict(X_processed)
