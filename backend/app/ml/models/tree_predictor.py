"""Tree-based model (Random Forest) for enrollment prediction."""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from models.base_predictor import EnrollmentPredictor
from utils.evaluation import analyze_per_course_accuracy


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
        
        # Report per-course accuracy if we have subj and crse features
        results['per_course_accuracy'] = analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred, self.model_type)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained tree model."""
        X_processed = self.preprocess_features(X, fit_transform=False)
        return self.model.predict(X_processed)
