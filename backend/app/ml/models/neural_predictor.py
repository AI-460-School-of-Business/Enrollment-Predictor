"""Neural Network implementation for enrollment prediction."""
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.base_predictor import EnrollmentPredictor
from utils.evaluation import analyze_per_course_accuracy


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
        mask = y_val > 0
        if mask.any():
            val_mape = np.mean(np.abs((y_val[mask] - y_val_pred.flatten()[mask]) / y_val[mask])) * 100
        else:
            val_mape = np.nan
        
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
        
        # Report per-course accuracy if we have subj and crse features
        results['per_course_accuracy'] = analyze_per_course_accuracy(X_val_raw, y_val, y_val_pred.flatten(), self.model_type)
        
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
        
        # Save preprocessing metadata
        model_data = {
            'model_path': str(keras_path),
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'feature_schema': self.feature_schema
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Neural network saved to {keras_path}")
        print(f"Preprocessing metadata saved to {filepath}")
