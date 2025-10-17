"""
Neural Network Model for Enrollment Prediction

Implements deep learning approach using TensorFlow/Keras.
"""

from train_model import NeuralNetworkPredictor
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train neural network enrollment prediction model")
    parser.add_argument("--features", choices=["min", "rich"], default="min",
                       help="Feature schema to use")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    print(f"=== Training Neural Network model with {args.features} features ===")
    
    predictor = NeuralNetworkPredictor(args.features)
    
    try:
        # Extract and prepare data
        raw_data = predictor.extract_training_data()
        X, y = predictor.prepare_features(raw_data)
        
        # Train model
        results = predictor.train(X, y)
        
        # Save model to persistent storage
        from pathlib import Path
        from datetime import datetime
        
        model_dir = Path("/app/data/prediction_models")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"enrollment_neural_{args.features}_{timestamp}.pkl"
        model_path = model_dir / model_filename
        
        predictor.save_model(str(model_path))
        
        print(f"\n=== Training Complete ===")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()