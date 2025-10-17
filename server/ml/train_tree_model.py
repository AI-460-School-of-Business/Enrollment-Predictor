"""
Tree-based Model for Enrollment Prediction

Implements Random Forest and Gradient Boosting models.
"""

from train_model import TreePredictor
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train tree-based enrollment prediction model")
    parser.add_argument("--features", choices=["min", "rich"], default="min",
                       help="Feature schema to use")
    parser.add_argument("--algorithm", choices=["random_forest", "gradient_boost"], default="random_forest",
                       help="Tree algorithm to use")
    
    args = parser.parse_args()
    
    print(f"=== Training {args.algorithm} model with {args.features} features ===")
    
    predictor = TreePredictor(args.features)
    
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
        model_filename = f"enrollment_tree_{args.features}_{timestamp}.pkl"
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