"""
Quick test to verify the refactored module structure works correctly.
Run this from the ml directory: python test_imports.py
"""

print("Testing module imports...")

try:
    print("\n1. Testing models package...")
    from models.linear_predictor import LinearRegressionPredictor
    from models.tree_predictor import TreePredictor
    from models.neural_predictor import NeuralNetworkPredictor
    print("   ✓ All model classes imported successfully")
    
    print("\n2. Testing data package...")
    from data.data_loader import DataLoader
    from data.feature_engineer import FeatureEngineer
    print("   ✓ Data classes imported successfully")
    
    print("\n3. Testing utils package...")
    from utils.db_config import DB_CONFIG
    from utils.evaluation import analyze_per_course_accuracy
    print("   ✓ Utility functions imported successfully")
    
    print("\n4. Testing predictor initialization...")
    linear = LinearRegressionPredictor("min")
    tree = TreePredictor("min")
    neural = NeuralNetworkPredictor("min")
    print("   ✓ All predictor classes instantiated successfully")
    
    print("\n5. Testing data loader initialization...")
    loader = DataLoader()
    print("   ✓ DataLoader instantiated successfully")
    
    print("\n6. Testing feature engineer initialization...")
    engineer = FeatureEngineer("min")
    print("   ✓ FeatureEngineer instantiated successfully")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Refactoring successful!")
    print("="*60)
    print("\nYou can now run training with:")
    print("  python train_model.py --model linear --features min")
    print("  python train_model.py --model tree --features min")
    print("  python train_model.py --model neural --features min")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease check the error above and fix any import issues.")
