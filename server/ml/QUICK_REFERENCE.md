# Quick Reference - Refactored Training System

## Directory Structure
```
server/ml/
├── train_model.py          # 🚀 Main entry point
├── models/                 # 🤖 ML Models
│   ├── base_predictor.py
│   ├── linear_predictor.py
│   ├── tree_predictor.py
│   └── neural_predictor.py
├── data/                   # 📊 Data Pipeline
│   ├── data_loader.py
│   └── feature_engineer.py
└── utils/                  # 🛠️ Utilities
    ├── db_config.py
    └── evaluation.py
```

## Quick Commands

### Test Imports
```bash
cd c:\Development\ccsu_enrollment\Enrollment-Predictor\server\ml
python test_imports.py
```

### Train Models
```bash
# Linear (fastest - use for testing)
python train_model.py --model linear --features min

# Random Forest (slower, more accurate)
python train_model.py --model tree --features min

# Neural Network (slowest, most complex)
python train_model.py --model neural --features min
```

### Rich Features
```bash
python train_model.py --model linear --features rich
python train_model.py --model tree --features rich
python train_model.py --model neural --features rich
```

### Custom Queries
```bash
# Interactive mode
python train_model.py --model linear --features min --interactive-query

# Inline query
python train_model.py --model tree --features min --data-query "SELECT * FROM table WHERE year >= 2024"
```

## Import in Other Scripts

```python
# Import models
from models.linear_predictor import LinearRegressionPredictor
from models.tree_predictor import TreePredictor
from models.neural_predictor import NeuralNetworkPredictor

# Import data pipeline
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer

# Import utilities
from utils.evaluation import analyze_per_course_accuracy
from utils.db_config import DB_CONFIG

# Use them
loader = DataLoader()
data = loader.extract_training_data()

engineer = FeatureEngineer("min")
X, y = engineer.prepare_features(data)

model = LinearRegressionPredictor("min")
results = model.train(X, y)
```

## Key Classes

### Models
- `EnrollmentPredictor` - Base class (in base_predictor.py)
- `LinearRegressionPredictor` - Linear regression with Ridge
- `TreePredictor` - Random Forest with grid search
- `NeuralNetworkPredictor` - TensorFlow/Keras neural network

### Data
- `DataLoader` - Extracts data from database
- `FeatureEngineer` - Prepares features for training

### Functions
- `analyze_per_course_accuracy()` - Per-course metrics

## Expected Workflow

1. **Data Loading** → `DataLoader`
   - Discovers tables
   - Joins related data
   - Filters to 2023+

2. **Feature Preparation** → `FeatureEngineer`
   - Selects features (min/rich)
   - Cleans data
   - Analyzes features

3. **Training** → Model classes
   - Preprocesses features
   - Trains model
   - Evaluates performance

4. **Evaluation** → `analyze_per_course_accuracy()`
   - Per-course metrics
   - Generates reports
   - Saves CSV

5. **Saving** → Model's `save_model()`
   - Saves trained model
   - Saves preprocessing info

## Files to Review

- 📖 `TESTING.md` - Full testing guide
- 📖 `REFACTORING_SUMMARY.md` - What changed
- 🧪 `test_imports.py` - Quick validation
- 📝 `README.md` - Original documentation

## Troubleshooting

**Import errors?** Make sure you're in the `server/ml` directory

**Database connection?** Check your environment variables

**Missing packages?** Run: `pip install -r ../requirements.txt`

## Success Indicators

✓ `test_imports.py` runs without errors
✓ Training completes successfully
✓ Per-course accuracy CSV generated
✓ Model can be saved
✓ Metrics look reasonable (R² > 0.5, MAPE < 50%)
