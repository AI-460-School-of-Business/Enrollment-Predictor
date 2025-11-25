# Testing Guide for Refactored Enrollment Predictor

## Overview
The training code has been refactored into a modular structure:

```
server/ml/
├── train_model.py              # Main CLI entry point
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── base_predictor.py      # Base class
│   ├── linear_predictor.py    # Linear regression
│   ├── tree_predictor.py      # Random Forest
│   └── neural_predictor.py    # Neural Network
├── data/                       # Data loading & feature engineering
│   ├── __init__.py
│   ├── data_loader.py         # Database extraction
│   └── feature_engineer.py    # Feature preparation
└── utils/                      # Utility functions
    ├── __init__.py
    ├── db_config.py           # Database configuration
    └── evaluation.py          # Model evaluation metrics
```

## Quick Test

### 1. Test Module Imports
From the `server/ml` directory:

```bash
cd c:\Development\ccsu_enrollment\Enrollment-Predictor\server\ml
python test_imports.py
```

This will verify all modules can be imported correctly.

### 2. Test Individual Model Training

#### Linear Regression (Fastest - good for testing)
```bash
python train_model.py --model linear --features min
```

#### Random Forest
```bash
python train_model.py --model tree --features min
```

#### Neural Network
```bash
python train_model.py --model neural --features min
```

### 3. Test with Rich Features
```bash
python train_model.py --model linear --features rich
python train_model.py --model tree --features rich
python train_model.py --model neural --features rich
```

### 4. Test with Custom SQL Query

#### Interactive Query Mode
```bash
python train_model.py --model linear --features min --interactive-query
```

#### Inline Query
```bash
python train_model.py --model linear --features min --data-query "SELECT * FROM section_detail_report_sbussection_detail_report_sbus WHERE (term / 100) >= 2024"
```

## Expected Output

A successful training run should show:

1. **Data Loading Phase**
   - Table discovery
   - Primary table identification
   - Data extraction and joining

2. **Feature Preparation Phase**
   - Feature selection
   - Data cleaning
   - Feature analysis

3. **Training Phase**
   - Model-specific training (hyperparameter tuning for linear/tree, epochs for neural)
   - Validation metrics (R², MAPE, etc.)

4. **Evaluation Phase**
   - Per-course accuracy report
   - Top 10 most predictable courses
   - Bottom 10 least predictable courses

5. **Save Prompt**
   - Option to save trained model

## Validation Checklist

- [ ] All imports work (`test_imports.py` passes)
- [ ] Linear model trains successfully
- [ ] Tree model trains successfully
- [ ] Neural model trains successfully
- [ ] Models can be saved and filenames are correct
- [ ] Per-course accuracy CSV files are generated in `test_results/`
- [ ] Validation metrics are reasonable (R² > 0.5, MAPE < 50%)

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, make sure you're running from the correct directory:
```bash
cd c:\Development\ccsu_enrollment\Enrollment-Predictor\server\ml
python train_model.py --model linear --features min
```

### Database Connection Issues
Check your environment variables:
- `DB_HOST`
- `DB_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`

### Missing Dependencies
Install required packages:
```bash
pip install -r ../requirements.txt
```

## Testing in Docker

If using Docker:

```bash
# From the Enrollment-Predictor directory
docker-compose up -d

# Execute training inside the container
docker-compose exec app python server/ml/train_model.py --model linear --features min
```

## Next Steps

After verifying the refactoring works:

1. Delete `train_model_old.py` (backup of original file)
2. Update any documentation that references the old structure
3. Update any CI/CD pipelines if applicable
4. Consider adding unit tests for individual components

## Benefits of This Refactoring

✅ **Separation of Concerns** - Each file has a single, clear responsibility
✅ **Easier Testing** - Components can be tested independently
✅ **Better Reusability** - Classes can be imported and used elsewhere
✅ **Cleaner Code** - Much easier to navigate and understand
✅ **Easier Maintenance** - Changes to one component don't affect others
✅ **All Functionality Preserved** - Linear, Tree, and Neural models all work identically
