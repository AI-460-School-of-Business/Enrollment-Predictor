# Machine Learning Module

This module implements a **three-tiered machine learning approach** for enrollment prediction using automated data discovery and feature engineering.

## Architecture Overview

### 🏗️ **Three-Tiered ML Approach**

Our system implements three complementary machine learning models, each with different strengths:

1. **Linear Regression** (`train_model.py --model linear`)
   - **Approach**: Ridge regression with hyperparameter tuning
   - **Strengths**: Fast training, interpretable coefficients, baseline performance
   - **Use Case**: Quick predictions and understanding linear relationships

2. **Tree-Based Models** (`train_model.py --model tree`)
   - **Approach**: Random Forest with grid search optimization
   - **Strengths**: Handles non-linear relationships, feature importance analysis, robust to outliers
   - **Use Case**: Complex pattern recognition and feature interaction discovery

3. **Neural Networks** (`train_model.py --model neural`)
   - **Approach**: Deep learning with dropout and early stopping
   - **Strengths**: Captures complex non-linear patterns, automatic feature combinations
   - **Use Case**: Maximum predictive accuracy for complex enrollment dynamics

### 🗄️ **Database Integration**

#### **Automated Data Discovery**
The system automatically:
- **Discovers all tables** in the PostgreSQL database
- **Identifies key columns** (CRN, Semester, Year, Enrollment) across tables
```
Primary Table (Enrollments)
    ↓ AUTO-JOIN via (Year, Semester) ↓
Related Tables:
    ├── Course Information
    ├── Student Demographics  
    ├── Faculty Data
    └── Financial Context
```

#### **Feature Engineering Pipeline**
1. **Column Detection**: Automatically finds CRN, semester, year, and enrollment columns
2. **Data Joining**: Left joins all related tables on temporal keys
3. **Feature Flattening**: Converts nested table structure into ML-ready format
4. **Type Inference**: Handles categorical vs numerical features automatically
5. **Preprocessing**: Scales numerical features, encodes categorical variables

### 📊 **Feature Schema System**

#### **Two Feature Modes**

**Minimal Features (`--features min`)**
- Core identifiers: CRN, Semester, Year
- Fast training, baseline performance
- All available database columns
- Automatic feature selection and filtering
The system creates `enrollment_features_auto.json` by:
- Analyzing actual database structure
- Documenting all available fields

```mermaid
Database → Discovery → Coalescing → Feature Engineering → Model Training → Prediction
```

1. **Data Extraction**: Query all tables from PostgreSQL
2. **Structure Analysis**: Identify table relationships and key columns
3. **Automatic Joining**: Coalesce data using (Year, Semester) connections
4. **Feature Preparation**: Clean, scale, and encode features
5. **Model Training**: Train with hyperparameter optimization
6. **Model Persistence**: Save to `/data/prediction_models/`

### 🎯 **Prediction Workflow**

**Input**: `CRN`, `Semester`, `Year`
**Process**: 
1. Look up historical data for the course/semester combination
2. Extract all related features from joined tables
3. Apply trained model preprocessing
4. Generate enrollment prediction

**Output**: Predicted enrollment count with confidence metrics

### 📁 **File Structure**

```
ml/
├── train_model.py          # Main training script (all 3 models)
├── train_tree_model.py     # Specialized tree model training
├── train_neural_model.py   # Specialized neural network training
├── feature_schema/         # Feature definitions
│   ├── enrollment_features_min.json    # Minimal feature set
│   ├── enrollment_features_rich.json   # Comprehensive features
│   └── enrollment_features_auto.json   # Auto-generated from DB
└── models/                 # Saved models (deprecated, now uses /data/prediction_models/)
```

### 🚀 **Usage Examples**

```bash
# Train linear model with minimal features
python server/ml/train_model.py --model linear --features min

# Train random forest with all available features
python server/ml/train_model.py --model tree --features rich

# Train neural network for maximum accuracy
python server/ml/train_model.py --model neural --features rich
```

### 📈 **Performance Characteristics**

| Model Type | Training Speed | Prediction Speed | Interpretability | Accuracy Potential |
|------------|---------------|------------------|------------------|-------------------|
| Linear     | Fast          | Fastest          | High            | Baseline          |
| Tree       | Medium        | Fast             | Medium          | Good              |
| Neural     | Slow          | Medium           | Low             | Highest           |

### 🔧 **Technical Features**

- **Automatic hyperparameter tuning** with cross-validation
- **Feature importance analysis** (tree models)

Models automatically adapt to your database structure:
- **No manual feature engineering** required
- **Automatic schema discovery** from actual data
