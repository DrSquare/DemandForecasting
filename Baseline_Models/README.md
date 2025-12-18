# Baseline Demand Estimation - Refactored

A modular and reusable Python package for baseline demand estimation using various machine learning models. This refactored version separates concerns into distinct modules, making the code more maintainable, testable, and extensible.

## 📁 Project Structure

```
Baseline_Models/
├── baseline_demand.py                 # Original monolithic script (legacy)
├── baseline_demand_train.py           # Main training script for baseline models
├── foundation_ts_models.py            # Foundation time series models (TimesFM, SunDial, TabPFN)
│
├── data_loader.py                     # Data loading and preprocessing
├── feature_engineering.py             # Feature creation and dummy variables
├── data_preparation.py                # Data prep for different model types
├── evaluation_metrics.py              # Model evaluation functions
├── model_training.py                  # Model classes and training logic
│
├── data_check.py                      # Data validation utilities
├── generate_stats.py                  # Statistics generation
│
├── output/                            # Results directory
│   ├── baseline_results.json          # Baseline model results
│   └── foundation_ts_results.json     # Foundation model results
│
├── README.md                          # This file
└── REFACTORING_SUMMARY.md            # Detailed refactoring notes
```

## 🎯 Key Features

### 1. **Modular Architecture**
- **data_loader.py**: Handles CSV loading, store filtering, and missing value imputation
- **feature_engineering.py**: Creates dummy variables and constructs feature matrices
- **data_preparation.py**: Prepares data for two distinct model types:
  - **Standard models**: Use log(units) as target variable
  - **Choice models**: Use share difference (log(share) - log(outside_share)) as target
- **evaluation_metrics.py**: Provides RMSE, R², MAPE, WMAPE, MPE, and counterfactual validity metrics
- **model_training.py**: Encapsulates model classes and training logic

### 2. **Two DataFrame Structures**

#### Standard Model Data Structure
- **Target**: `logunits` (log of units sold)
- **Features**: Price, promotions, product attributes, store/week fixed effects
- **Use case**: OLS, XGBoost, Random Forest, Mixed Effects models

#### Choice Model Data Structure
- **Target**: `sharedp` (log(share) - log(outside_share))
- **Features**: Same as standard models
- **Additional variables**: Market potential, shares, outside share
- **Use case**: Discrete choice models (Logit)

### 3. **Supported Models**

**Baseline Models (baseline_demand_train.py):**
- Linear Regression (OLS)
- Linear Regression with Product Fixed Effects
- Homogeneous Logit (with outside share)
- Mixed Effects Model (random price slopes by product)
- LightGBM
- Random Forest Regressor

**Foundation Time Series Models (foundation_ts_models.py):**
- TimesFM 2.0 (Google's foundation model)
- SunDial/Chronos (Amazon's time series model)
- TabPFN-TS (Tabular Prior-Data Fitted Network)

### 4. **Comprehensive Evaluation**
All models are evaluated using:
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **WMAPE**: Weighted MAPE (revenue-weighted)
- **MPE**: Mean Percentage Error (measures bias)
- **Counterfactual Validity**: % of predictions violating economic intuition

## 🚀 Quick Start

### Installation

```bash
# Core dependencies for baseline models
pip install pandas numpy scikit-learn lightgbm statsmodels openpyxl

# Optional: Foundation time series models
pip install torch transformers chronos-forecasting tabpfn tqdm
```

### Basic Usage

```bash
# Run baseline models pipeline
python baseline_demand_train.py

# Run foundation time series models
python foundation_ts_models.py

# Run legacy monolithic script (for reference)
python baseline_demand.py
```

**Programmatic usage:**

```python
# Import and run baseline models
from baseline_demand_train import main

results = main()
# Results saved to output/baseline_results.json
```

### Using Individual Modules

```python
# Example: Load and preprocess data
from data_loader import load_data, filter_training_stores, handle_missing_values

df = load_data('salty_snack_0.05_store.csv')
df = filter_training_stores(df, train_week_cutoff=1375)
df = handle_missing_values(df)

# Example: Train a specific baseline model
from baseline_demand_train import (
    prepare_standard_model_data,
    train_evaluate_standard_model
)
from model_training import LinearRegressionModel

# Prepare data
standard_data = prepare_standard_model_data(
    df,
    val_week_start=1375,
    include_product_fe=False
)

# Train model
model = LinearRegressionModel(fit_intercept=True)
results = train_evaluate_standard_model(model, standard_data, in_log_space=True)

print(results['metrics'])
```

### Prepare Choice Model Data

```python
from baseline_demand_train import prepare_choice_model_data, train_evaluate_choice_model
from model_training import HomogeneousLogitModel

choice_data = prepare_choice_model_data(
    df,
    val_week_start=1375,
    market_potential_multiplier=3.0
)

# Train logit model
logit_model = HomogeneousLogitModel(fit_intercept=True)
logit_results = train_evaluate_choice_model(logit_model, choice_data)
print(logit_results['metrics'])
```

## 🧪 Testing

**Note:** Unit tests have not yet been implemented. The refactoring created the modular structure to support testing, but test files are not currently present.

To add tests in the future:

```bash
# Run all tests (when implemented)
python -m unittest discover -s . -p "test_*.py"

# Run specific test module
python -m unittest test_data_loader
```

Refer to `REFACTORING_SUMMARY.md` for the planned test structure.

## 📊 Data Requirements

The codebase expects two data files in the `../Data/` directory:

1. **Sales Data**: `salty_snack_0.05_store.csv`
2. **Product Attributes**: `prod_saltsnck.xls`

### Sales CSV Required Columns
- `iri_key` or `IRI_KEY`: Store identifier
- `week` or `WEEK`: Week number
- `colupc` or `COLUPC`: Product UPC code
- `units` or `UNITS`: Units sold
- `price` or `PRICE`: Price per unit

### Sales CSV Optional Columns
- `numweek`: Sequential week number (created if not present)
- `logprice`: Log of price (calculated if not present)
- `logunits`: Log of units (calculated if not present)
- `vol_eq`: Volume equivalent (defaults to 1.0 if missing)

### Product Attributes (from Excel file)
- `brand` or `L5`: Brand name
- `package` or `PACKAGE`: Package type
- `flavorscent` or `FLAVOR/SCENT`: Flavor/scent
- `fatcontent` or `FAT CONTENT`: Fat content
- `cookingmethod` or `COOKING METHOD`: Cooking method
- `saltsodiumcontent` or `SALT/SODIUM CONTENT`: Salt/sodium content
- `typeofcut` or `TYPE OF CUT`: Type of cut

### Promotions (from sales data)
- `f` or `F`: Feature promotion (0/1 or categorical)
- `d` or `D`: Display promotion (0/1 or categorical)
- `pr` or `PR`: Price reduction (0/1 or categorical)

### Data Splits
- **Legacy script** (`baseline_demand.py`): train < week 285, test ≥ 285
- **Current scripts**: train weeks 1114-1374, validation weeks 1375-1426 (last 52 weeks)

## 🔧 Customization

### Adding a New Model

```python
from model_training import BaselineModel
import numpy as np

class MyCustomModel(BaselineModel):
    def __init__(self, **kwargs):
        super().__init__("My Custom Model")
        # Initialize your model
        self.model = YourModelClass(**kwargs)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

### Adding a New Evaluation Metric

```python
# In evaluation_metrics.py
def my_custom_metric(actual, forecast):
    # Your metric calculation
    return metric_value

# Update evaluate_model function to include it
```

### Customizing Feature Engineering

```python
from feature_engineering import FeatureEngineer

# Extend the class
class CustomFeatureEngineer(FeatureEngineer):
    def create_custom_features(self):
        # Add custom feature creation logic
        pass
```

## 📈 Example Output

```
================================================================================
BASELINE DEMAND ESTIMATION - SALTY SNACK DATA
================================================================================

Configuration:
  Data: ../Data/salty_snack_0.05_store.csv
  Validation week start: 1375
  Market potential multiplier: 3.0

================================================================================
[1/5] Loading and preprocessing data...
================================================================================
Loading sales data...
  Sales data shape: (156789, 12)
  Week range: 1114 - 1426
Loading product attributes...
  Product data shape: (234, 10)

Data after preprocessing:
  Shape: (156789, 22)
  Unique stores: 48
  Unique products: 187
  Week range: 1114 - 1426

================================================================================
[2/5] Preparing data for standard models (log units target)...
================================================================================
Training set: (98765, 342)
Validation set: (24691, 342)
Train weeks: 1114 - 1374
Val weeks: 1375 - 1426

================================================================================
[3/5] Preparing data for choice models (share difference target)...
================================================================================
Choice model training set: (98765, 342)
Choice model validation set: (24691, 342)

================================================================================
[4/5] Training and evaluating models...
================================================================================

--- Training Linear Regression (OLS) ---
================================================================================
Linear Regression (OLS) Performance Metrics
================================================================================
RMSE                          : 12.345678
R²                            : 0.856234
MAPE                          : 0.123456
WMAPE                         : 0.098765
MPE                           : -0.012345
Counterfactual Validity       : 0.0234
================================================================================

--- Training Mixed Effects Model (Random Price Slopes) ---
  Model: Random price coefficient by product
  Building feature matrix...
  Training features: 4
  Training samples: 98765
  Number of products (groups): 187
  Fitting Mixed Effects model...
  Model converged: True

================================================================================
[5/5] Comparing all models...
================================================================================
MODEL COMPARISON (sorted by WMAPE)
================================================================================
                                     rmse        r2      mape     wmape       mpe    counterfactual_validity
LightGBM                          9.876543  0.902341  0.089765  0.076543 -0.008876                  0.0156
Random Forest                    10.234567  0.894321  0.095432  0.082109 -0.009321                  0.0178
Mixed Effects (Random Slopes)    10.456789  0.887654  0.098234  0.083456 -0.009654                  0.0145
Linear Regression (Product FE)   11.123456  0.878901  0.105678  0.089876 -0.010234                  0.0198
Homogeneous Logit                11.987654  0.865432  0.112345  0.095432 -0.011123                  0.0212
Linear Regression                12.345678  0.856234  0.123456  0.098765 -0.012345                  0.0234
================================================================================

Best model: LightGBM
WMAPE: 0.0765
R²: 0.9023

Results saved to: output/baseline_results.json

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

## 📦 Output Files

Results are saved to the `output/` directory in JSON format:

### baseline_results.json
Contains results from `baseline_demand_train.py`:
- Model configurations
- Individual model metrics (RMSE, R², MAPE, WMAPE, MPE, counterfactual validity)
- Comparison table sorted by WMAPE
- Best model identification
- Timestamp

### foundation_ts_results.json
Contains results from `foundation_ts_models.py`:
- Foundation model forecasts and metrics
- Comparison with baseline models
- Note: Counterfactual validity is NaN for time series models (not causal)

## 🤝 Contributing

Contributions are welcome! Please:
1. Follow the existing modular code structure
2. Document new functions with docstrings and type hints
3. Update this README when adding new features
4. Consider adding unit tests for new functionality

## 📝 Notes

### Key Differences from Original Code

1. **Modularity**: Code is separated into logical modules instead of one long script
2. **Reusability**: Functions can be imported and used independently
3. **Testability**: Comprehensive unit tests for all modules
4. **Documentation**: Clear docstrings and type hints
5. **Maintainability**: Easier to update and extend individual components
6. **Two Data Structures**: Explicitly handles both standard and choice model data preparation

### Why Two DataFrame Structures?

- **Standard models** (OLS, XGBoost, Random Forest) predict demand directly using log(units)
- **Choice models** (Logit) model consumer choice using shares and outside option
- Each requires different target variables and additional preprocessing
- The modular design makes it easy to prepare data for either approach

## 📄 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Original code by h_min
Refactored version: 2023
