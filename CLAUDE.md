# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demand forecasting codebase for retail/grocery products using point-of-sale (POS) data. The project implements and compares multiple approaches:

1. **Baseline Models** - Traditional ML models (Linear Regression, XGBoost, Random Forest, Mixed Effects, Logit)
2. **Foundation Time Series Models** - Modern deep learning models (TimesFM, SunDial/Chronos, TabPFN-TS)

The code has been refactored from a monolithic script (`baseline_demand.py`) into modular components located in `Baseline_Models/`.

## Core Architecture

### Data Pipeline Architecture

The codebase supports two distinct modeling approaches with different data structures:

**Standard Models (OLS, XGBoost, Random Forest, Mixed Effects)**
- Target variable: `logunits` (log of units sold)
- Features: logprice, promotions, product attributes, store/week fixed effects
- Data preparation: `prepare_standard_model_data()` in `baseline_demand_train.py`

**Choice Models (Logit)**
- Target variable: `sharedp` = log(share) - log(outside_share)
- Features: Same as standard models
- Additional variables: market potential, shares, outside share
- Data preparation: `prepare_choice_model_data()` in `baseline_demand_train.py`

**Foundation Time Series Models**
- Uses panel format: store-product-week level aggregation
- Context-based forecasting with historical windows
- No explicit feature engineering - models learn patterns from raw time series

### Module Structure

**Core modules in `Baseline_Models/`:**
- `data_loader.py` - Load CSV, filter stores, handle missing values
- `feature_engineering.py` - Create dummy variables and feature matrices
- `data_preparation.py` - Prepare data for different model types (standard vs choice)
- `evaluation_metrics.py` - RMSE, R², MAPE, WMAPE, MPE, counterfactual validity
- `model_training.py` - Model classes (LinearRegressionModel, XGBoostModel, etc.)

**Main scripts:**
- `baseline_demand_train.py` - Trains baseline models on `salty_snack_0.05_store.csv`
- `foundation_ts_models.py` - Trains foundation time series models
- `baseline_demand.py` - Original monolithic script (legacy, not actively used)

### Data Requirements

**Expected CSV columns:**
- Store/time: `iri_key`, `week`, `numweek`
- Product: `colupc` (UPC code)
- Target: `units`, `logunits`
- Pricing: `price`, `logprice`
- Product attributes: `brand`, `package`, `flavorscent`, `fatcontent`, `cookingmethod`, `saltsodiumcontent`, `typeofcut`
- Promotions: `f` (feature), `d` (display), `pr` (price reduction)
- Volume: `vol_eq` (volume equivalent)

**Data splits:**
- `baseline_demand.py` (legacy): training < week 285, test >= 285
- `baseline_demand_train.py`: training weeks 1114-1374, validation weeks 1375-1426
- Split is time-based (temporal validation)

## Common Development Commands

### Running Models

```bash
# Train baseline models (Linear, Logit, XGBoost, RF, Mixed Effects)
cd Baseline_Models
python baseline_demand_train.py

# Train foundation time series models (TimesFM, SunDial, TabPFN-TS)
python foundation_ts_models.py

# Run legacy monolithic script
python baseline_demand.py
```

### Testing

Currently no test files exist despite references in README. If tests need to be created:
```bash
python -m unittest discover -s Baseline_Models -p "test_*.py"
```

## Key Implementation Details

### Missing Value Handling
- `fatcontent`: NaN → "REGULAR"
- `cookingmethod`: NaN → "MISSING"
- `saltsodiumcontent`: NaN → "MISSING"
- Other categorical columns: NaN → "MISSING"

### Feature Engineering
- All categorical variables converted to dummy variables with `drop_first=True`
- Continuous features: logprice, vol_eq
- Fixed effects: store (iri_key) and week (numweek) dummies

### Market Potential Calculation (for Logit models)
```python
market_potential = 3 × max(total_equivalent_units_per_store)
share = eq_units / market_potential
outside_share = 1 - (total_eq_units / market_potential)
sharedp = log(share) - log(outside_share)
```

### Counterfactual Validity
Tests economic intuition: reducing price should increase demand.
- Creates counterfactual with price × 0.9
- Measures % of predictions where demand decreased (invalid)
- For standard models: randomly select one product per store-week
- For choice models: specialized calculation using share-based predictions

### Mixed Effects Model
- Random intercept by product
- Random slope for logprice by product (price elasticity varies by product)
- Uses `statsmodels.regression.mixed_linear_model.MixedLM`
- Fallback to OLS with product dummies if convergence fails

## Configuration

Configuration is embedded in main scripts (not in separate config files):

**`baseline_demand_train.py`:**
```python
CONFIG = {
    'data_path': '../Data/salty_snack_0.05_store.csv',
    'product_path': '../Data/prod_saltsnck.xls',
    'output_dir': 'output',
    'val_week_start': 1375,
    'market_potential_multiplier': 3.0,
}

MODEL_CONFIGS = {
    'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    'random_forest': {'n_estimators': 100, 'max_depth': 10},
    # ... other model configs
}
```

**`foundation_ts_models.py`:**
```python
CONFIG = {
    'data_path': '../Data/salty_snack_0.05_store.csv',
    'val_week_start': 1375,
    'forecast_horizon': 52,
    'context_length': 104,
}
```

## Output Files

Results are saved to `Baseline_Models/output/`:
- `baseline_results.json` - Baseline model metrics and comparison
- `foundation_ts_results.json` - Foundation model metrics and comparison

Format includes:
- Individual model metrics (RMSE, R², MAPE, WMAPE, MPE, counterfactual_validity)
- Comparison table sorted by WMAPE
- Best model identification
- Configuration and timestamp

## Dependencies

**Required packages:**
```
pandas
numpy
scikit-learn
xgboost  # or lightgbm (baseline_demand_train.py uses lightgbm)
statsmodels
openpyxl  # for reading .xls product data
```

**Foundation models (optional):**
```
torch
transformers
chronos-forecasting  # for SunDial/Chronos
tabpfn  # for TabPFN-TS
tqdm
```

## Important Notes

### When Adding New Models

1. **For baseline models:** Create class inheriting from `BaselineModel` in `model_training.py`
2. **For standard models:** Use `train_and_evaluate_standard_model()` helper
3. **For choice models:** Use `train_and_evaluate_choice_model()` helper
4. **For mixed effects:** Follow pattern in `train_evaluate_mixed_effects_model()`

### When Modifying Data Preparation

- Ensure consistency between `prepare_standard_model_data()` and `prepare_choice_model_data()`
- Maintain the temporal split (training before validation week)
- Keep the two-dataframe structure (standard vs choice) distinct

### Model Evaluation

All models evaluated on same metrics for comparison:
- RMSE, R² evaluated in **actual space** (not log space)
- Predictions in log space must be converted: `np.exp(y_pred)`
- Weights for WMAPE: `actual_units × price` (revenue weighting)
- Counterfactual validity measures % invalid predictions (lower is better)

### Path Assumptions

- Data files expected in `../Data/` relative to `Baseline_Models/`
- Output saved to `Baseline_Models/output/`
- Scripts should be run from `Baseline_Models/` directory

### Legacy Code

`baseline_demand.py` is the original monolithic script (454 lines). It's kept for reference but superseded by the refactored modular code. Use `baseline_demand_train.py` instead.
