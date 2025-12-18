# Demand Forecasting

Machine learning and time series models for retail demand forecasting using point-of-sale (POS) data.

## Overview

This repository implements multiple approaches to demand forecasting for retail products:

1. **Baseline Machine Learning Models** - Traditional econometric and ML approaches
2. **Foundation Time Series Models** - State-of-the-art deep learning time series models

The codebase has been refactored from a monolithic script into modular, reusable components.

## Project Structure

```
DemandForecasting/
├── Baseline_Models/           # Main codebase
│   ├── baseline_demand_train.py         # Train baseline models (Linear, Logit, LightGBM, RF, Mixed Effects)
│   ├── foundation_ts_models.py          # Train foundation models (TimesFM, SunDial, TabPFN)
│   ├── baseline_demand.py               # Legacy monolithic script
│   │
│   ├── data_loader.py                   # Data loading and preprocessing
│   ├── feature_engineering.py           # Feature creation
│   ├── data_preparation.py              # Data prep for different model types
│   ├── evaluation_metrics.py            # Model evaluation
│   ├── model_training.py                # Model classes
│   │
│   ├── output/                          # Results directory
│   │   ├── baseline_results.json
│   │   └── foundation_ts_results.json
│   │
│   ├── README.md                        # Detailed documentation
│   └── REFACTORING_SUMMARY.md          # Refactoring details
│
├── Data/                      # Data directory (not tracked in git)
│   ├── salty_snack_0.05_store.csv
│   └── prod_saltsnck.xls
│
└── CLAUDE.md                  # Claude Code guidance
```

## Quick Start

### Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn lightgbm statsmodels openpyxl

# Optional: Foundation time series models
pip install torch transformers chronos-forecasting tabpfn tqdm
```

### Running Models

```bash
cd Baseline_Models

# Train baseline models
python baseline_demand_train.py

# Train foundation time series models
python foundation_ts_models.py
```

## Models Implemented

### Baseline Models
- **Linear Regression (OLS)** - Standard linear regression
- **Linear Regression with Product Fixed Effects** - Controls for unobserved product heterogeneity
- **Homogeneous Logit** - Discrete choice model with market shares and outside option
- **Mixed Effects Model** - Random price slopes by product (product-specific price elasticities)
- **LightGBM** - Gradient boosting machine
- **Random Forest** - Ensemble of decision trees

### Foundation Time Series Models
- **TimesFM 2.0** - Google's foundation model for time series forecasting
- **SunDial/Chronos** - Amazon's time series foundation model
- **TabPFN-TS** - Tabular Prior-Data Fitted Network for time series

## Key Features

### Dual Data Preparation Pipeline
The codebase supports two distinct modeling approaches:

**Standard Models** (Linear, Tree-based, Mixed Effects):
- Target: `logunits` (log of units sold)
- Direct demand prediction
- Functions: `prepare_standard_model_data()`

**Choice Models** (Logit):
- Target: `sharedp` (log(share) - log(outside_share))
- Market share-based approach with outside option
- Additional calculations: market potential, shares
- Functions: `prepare_choice_model_data()`

### Comprehensive Evaluation
All models evaluated using:
- **RMSE** - Root Mean Squared Error
- **R²** - Coefficient of determination
- **MAPE** - Mean Absolute Percentage Error
- **WMAPE** - Revenue-weighted MAPE
- **MPE** - Mean Percentage Error (bias)
- **Counterfactual Validity** - % predictions violating economic intuition (price decrease should increase demand)

## Data

The models expect retail POS data with:
- **Sales data**: store, week, product, units, price
- **Product attributes**: brand, package, flavor, fat content, etc.
- **Promotions**: feature, display, price reduction

**Validation Split**: Time-based (last 52 weeks)
- Training: weeks 1114-1374
- Validation: weeks 1375-1426

## Documentation

- **[Baseline_Models/README.md](Baseline_Models/README.md)** - Detailed usage guide, API reference, examples
- **[Baseline_Models/REFACTORING_SUMMARY.md](Baseline_Models/REFACTORING_SUMMARY.md)** - Refactoring process and decisions
- **[CLAUDE.md](CLAUDE.md)** - Guidance for Claude Code when working in this repository

## Results

Results are saved to `Baseline_Models/output/`:
- `baseline_results.json` - Baseline model metrics and comparison
- `foundation_ts_results.json` - Foundation model results

## Use Cases

This codebase is designed for:
- **Demand forecasting** for retail products
- **Price elasticity** estimation
- **Promotion effectiveness** analysis
- **Model comparison** across traditional ML and foundation models
- **Research** in retail analytics and demand modeling

## Author

Original code by h_min
Refactored version: 2023
