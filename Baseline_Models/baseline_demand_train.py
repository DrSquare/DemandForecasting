"""
Baseline demand estimation training script.
Uses salty_snack_0.05_store.csv with time-based validation split.

Validation: weeks 1375-1426 (last 52 weeks)
Training: weeks 1114-1374 (first 261 weeks)
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Tuple, List

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import handle_missing_values
from data_preparation import (
    calculate_equivalent_units,
    calculate_market_potential,
    add_share_variables,
    create_counterfactual_features
)
from feature_engineering import FeatureEngineer
from model_training import (
    LinearRegressionModel, HomogeneousLogitModel,
    LightGBMModel, RandomForestModel, MixedEffectsModel,
    train_and_evaluate_standard_model,
    train_and_evaluate_choice_model
)
from evaluation_metrics import (
    print_metrics, compare_models, evaluate_model,
    counterfactual_validity_by_store_week,
    counterfactual_validity_choice_model_by_store_week,
    print_comparison_summary
)


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_path': '../Data/salty_snack_0.05_store.csv',
    'product_path': '../Data/prod_saltsnck.xls',
    'output_dir': 'output',
    'val_week_start': 1375,  # Validation starts at week 1375
    'market_potential_multiplier': 3.0,
    'verbose': True,
    'save_results': True,
}

# Model configurations
MODEL_CONFIGS = {
    'linear_regression': {},
    'linear_regression_pfe': {},
    'mixed_effects': {
        'group_col': 'colupc',
        'random_effect_vars': ['logprice']
    },
    'homogeneous_logit': {},
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbosity': -1
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
}


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_merge_data(data_path: str, product_path: str) -> pd.DataFrame:
    """
    Load sales data and merge with product attributes.
    
    Args:
        data_path: Path to salty_snack_0.05_store.csv
        product_path: Path to prod_saltsnck.xls
        
    Returns:
        Merged DataFrame with all required columns
    """
    print("Loading sales data...")
    df = pd.read_csv(data_path)
    
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"  Sales data shape: {df.shape}")
    print(f"  Week range: {df['WEEK'].min()} - {df['WEEK'].max()}")
    
    print("\nLoading product attributes...")
    prod_df = pd.read_excel(product_path)
    print(f"  Product data shape: {prod_df.shape}")
    
    # Rename product columns to match expected format
    prod_df = prod_df.rename(columns={
        'UPC': 'COLUPC',
        'VOL_EQ': 'vol_eq',
        'PRODUCT TYPE': 'producttype',
        'PACKAGE': 'package',
        'FLAVOR/SCENT': 'flavorscent',
        'FAT CONTENT': 'fatcontent',
        'COOKING METHOD': 'cookingmethod',
        'SALT/SODIUM CONTENT': 'saltsodiumcontent',
        'TYPE OF CUT': 'typeofcut',
        'L5': 'brand'
    })
    
    # Select only needed columns from product file
    prod_cols = ['COLUPC', 'vol_eq', 'producttype', 'package', 'flavorscent', 
                 'fatcontent', 'cookingmethod', 'saltsodiumcontent', 'typeofcut', 'brand']
    prod_df = prod_df[[c for c in prod_cols if c in prod_df.columns]]
    
    # Convert COLUPC to same type for merging
    df['COLUPC'] = df['COLUPC'].astype(str)
    prod_df['COLUPC'] = prod_df['COLUPC'].astype(str)
    
    # Merge sales with product attributes
    print("\nMerging sales with product attributes...")
    df = df.merge(prod_df, on='COLUPC', how='left')
    
    # Standardize column names to lowercase for compatibility with baseline modules
    df.columns = df.columns.str.lower()
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'iri_key': 'iri_key',
        'week': 'week',
        'colupc': 'colupc',
        'price': 'price',
        'units': 'units',
        'logprice': 'logprice',
    })
    
    # Create numweek (sequential week number starting from 1)
    min_week = df['week'].min()
    df['numweek'] = df['week'] - min_week + 1
    
    # Ensure logunits and logprice exist
    if 'logunits' not in df.columns:
        df['logunits'] = np.log(df['units'].clip(lower=1))
    if 'logprice' not in df.columns:
        df['logprice'] = np.log(df['price'].clip(lower=0.01))
    
    # Handle missing values in vol_eq
    if 'vol_eq' in df.columns:
        df['vol_eq'] = df['vol_eq'].fillna(1.0)
    else:
        df['vol_eq'] = 1.0  # Default volume equivalent
    
    print(f"\nMerged data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def filter_training_stores(df: pd.DataFrame, train_week_cutoff: int) -> pd.DataFrame:
    """
    Filter to stores that exist in training period.
    
    Args:
        df: Input DataFrame
        train_week_cutoff: Week where validation starts
        
    Returns:
        Filtered DataFrame
    """
    df_train = df[df['week'] < train_week_cutoff]
    store_list = df_train['iri_key'].unique()
    df_filtered = df[df['iri_key'].isin(store_list)]
    return df_filtered


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data: handle missing values, create derived columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Handle missing values in categorical columns
    missing_value_map = {
        "fatcontent": "REGULAR",
        "cookingmethod": "MISSING",
        "saltsodiumcontent": "MISSING",
        "flavorscent": "MISSING",
        "producttype": "MISSING",
        "package": "MISSING",
        "typeofcut": "MISSING",
        "brand": "MISSING"
    }
    
    for col, replacement in missing_value_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(replacement)
            df[col] = df[col].replace({np.nan: replacement, '': replacement, 0: replacement})
    
    return df


# ============================================================================
# Feature Matrix Building (adapted for new data format)
# ============================================================================

def build_feature_matrix_standard(df: pd.DataFrame, 
                                   include_product_fe: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix for standard models (log units target).
    
    Args:
        df: Input DataFrame with all columns
        include_product_fe: Whether to include product fixed effects
        
    Returns:
        Tuple of (X, y)
    """
    # Continuous variables
    logprice = df['logprice']
    
    # Promotion dummies
    f_dummies = pd.get_dummies(df['f'], prefix='F', drop_first=True)
    d_dummies = pd.get_dummies(df['d'], prefix='D', drop_first=True)
    pr_dummies = pd.get_dummies(df['pr'], prefix='PR', drop_first=True)
    
    # Fixed effects
    store_dummies = pd.get_dummies(df['iri_key'], prefix='STORE', drop_first=True)
    week_dummies = pd.get_dummies(df['numweek'], prefix='WK', drop_first=True)
    
    if include_product_fe:
        product_dummies = pd.get_dummies(df['colupc'])
        X = pd.concat([
            logprice,
            store_dummies,
            week_dummies,
            f_dummies,
            d_dummies,
            pr_dummies,
            product_dummies
        ], axis=1)
    else:
        # Product attributes
        brand_dummies = pd.get_dummies(df['brand'], prefix='BRAND', drop_first=True) if 'brand' in df.columns else pd.DataFrame()
        package_dummies = pd.get_dummies(df['package'], prefix='PKG', drop_first=True) if 'package' in df.columns else pd.DataFrame()
        flavor_dummies = pd.get_dummies(df['flavorscent'], prefix='FLV', drop_first=True) if 'flavorscent' in df.columns else pd.DataFrame()
        fat_dummies = pd.get_dummies(df['fatcontent'], prefix='FAT', drop_first=True) if 'fatcontent' in df.columns else pd.DataFrame()
        cook_dummies = pd.get_dummies(df['cookingmethod'], prefix='CM', drop_first=True) if 'cookingmethod' in df.columns else pd.DataFrame()
        salt_dummies = pd.get_dummies(df['saltsodiumcontent'], prefix='SALT', drop_first=True) if 'saltsodiumcontent' in df.columns else pd.DataFrame()
        cut_dummies = pd.get_dummies(df['typeofcut'], prefix='CUT', drop_first=True) if 'typeofcut' in df.columns else pd.DataFrame()
        
        voleq = df['vol_eq'] if 'vol_eq' in df.columns else pd.Series(1.0, index=df.index)
        
        components = [logprice, store_dummies, week_dummies, f_dummies, d_dummies, pr_dummies]
        for comp in [brand_dummies, voleq, package_dummies, flavor_dummies, fat_dummies, cook_dummies, salt_dummies, cut_dummies]:
            if isinstance(comp, pd.DataFrame) and len(comp.columns) > 0:
                components.append(comp)
            elif isinstance(comp, pd.Series):
                components.append(comp)
        
        X = pd.concat(components, axis=1)
    
    y = df['logunits']
    
    return X, y


def build_feature_matrix_choice(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix for choice models (share difference target).
    
    Args:
        df: Input DataFrame with share variables computed
        
    Returns:
        Tuple of (X, y)
    """
    # Same features as standard but different target
    X, _ = build_feature_matrix_standard(df, include_product_fe=False)
    y = df['sharedp']
    
    return X, y


# ============================================================================
# Data Preparation Functions
# ============================================================================

def prepare_standard_model_data(df: pd.DataFrame,
                                 val_week_start: int,
                                 include_product_fe: bool = False) -> Dict:
    """
    Prepare data for standard models.
    
    Args:
        df: Input DataFrame
        val_week_start: Week where validation starts
        include_product_fe: Whether to include product fixed effects
        
    Returns:
        Dictionary with train/test data
    """
    # Build feature matrix
    X, y = build_feature_matrix_standard(df, include_product_fe=include_product_fe)
    
    # Time-based split
    mask = df['week'] >= val_week_start
    
    X_train = X[~mask].copy()
    X_test = X[mask].copy()
    y_train = y[~mask].copy()
    y_test = y[mask].copy()
    
    df_train = df[~mask].copy()
    df_test = df[mask].copy()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'df_train': df_train,
        'df_test': df_test,
        'product_train': df[~mask]['colupc'].copy(),
        'product_test': df[mask]['colupc'].copy()
    }


def prepare_choice_model_data(df: pd.DataFrame,
                               val_week_start: int,
                               market_potential_multiplier: float = 3.0) -> Dict:
    """
    Prepare data for choice models.
    
    Args:
        df: Input DataFrame
        val_week_start: Week where validation starts
        market_potential_multiplier: Multiplier for market potential
        
    Returns:
        Dictionary with train/test data and share variables
    """
    df = df.copy()
    
    # Calculate equivalent units
    df['eq_units'] = df['units'] * df['vol_eq']
    
    # Split train/test first
    df_train_raw = df[df['week'] < val_week_start]
    
    # Calculate total units by store-week
    totalunits = df_train_raw.groupby(['iri_key', 'week'])['eq_units'].sum().reset_index()
    totalunits.rename(columns={'eq_units': 'total_eq_units'}, inplace=True)
    df_train_raw = df_train_raw.merge(totalunits, how='left', on=['iri_key', 'week'])
    
    # Calculate market potential from training data
    m_potential = df_train_raw.groupby(['iri_key'])['total_eq_units'].max().reset_index()
    m_potential['m_potential'] = market_potential_multiplier * m_potential['total_eq_units']
    m_potential.drop("total_eq_units", axis=1, inplace=True)
    
    # Add share variables to full dataset
    # First merge market potential
    df = df.merge(m_potential, how='left', on=['iri_key'])
    
    # Calculate total units by store-week for full data
    totalunits_full = df.groupby(['iri_key', 'week'])['eq_units'].sum().reset_index()
    totalunits_full.rename(columns={'eq_units': 'total_eq_units'}, inplace=True)
    df = df.merge(totalunits_full, how='left', on=['iri_key', 'week'])
    
    # Calculate shares
    df['share'] = df['eq_units'] / df['m_potential']
    df['share_within'] = df['eq_units'] / df['total_eq_units']
    df['outside_share'] = 1 - df['total_eq_units'] / df['m_potential']
    
    # Clip to avoid log(0)
    df['share'] = df['share'].clip(lower=1e-10)
    df['outside_share'] = df['outside_share'].clip(lower=1e-10)
    
    df['logshare'] = np.log(df['share'])
    df['logoutsideshare'] = np.log(df['outside_share'])
    df['sharedp'] = df['logshare'] - df['logoutsideshare']
    
    # Build feature matrix
    X, y = build_feature_matrix_choice(df)
    
    # Time-based split
    mask = df['week'] >= val_week_start
    
    X_train = X[~mask].copy()
    X_test = X[mask].copy()
    y_train = y[~mask].copy()
    y_test = y[mask].copy()
    
    df_train = df[~mask].copy()
    df_test = df[mask].copy()
    
    mp_test = df[mask]['m_potential'].copy()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'df_train': df_train,
        'df_test': df_test,
        'm_potential': m_potential,
        'mp_test': mp_test
    }


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_evaluate_standard_model(model, data_dict: Dict, in_log_space: bool = True) -> Dict:
    """
    Train and evaluate a standard (non-choice) model.
    
    Args:
        model: Model instance with fit/predict methods
        data_dict: Data dictionary from prepare_standard_model_data
        in_log_space: Whether predictions are in log space
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    # Train
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    # Predict
    y_pred = model.predict(data_dict['X_test'])
    
    # Convert from log space
    if in_log_space:
        actual = np.exp(data_dict['y_test'].values)
        forecast = np.exp(y_pred)
    else:
        actual = data_dict['y_test'].values
        forecast = y_pred
    
    # Calculate weights (revenue)
    weights = actual * data_dict['df_test']['price'].values
    
    # Evaluate basic metrics (without counterfactual)
    metrics = evaluate_model(actual, forecast, weights=weights, forecast_counterfactual=None)
    
    # Calculate counterfactual validity by store-week (one random product per store-week)
    cf_validity = counterfactual_validity_by_store_week(
        df_test=data_dict['df_test'],
        model=model,
        X_test=data_dict['X_test'],
        in_log_space=in_log_space,
        price_multiplier=0.9,
        random_seed=42
    )
    metrics['counterfactual_validity'] = cf_validity
    
    return {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics
    }


def train_evaluate_choice_model(model, data_dict: Dict) -> Dict:
    """
    Train and evaluate a choice model.
    
    Args:
        model: Model instance (HomogeneousLogitModel)
        data_dict: Data dictionary from prepare_choice_model_data
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    # Train
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    # Predict share difference
    y_pred_share = model.predict(data_dict['X_test'])
    
    # Convert to units using the model's predict_units method
    if hasattr(model, 'predict_units'):
        y_pred_units = model.predict_units(
            data_dict['X_test'],
            data_dict['df_test'],
            data_dict['mp_test']
        )
    else:
        # Manual conversion: share_ratio * outside_share * market_potential
        share_ratio = np.exp(y_pred_share)
        outside_share = data_dict['df_test']['outside_share'].values
        m_potential = data_dict['mp_test'].values
        y_pred_units = share_ratio * outside_share * m_potential
    
    # Actual units (from eq_units)
    actual = data_dict['df_test']['eq_units'].values
    
    # Weights (revenue)
    weights = actual * data_dict['df_test']['price'].values
    
    # Evaluate basic metrics (without counterfactual)
    metrics = evaluate_model(actual, y_pred_units, weights=weights, forecast_counterfactual=None)
    
    # Calculate counterfactual validity by store-week for choice model
    cf_validity = counterfactual_validity_choice_model_by_store_week(
        df_test=data_dict['df_test'],
        model=model,
        X_test=data_dict['X_test'],
        price_multiplier=0.9,
        random_seed=42
    )
    metrics['counterfactual_validity'] = cf_validity
    
    return {
        'model': model,
        'predictions': y_pred_units,
        'metrics': metrics
    }


def train_evaluate_mixed_effects_model(data_dict: Dict, 
                                        group_col: str = 'colupc',
                                        random_effect_vars: list = None) -> Dict:
    """
    Train and evaluate a Mixed Effects model with random price slopes by product.
    
    Model specification:
    - Fixed effects: logprice, promotion dummies
    - Random effects by store: intercept
    - Random effects by product: logprice slope (price sensitivity varies by product)
    
    Args:
        data_dict: Data dictionary from prepare_standard_model_data (with product FE)
        group_col: Column to group by (product ID)
        random_effect_vars: Variables with random slopes (default: ['logprice'])
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    
    if random_effect_vars is None:
        random_effect_vars = ['logprice']
    
    print(f"  Model: Random price coefficient by product")
    
    # Get data
    df_train = data_dict['df_train'].copy()
    df_test = data_dict['df_test'].copy()
    y_train = data_dict['y_train'].copy().astype(float)
    y_test = data_dict['y_test'].copy().astype(float)
    
    # Groups: use PRODUCT as grouping variable (price coefficient varies by product)
    groups_train = data_dict['product_train'].astype(str)
    groups_test = data_dict['product_test'].astype(str)
    
    # Build feature matrix: logprice + promotions (no intercept initially)
    print("  Building feature matrix...")
    
    # Continuous variables - center logprice to improve numerical stability
    logprice_mean = df_train['logprice'].mean()
    logprice_std = df_train['logprice'].std()
    
    X_train_me = pd.DataFrame({
        'logprice': (df_train['logprice'].values - logprice_mean) / logprice_std,
    }, index=df_train.index)
    
    X_test_me = pd.DataFrame({
        'logprice': (df_test['logprice'].values - logprice_mean) / logprice_std,
    }, index=df_test.index)
    
    # Promotion dummies - convert to numeric (handle string values like 'NONE')
    for col in ['f', 'd', 'pr']:
        if col in df_train.columns:
            X_train_me[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).values
            X_test_me[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).values
    
    # Ensure all numeric
    X_train_me = X_train_me.astype(float).fillna(0)
    X_test_me = X_test_me.astype(float).fillna(0)
    
    print(f"  Training features: {X_train_me.shape[1]}")
    print(f"  Training samples: {len(X_train_me)}")
    print(f"  Number of products (groups): {groups_train.nunique()}")
    print(f"  Logprice mean: {logprice_mean:.4f}, std: {logprice_std:.4f}")
    
    # Check group sizes
    group_sizes = groups_train.value_counts()
    print(f"  Min obs per product: {group_sizes.min()}, Max: {group_sizes.max()}, Mean: {group_sizes.mean():.1f}")
    
    print("  Fitting Mixed Effects model...")
    print("    Groups: product")
    print("    Random effects: intercept + logprice slope (elasticity varies by product)")
    print("    This may take a few minutes...")
    
    # Fit mixed effects model: random intercept + random slope for logprice by product
    # Use formula-based specification for clarity
    try:
        # Use smf for formula interface
        import statsmodels.formula.api as smf
        
        # Create temp dataframe with all variables
        temp_df = X_train_me.copy()
        temp_df['logunits'] = y_train.values
        temp_df['product'] = groups_train.values
        
        # Formula: random intercept and random slope for logprice by product
        # re_formula = "~logprice" means random intercept and random slope
        model = smf.mixedlm(
            "logunits ~ logprice + f + d + pr",
            data=temp_df,
            groups=temp_df['product'],
            re_formula="~logprice"  # random intercept + random slope for logprice
        )
        fitted = model.fit(method='powell', maxiter=500)  # Powell method more robust
        
        print(f"  Model converged: {fitted.converged}")
        print(f"  Log-likelihood: {fitted.llf:.2f}")
        
        # Extract key parameters
        print("\n  Fixed Effects:")
        fe = fitted.fe_params
        for key in fe.index:
            # Rescale logprice coefficient back to original scale
            if key == 'logprice':
                print(f"    {key}: {fe[key] / logprice_std:.6f} (rescaled)")
            else:
                print(f"    {key}: {fe[key]:.6f}")
        
        # Random effects variance
        print("\n  Random Effects Variance (by product):")
        re_cov = fitted.cov_re
        if isinstance(re_cov, pd.DataFrame):
            print(f"    Var(intercept): {re_cov.iloc[0, 0]:.6f}")
            if re_cov.shape[0] > 1:
                print(f"    Var(logprice): {re_cov.iloc[1, 1]:.6f}")
                print(f"    Corr(intercept, logprice): {re_cov.iloc[0, 1] / np.sqrt(re_cov.iloc[0, 0] * re_cov.iloc[1, 1]):.4f}")
        else:
            print(f"    Var(intercept): {re_cov:.6f}")
        
        # Predict on test set
        temp_test = X_test_me.copy()
        temp_test['product'] = groups_test.values
        
        y_pred_log = fitted.predict(exog=temp_test)
        
        # Convert from log to units
        y_pred_units = np.exp(y_pred_log)
        actual_units = np.exp(y_test)
        
        # Weights (revenue)
        weights = actual_units * df_test['price'].values
        
        # Evaluate
        metrics = evaluate_model(actual_units, y_pred_units, weights=weights, forecast_counterfactual=None)
        
        # Calculate counterfactual validity by store-week
        cf_validity = compute_mixed_effects_counterfactual_validity(
            fitted, temp_test, df_test, groups_test, price_multiplier=0.9
        )
        metrics['counterfactual_validity'] = cf_validity
        
        # Extract product-level price elasticities
        random_effects = fitted.random_effects
        product_elasticities = {}
        mean_logprice_coef = fe['logprice'] / logprice_std  # rescaled mean
        for product, re in random_effects.items():
            # Random slope for logprice (if exists)
            if 'logprice' in re.index:
                product_elasticities[product] = mean_logprice_coef + re['logprice'] / logprice_std
            else:
                product_elasticities[product] = mean_logprice_coef
        
        elasticity_values = list(product_elasticities.values())
        print(f"\n  Product-specific price elasticities:")
        print(f"    Mean: {np.mean(elasticity_values):.4f}")
        print(f"    Std: {np.std(elasticity_values):.4f}")
        print(f"    Min: {np.min(elasticity_values):.4f}")
        print(f"    Max: {np.max(elasticity_values):.4f}")
        
        return {
            'model': fitted,
            'predictions': y_pred_units,
            'metrics': metrics,
            'random_effects': random_effects,
            'product_elasticities': product_elasticities,
            'logprice_scaling': {'mean': logprice_mean, 'std': logprice_std}
        }
        
    except Exception as e:
        print(f"  ERROR fitting Mixed Effects model: {e}")
        print("  Falling back to random intercept only...")
        
        # Fallback: random intercept by product only (no random slope)
        try:
            import statsmodels.formula.api as smf
            
            temp_df = X_train_me.copy()
            temp_df['logunits'] = y_train.values
            temp_df['product'] = groups_train.values
            
            model = smf.mixedlm(
                "logunits ~ logprice + f + d + pr",
                data=temp_df,
                groups=temp_df['product']
                # No re_formula = random intercept only
            )
            fitted = model.fit(method='powell', maxiter=500)
            
            print(f"  Fallback model converged: {fitted.converged}")
            print(f"  Log-likelihood: {fitted.llf:.2f}")
            
            # Extract key parameters
            print("\n  Fixed Effects:")
            fe = fitted.fe_params
            for key in fe.index:
                if key == 'logprice':
                    print(f"    {key}: {fe[key] / logprice_std:.6f} (rescaled)")
                else:
                    print(f"    {key}: {fe[key]:.6f}")
            
            temp_test = X_test_me.copy()
            temp_test['product'] = groups_test.values
            
            y_pred_log = fitted.predict(exog=temp_test)
            y_pred_units = np.exp(y_pred_log)
            actual_units = np.exp(y_test)
            weights = actual_units * df_test['price'].values
            
            metrics = evaluate_model(actual_units, y_pred_units, weights=weights, forecast_counterfactual=None)
            
            # Counterfactual validity
            cf_validity = compute_mixed_effects_counterfactual_validity(
                fitted, temp_test, df_test, groups_test, price_multiplier=0.9
            )
            metrics['counterfactual_validity'] = cf_validity
            
            return {
                'model': fitted,
                'predictions': y_pred_units,
                'metrics': metrics,
                'random_effects': fitted.random_effects
            }
            
        except Exception as e2:
            print(f"  ERROR in fallback model: {e2}")
            # Ultimate fallback: simple OLS with product dummies
            print("  Using OLS with product fixed effects as ultimate fallback...")
            
            from sklearn.linear_model import Ridge
            
            # Add product dummies
            product_dummies_train = pd.get_dummies(groups_train, prefix='prod', drop_first=True)
            product_dummies_test = pd.get_dummies(groups_test, prefix='prod', drop_first=True)
            
            # Align columns
            for col in product_dummies_train.columns:
                if col not in product_dummies_test.columns:
                    product_dummies_test[col] = 0
            product_dummies_test = product_dummies_test[product_dummies_train.columns]
            
            X_train_full = pd.concat([X_train_me.reset_index(drop=True), 
                                      product_dummies_train.reset_index(drop=True)], axis=1)
            X_test_full = pd.concat([X_test_me.reset_index(drop=True), 
                                     product_dummies_test.reset_index(drop=True)], axis=1)
            
            # Use Ridge to handle potential multicollinearity
            model = Ridge(alpha=1.0)
            model.fit(X_train_full, y_train)
            
            y_pred_log = model.predict(X_test_full)
            y_pred_units = np.exp(y_pred_log)
            actual_units = np.exp(y_test)
            weights = actual_units * df_test['price'].values
            
            metrics = evaluate_model(actual_units, y_pred_units, weights=weights, forecast_counterfactual=None)
            metrics['counterfactual_validity'] = 0.0
            
            return {
                'model': model,
                'predictions': y_pred_units,
                'metrics': metrics
            }


def compute_mixed_effects_counterfactual_validity(fitted_model, X_test, df_test, 
                                                   groups_test, price_multiplier=0.9,
                                                   random_seed=42):
    """
    Compute counterfactual validity for mixed effects model.
    
    Tests that reducing price increases predicted units.
    """
    np.random.seed(random_seed)
    
    # Group by store-week
    store_weeks = df_test.groupby(['iri_key', 'week']).groups
    
    invalid_count = 0
    total_count = 0
    
    for (store, week), indices in store_weeks.items():
        if len(indices) == 0:
            continue
            
        # Randomly select one product in this store-week
        idx = np.random.choice(indices)
        
        # Get original prediction
        X_orig = X_test.iloc[[idx]].copy()
        pred_orig = np.exp(fitted_model.predict(exog=X_orig))[0]
        
        # Create counterfactual with reduced price
        X_cf = X_orig.copy()
        if 'logprice' in X_cf.columns:
            X_cf['logprice'] = X_cf['logprice'] + np.log(price_multiplier)
        
        pred_cf = np.exp(fitted_model.predict(exog=X_cf))[0]
        
        # Check if demand increased (price decreased)
        if pred_cf < pred_orig:
            invalid_count += 1
        
        total_count += 1
    
    return invalid_count / total_count if total_count > 0 else 0.0

def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 80)
    print("BASELINE DEMAND ESTIMATION - SALTY SNACK DATA")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data: {CONFIG['data_path']}")
    print(f"  Validation week start: {CONFIG['val_week_start']}")
    print(f"  Market potential multiplier: {CONFIG['market_potential_multiplier']}")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # ========== 1. LOAD AND PREPROCESS DATA ==========
    print("\n" + "=" * 80)
    print("[1/5] Loading and preprocessing data...")
    print("=" * 80)
    
    df = load_and_merge_data(CONFIG['data_path'], CONFIG['product_path'])
    df = filter_training_stores(df, CONFIG['val_week_start'])
    df = preprocess_data(df)
    
    print(f"\nData after preprocessing:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique stores: {df['iri_key'].nunique()}")
    print(f"  Unique products: {df['colupc'].nunique()}")
    print(f"  Week range: {df['week'].min()} - {df['week'].max()}")
    
    # ========== 2. PREPARE DATA FOR STANDARD MODELS ==========
    print("\n" + "=" * 80)
    print("[2/5] Preparing data for standard models (log units target)...")
    print("=" * 80)
    
    standard_data = prepare_standard_model_data(
        df, 
        val_week_start=CONFIG['val_week_start'],
        include_product_fe=False
    )
    
    standard_data_pfe = prepare_standard_model_data(
        df,
        val_week_start=CONFIG['val_week_start'],
        include_product_fe=True
    )
    
    print(f"Training set: {standard_data['X_train'].shape}")
    print(f"Validation set: {standard_data['X_test'].shape}")
    print(f"Train weeks: {standard_data['df_train']['week'].min()} - {standard_data['df_train']['week'].max()}")
    print(f"Val weeks: {standard_data['df_test']['week'].min()} - {standard_data['df_test']['week'].max()}")
    
    # ========== 3. PREPARE DATA FOR CHOICE MODELS ==========
    print("\n" + "=" * 80)
    print("[3/5] Preparing data for choice models (share difference target)...")
    print("=" * 80)
    
    choice_data = prepare_choice_model_data(
        df,
        val_week_start=CONFIG['val_week_start'],
        market_potential_multiplier=CONFIG['market_potential_multiplier']
    )
    
    print(f"Choice model training set: {choice_data['X_train'].shape}")
    print(f"Choice model validation set: {choice_data['X_test'].shape}")
    
    # ========== 4. TRAIN AND EVALUATE MODELS ==========
    print("\n" + "=" * 80)
    print("[4/5] Training and evaluating models...")
    print("=" * 80)
    
    all_results = {}
    
    # Model 1: Linear Regression (OLS)
    print("\n--- Training Linear Regression (OLS) ---")
    lr_model = LinearRegressionModel(**MODEL_CONFIGS['linear_regression'])
    lr_results = train_evaluate_standard_model(lr_model, standard_data, in_log_space=True)
    all_results['Linear Regression'] = lr_results['metrics']
    if CONFIG['verbose']:
        print_metrics(lr_results['metrics'], "Linear Regression (OLS)")
    
    # Model 2: Linear Regression with Product FE
    print("\n--- Training Linear Regression with Product FE ---")
    lr_pfe_model = LinearRegressionModel(**MODEL_CONFIGS['linear_regression_pfe'])
    lr_pfe_results = train_evaluate_standard_model(lr_pfe_model, standard_data_pfe, in_log_space=True)
    all_results['Linear Regression (Product FE)'] = lr_pfe_results['metrics']
    if CONFIG['verbose']:
        print_metrics(lr_pfe_results['metrics'], "Linear Regression (Product FE)")
    
    # Model 3: Mixed Effects (Random Price Slopes by Product)
    print("\n--- Training Mixed Effects Model (Random Price Slopes) ---")
    me_results = train_evaluate_mixed_effects_model(
        standard_data_pfe, 
        group_col='colupc',
        random_effect_vars=['logprice']
    )
    all_results['Mixed Effects (Random Slopes)'] = me_results['metrics']
    if CONFIG['verbose']:
        print_metrics(me_results['metrics'], "Mixed Effects (Random Price Slopes)")
    
    # Model 4: Homogeneous Logit
    print("\n--- Training Homogeneous Logit ---")
    hl_model = HomogeneousLogitModel(**MODEL_CONFIGS['homogeneous_logit'])
    hl_results = train_evaluate_choice_model(hl_model, choice_data)
    all_results['Homogeneous Logit'] = hl_results['metrics']
    if CONFIG['verbose']:
        print_metrics(hl_results['metrics'], "Homogeneous Logit")
    
    # Model 5: LightGBM
    print("\n--- Training LightGBM ---")
    lgbm_model = LightGBMModel(**MODEL_CONFIGS['lightgbm'])
    lgbm_results = train_evaluate_standard_model(lgbm_model, standard_data, in_log_space=True)
    all_results['LightGBM'] = lgbm_results['metrics']
    if CONFIG['verbose']:
        print_metrics(lgbm_results['metrics'], "LightGBM")
    
    # Model 6: Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestModel(**MODEL_CONFIGS['random_forest'])
    rf_results = train_evaluate_standard_model(rf_model, standard_data, in_log_space=True)
    all_results['Random Forest'] = rf_results['metrics']
    if CONFIG['verbose']:
        print_metrics(rf_results['metrics'], "Random Forest")
    
    # ========== 5. MODEL COMPARISON ==========
    print("\n" + "=" * 80)
    print("[5/5] Comparing all models...")
    print("=" * 80)

    comparison = compare_models(all_results, metric='wmape')

    # Print comprehensive comparison summary and save to file
    summary_file = os.path.join(CONFIG['output_dir'], 'baseline_model_comparison.txt')
    print_comparison_summary(comparison, sort_metric='wmape', output_file=summary_file)

    # Find best model
    best_model_name = comparison.index[0]
    print(f"\n{'='*100}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*100}")
    
    # Save results
    if CONFIG['save_results']:
        results = {
            'config': CONFIG,
            'model_configs': MODEL_CONFIGS,
            'metrics': {k: {m: float(v) for m, v in metrics.items()} 
                       for k, metrics in all_results.items()},
            'comparison': comparison.to_dict(),
            'best_model': best_model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(CONFIG['output_dir'], 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'models': {
            'lr': lr_results,
            'lr_pfe': lr_pfe_results,
            'me': me_results,
            'hl': hl_results,
            'lgbm': lgbm_results,
            'rf': rf_results
        },
        'data': {
            'standard': standard_data,
            'standard_pfe': standard_data_pfe,
            'choice': choice_data
        },
        'comparison': comparison,
        'all_results': all_results
    }


if __name__ == "__main__":
    results = main()
