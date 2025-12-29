"""
Evaluation metrics module for demand estimation models.
Contains functions for calculating various performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Union


def wmape_score(actual: Union[np.ndarray, pd.Series], 
                forecast: Union[np.ndarray, pd.Series],
                weight: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        weight: Weights for each observation (e.g., revenue)
        
    Returns:
        WMAPE score
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    weight = np.asarray(weight)
    
    wmape = ((np.abs(forecast - actual) / np.abs(actual)) * weight / weight.sum()).sum()
    return wmape


def mape_score(actual: Union[np.ndarray, pd.Series],
               forecast: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Only considers observations where actual > 0.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        
    Returns:
        MAPE score
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    
    mape = np.abs((actual - forecast) / actual)[mask].mean()
    return mape


def mpe_score(actual: Union[np.ndarray, pd.Series],
              forecast: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Percentage Error (MPE).
    Measures bias in predictions.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        
    Returns:
        MPE score
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    
    mpe = np.mean((actual - forecast) / actual)
    return mpe


def counterfactual_validity(forecast_baseline: Union[np.ndarray, pd.Series],
                            forecast_counterfactual: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate counterfactual validity metric (DEPRECATED - use counterfactual_validity_by_store_week).
    Measures the percentage of predictions that violate economic intuition
    (e.g., demand increases when price increases).
    
    Args:
        forecast_baseline: Baseline forecasts
        forecast_counterfactual: Counterfactual forecasts (e.g., with price decrease)
        
    Returns:
        Percentage of observations where counterfactual <= baseline (invalid)
    """
    forecast_baseline = np.asarray(forecast_baseline)
    forecast_counterfactual = np.asarray(forecast_counterfactual)
    
    # For price decrease, we expect demand to increase
    # Invalid cases: counterfactual demand <= baseline demand
    per_not_valid = np.mean((forecast_counterfactual - forecast_baseline) <= 0)
    return per_not_valid


def counterfactual_validity_by_store_week(
    df_test: pd.DataFrame,
    model,
    X_test: pd.DataFrame,
    in_log_space: bool = False,
    price_multiplier: float = 0.9,
    random_seed: int = 42
) -> float:
    """
    Calculate counterfactual validity by randomly selecting one product per store-week
    and checking if its demand increases when its price decreases.
    
    This properly tests the Law of Demand: for each store-week, we:
    1. Randomly select one product
    2. Decrease only that product's price by (1 - price_multiplier)
    3. Check if that product's predicted demand increases
    
    Args:
        df_test: Test dataframe with 'iri_key', 'week', 'colupc' columns
        model: Trained model with predict method
        X_test: Feature matrix for test set
        in_log_space: Whether predictions are in log space
        price_multiplier: Price change factor (0.9 = 10% decrease)
        random_seed: Random seed for reproducibility
        
    Returns:
        Percentage of store-weeks where the counterfactual is invalid
        (demand did not increase when price decreased)
    """
    np.random.seed(random_seed)
    
    # Get unique store-week combinations
    store_weeks = df_test.groupby(['iri_key', 'week']).size().reset_index()[['iri_key', 'week']]
    
    invalid_count = 0
    total_count = 0
    
    for _, row in store_weeks.iterrows():
        store_id = row['iri_key']
        week = row['week']
        
        # Get indices for this store-week
        mask = (df_test['iri_key'] == store_id) & (df_test['week'] == week)
        indices = df_test[mask].index.tolist()
        
        if len(indices) == 0:
            continue
        
        # Randomly select one product in this store-week
        selected_idx = np.random.choice(indices)
        
        # Get baseline prediction for the selected product
        X_baseline = X_test.loc[[selected_idx]]
        y_baseline = model.predict(X_baseline)
        
        # Create counterfactual: decrease price only for this product
        X_cf = X_baseline.copy()
        X_cf['logprice'] = np.log(np.exp(X_cf['logprice']) * price_multiplier)
        y_cf = model.predict(X_cf)
        
        # Convert from log space if needed
        if in_log_space:
            baseline_units = np.exp(y_baseline[0])
            cf_units = np.exp(y_cf[0])
        else:
            baseline_units = y_baseline[0]
            cf_units = y_cf[0]
        
        # Check validity: demand should increase when price decreases
        # Invalid if counterfactual demand <= baseline demand
        if cf_units <= baseline_units:
            invalid_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 0.0
    
    return invalid_count / total_count


def counterfactual_validity_choice_model_by_store_week(
    df_test: pd.DataFrame,
    model,
    X_test: pd.DataFrame,
    price_multiplier: float = 0.9,
    random_seed: int = 42
) -> float:
    """
    Calculate counterfactual validity for choice models (logit) by randomly selecting 
    one product per store-week and checking if its share/units increase when its 
    price decreases.
    
    For choice models, the prediction is log(share) - log(outside_share), so we need
    to convert to units: units = exp(y_pred) * outside_share * market_potential
    
    Args:
        df_test: Test dataframe with 'iri_key', 'week', 'colupc', 'outside_share', 'm_potential' columns
        model: Trained choice model with predict method
        X_test: Feature matrix for test set
        price_multiplier: Price change factor (0.9 = 10% decrease)
        random_seed: Random seed for reproducibility
        
    Returns:
        Percentage of store-weeks where the counterfactual is invalid
        (demand did not increase when price decreased)
    """
    np.random.seed(random_seed)
    
    # Get unique store-week combinations
    store_weeks = df_test.groupby(['iri_key', 'week']).size().reset_index()[['iri_key', 'week']]
    
    invalid_count = 0
    total_count = 0
    
    for _, row in store_weeks.iterrows():
        store_id = row['iri_key']
        week = row['week']
        
        # Get indices for this store-week
        mask = (df_test['iri_key'] == store_id) & (df_test['week'] == week)
        indices = df_test[mask].index.tolist()
        
        if len(indices) == 0:
            continue
        
        # Randomly select one product in this store-week
        selected_idx = np.random.choice(indices)
        
        # Get baseline prediction for the selected product (share difference)
        X_baseline = X_test.loc[[selected_idx]]
        y_baseline_share = model.predict(X_baseline)
        
        # Get outside_share and m_potential for this observation
        outside_share = df_test.loc[selected_idx, 'outside_share']
        m_potential = df_test.loc[selected_idx, 'm_potential']
        
        # Convert share difference to units
        baseline_units = np.exp(y_baseline_share[0]) * outside_share * m_potential
        
        # Create counterfactual: decrease price only for this product
        X_cf = X_baseline.copy()
        X_cf['logprice'] = np.log(np.exp(X_cf['logprice']) * price_multiplier)
        y_cf_share = model.predict(X_cf)
        
        # Convert counterfactual share to units
        cf_units = np.exp(y_cf_share[0]) * outside_share * m_potential
        
        # Check validity: demand should increase when price decreases
        # Invalid if counterfactual demand <= baseline demand
        if cf_units <= baseline_units:
            invalid_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 0.0
    
    return invalid_count / total_count


def rmse_score(actual: Union[np.ndarray, pd.Series],
               forecast: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        
    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(actual, forecast))


def evaluate_model(actual: Union[np.ndarray, pd.Series],
                   forecast: Union[np.ndarray, pd.Series],
                   weights: Union[np.ndarray, pd.Series] = None,
                   forecast_counterfactual: Union[np.ndarray, pd.Series] = None) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        weights: Optional weights for WMAPE calculation
        forecast_counterfactual: Optional counterfactual forecasts for validity check
        
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {
        'rmse': rmse_score(actual, forecast),
        'r2': r2_score(actual, forecast),
        'mape': mape_score(actual, forecast),
        'mpe': mpe_score(actual, forecast)
    }
    
    if weights is not None:
        metrics['wmape'] = wmape_score(actual, forecast, weights)
    
    if forecast_counterfactual is not None:
        metrics['counterfactual_validity'] = counterfactual_validity(forecast, forecast_counterfactual)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
        model_name: Name of the model for display
    """
    print("=" * 80)
    print(f"{model_name} Performance Metrics")
    print("=" * 80)
    
    metric_names = {
        'rmse': 'RMSE',
        'r2': 'R²',
        'mape': 'MAPE',
        'wmape': 'WMAPE',
        'mpe': 'MPE',
        'counterfactual_validity': 'Counterfactual Validity (% invalid)'
    }
    
    for key, value in metrics.items():
        display_name = metric_names.get(key, key)
        if key == 'counterfactual_validity':
            print(f"{display_name:30s}: {value:.4%}")
        else:
            print(f"{display_name:30s}: {value:.6f}")
    
    print("=" * 80)


def compare_models(model_results: Dict[str, Dict[str, float]],
                  metric: str = 'rmse') -> pd.DataFrame:
    """
    Compare multiple models based on a specific metric.

    Args:
        model_results: Dictionary mapping model names to their metrics dictionaries
        metric: Metric to sort by (default: 'rmse')

    Returns:
        DataFrame with models ranked by the specified metric
    """
    comparison_df = pd.DataFrame(model_results).T

    # Ensure consistent column order for better readability
    desired_order = ['rmse', 'r2', 'mape', 'wmape', 'mpe', 'counterfactual_validity']
    existing_cols = [col for col in desired_order if col in comparison_df.columns]
    comparison_df = comparison_df[existing_cols]

    # Sort by metric (ascending for error metrics, descending for R²)
    ascending = metric != 'r2'
    comparison_df = comparison_df.sort_values(by=metric, ascending=ascending)

    return comparison_df


def print_comparison_summary(comparison_df: pd.DataFrame, sort_metric: str = 'wmape', output_file: str = None) -> None:
    """
    Print a formatted comparison summary of all models and optionally save to file.

    Args:
        comparison_df: DataFrame with model comparison (from compare_models)
        sort_metric: Metric used for sorting (for display purposes)
        output_file: Optional path to save the summary as a text file
    """
    # Build the summary text
    lines = []
    lines.append("=" * 100)
    lines.append(f"MODEL COMPARISON SUMMARY (sorted by {sort_metric.upper()})")
    lines.append("=" * 100)

    # Create a formatted version with better column names and formatting
    formatted_df = comparison_df.copy()

    # Rename columns for display
    column_display_names = {
        'rmse': 'RMSE',
        'r2': 'R²',
        'mape': 'MAPE',
        'wmape': 'WMAPE',
        'mpe': 'MPE',
        'counterfactual_validity': 'CF Invalid %'
    }

    formatted_df.columns = [column_display_names.get(col, col) for col in formatted_df.columns]

    # Format numeric values for better readability
    table_str = formatted_df.to_string(float_format=lambda x: f'{x:.6f}')
    lines.append(table_str)
    lines.append("=" * 100)

    # Add summary statistics
    lines.append("")
    lines.append("Key Metrics Summary:")
    lines.append("-" * 100)

    if 'WMAPE' in formatted_df.columns:
        best_wmape_model = formatted_df['WMAPE'].idxmin()
        lines.append(f"  Best WMAPE:   {best_wmape_model:30s} = {formatted_df.loc[best_wmape_model, 'WMAPE']:.6f}")

    if 'RMSE' in formatted_df.columns:
        best_rmse_model = formatted_df['RMSE'].idxmin()
        lines.append(f"  Best RMSE:    {best_rmse_model:30s} = {formatted_df.loc[best_rmse_model, 'RMSE']:.6f}")

    if 'R²' in formatted_df.columns:
        best_r2_model = formatted_df['R²'].idxmax()
        lines.append(f"  Best R²:      {best_r2_model:30s} = {formatted_df.loc[best_r2_model, 'R²']:.6f}")

    if 'CF Invalid %' in formatted_df.columns:
        # Only consider models with non-zero counterfactual validity (0.0 means "not computed")
        cf_models = formatted_df[formatted_df['CF Invalid %'] > 0.0]
        if len(cf_models) > 0:
            best_cf_model = cf_models['CF Invalid %'].idxmin()
            lines.append(f"  Best CF:      {best_cf_model:30s} = {formatted_df.loc[best_cf_model, 'CF Invalid %']:.6f} ({formatted_df.loc[best_cf_model, 'CF Invalid %']*100:.2f}% invalid)")

    lines.append("-" * 100)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Lower is better for: RMSE, MAPE, WMAPE, MPE (absolute value), CF Invalid %")
    lines.append("  - Higher is better for: R²")
    lines.append("  - CF Invalid % of 0.0 means counterfactual validity was not computed (model doesn't use price)")
    lines.append("=" * 100)

    # Print to console
    summary_text = "\n" + "\n".join(lines)
    print(summary_text)

    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            print(f"\n  Summary saved to: {output_file}")
        except Exception as e:
            print(f"\n  WARNING: Could not save summary to file: {e}")
