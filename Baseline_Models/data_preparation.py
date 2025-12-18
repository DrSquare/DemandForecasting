"""
Data preparation module for demand estimation.
Prepares data for both choice models and standard models with train/test splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def calculate_equivalent_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate equivalent units by multiplying units by volume equivalent.
    
    Args:
        df: Input dataframe with 'units' and 'vol_eq' columns
        
    Returns:
        DataFrame with 'eq_units' column added
    """
    df = df.copy()
    df['eq_units'] = df['units'] * df['vol_eq']
    return df


def calculate_market_potential(df_train: pd.DataFrame, 
                               multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate market potential for each store as multiplier * max(total_eq_units).
    
    Args:
        df_train: Training dataframe
        multiplier: Multiplier for market potential calculation (default 3.0)
        
    Returns:
        DataFrame with store-level market potential
    """
    # Calculate total units by store-week
    totalunits = df_train.groupby(['iri_key', 'week'])['eq_units'].sum().reset_index()
    totalunits.rename(columns={'eq_units': 'total_eq_units'}, inplace=True)
    
    # Merge back to training data
    df_train = df_train.merge(totalunits, how='left', on=['iri_key', 'week'])
    
    # Calculate market potential by store
    m_potential = df_train.groupby(['iri_key'])['total_eq_units'].max().reset_index()
    m_potential['m_potential'] = multiplier * m_potential['total_eq_units']
    m_potential.drop("total_eq_units", axis=1, inplace=True)
    
    return m_potential


def add_share_variables(df: pd.DataFrame, 
                       m_potential: pd.DataFrame) -> pd.DataFrame:
    """
    Add share variables including outside share for choice models.
    
    Args:
        df: Input dataframe with equivalent units
        m_potential: Market potential dataframe from calculate_market_potential
        
    Returns:
        DataFrame with share variables added
    """
    df = df.copy()
    
    # Calculate total units by store-week
    totalunits = df.groupby(['iri_key', 'week'])['eq_units'].sum().reset_index()
    totalunits.rename(columns={'eq_units': 'total_eq_units'}, inplace=True)
    
    # Merge market potential
    df = df.merge(m_potential, how='left', on=['iri_key'])
    
    # Calculate share (including outside option)
    df['share'] = df['eq_units'] / df['m_potential']
    
    # Merge total units and calculate within-market share
    df = df.merge(totalunits, how='left', on=['iri_key', 'week'])
    df['share_within'] = df['eq_units'] / df['total_eq_units']
    
    # Calculate outside share and log transformations
    df['outside_share'] = 1 - df['total_eq_units'] / df['m_potential']
    df['logshare'] = np.log(df['share'])
    df['logoutsideshare'] = np.log(df['outside_share'])
    
    # Create share difference for logit model: log(share) - log(outside_share)
    df['sharedp'] = df['logshare'] - df['logoutsideshare']
    
    return df


def train_test_split_by_week(df: pd.DataFrame, 
                             test_week_start: int = 285) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets based on week cutoff.
    
    Args:
        df: Input dataframe
        test_week_start: Week number where test set starts
        
    Returns:
        Tuple of (train_df, test_df)
    """
    mask = df['week'] >= test_week_start
    df_train = df[~mask].copy()
    df_test = df[mask].copy()
    return df_train, df_test


def prepare_standard_model_data(df: pd.DataFrame,
                                test_week_start: int = 285,
                                include_product_fe: bool = False) -> Dict:
    """
    Prepare data for standard (non-choice) models using log units as target.
    
    Args:
        df: Input dataframe
        test_week_start: Week number where test set starts
        include_product_fe: Whether to include product fixed effects
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test, df_train, df_test
    """
    from feature_engineering import build_feature_matrix_standard
    
    # Build feature matrix
    X, y = build_feature_matrix_standard(df, include_product_fe=include_product_fe)
    
    # Create train/test split mask
    mask = df['week'] >= test_week_start
    
    # Split features and target
    X_train = X[~mask].copy()
    X_test = X[mask].copy()
    y_train = y[~mask].copy()
    y_test = y[mask].copy()
    
    # Split original dataframes
    df_train = df[~mask].copy()
    df_test = df[mask].copy()
    
    # Get product identifiers for train/test
    product_train = df[~mask]['colupc'].copy()
    product_test = df[mask]['colupc'].copy()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'df_train': df_train,
        'df_test': df_test,
        'product_train': product_train,
        'product_test': product_test
    }


def prepare_choice_model_data(df: pd.DataFrame,
                              test_week_start: int = 285,
                              market_potential_multiplier: float = 3.0) -> Dict:
    """
    Prepare data for choice models using share difference as target.
    
    Args:
        df: Input dataframe
        test_week_start: Week number where test set starts
        market_potential_multiplier: Multiplier for market potential calculation
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test, df_train, df_test,
        and additional choice model specific variables
    """
    from feature_engineering import build_feature_matrix_choice
    
    # Add equivalent units
    df = calculate_equivalent_units(df)
    
    # Split train/test first to calculate market potential on training data only
    df_train, df_test = train_test_split_by_week(df, test_week_start)
    
    # Calculate market potential from training data
    m_potential = calculate_market_potential(df_train, market_potential_multiplier)
    
    # Add share variables to full dataset
    df = add_share_variables(df, m_potential)
    
    # Build feature matrix
    X, y = build_feature_matrix_choice(df, share_column='sharedp')
    
    # Create train/test split mask
    mask = df['week'] >= test_week_start
    
    # Split features and target
    X_train = X[~mask].copy()
    X_test = X[mask].copy()
    y_train = y[~mask].copy()
    y_test = y[mask].copy()
    
    # Split original dataframes with share variables
    df_train = df[~mask].copy()
    df_test = df[mask].copy()
    
    # Extract market potential and other variables for test set
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


def create_counterfactual_features(X: pd.DataFrame, 
                                  price_multiplier: float = 0.9) -> pd.DataFrame:
    """
    Create counterfactual feature matrix with adjusted prices.
    
    Args:
        X: Original feature matrix
        price_multiplier: Multiplier for price adjustment (e.g., 0.9 for 10% decrease)
        
    Returns:
        Feature matrix with adjusted log prices
    """
    X_cf = X.copy()
    X_cf['logprice'] = np.log(np.exp(X_cf['logprice']) * price_multiplier)
    return X_cf
