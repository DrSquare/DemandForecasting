"""
Data loading and preprocessing module for demand estimation.
Handles data loading, filtering, and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load POS dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with POS data
    """
    df = pd.read_csv(filepath)
    return df


def filter_training_stores(df: pd.DataFrame, train_week_cutoff: int = 285) -> pd.DataFrame:
    """
    Filter dataset to only include stores that exist in the training period.
    
    Args:
        df: Input dataframe
        train_week_cutoff: Week number that separates training and test periods
        
    Returns:
        Filtered dataframe containing only stores from training set
    """
    df_train = df[df['week'] < train_week_cutoff]
    store_list = df_train['iri_key'].unique()
    df_filtered = df[df['iri_key'].isin(store_list)]
    return df_filtered


def handle_missing_values(df: pd.DataFrame, 
                         missing_value_map: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values in categorical columns.
    
    Args:
        df: Input dataframe
        missing_value_map: Dictionary mapping column names to replacement values
                          If None, uses default mapping
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if missing_value_map is None:
        missing_value_map = {
            "fatcontent": "REGULAR",
            "cookingmethod": "MISSING",
            "saltsodiumcontent": "MISSING"
        }
    
    for col, replacement in missing_value_map.items():
        if col in df.columns:
            df[col].replace({np.nan: replacement}, inplace=True)
    
    return df


def get_categorical_columns() -> List[str]:
    """
    Get list of categorical column names in the dataset.
    
    Returns:
        List of categorical column names
    """
    return ["producttype", "package", "flavorscent", "fatcontent", 
            "cookingmethod", "saltsodiumcontent", "typeofcut"]


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive data summary including shape, columns, nulls, and statistics.
    
    Args:
        df: Input dataframe
    """
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNull values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    pd.set_option("display.max_columns", None)
    print(f"\nDescriptive statistics:\n{df.describe()}")
    print("=" * 80)


def print_categorical_values(df: pd.DataFrame, 
                             categorical_cols: List[str] = None) -> None:
    """
    Print unique values for all categorical columns.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names. If None, uses default list.
    """
    if categorical_cols is None:
        categorical_cols = get_categorical_columns()
    
    print("=" * 80)
    print("CATEGORICAL VALUES")
    print("=" * 80)
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}: {df[col].unique()}")
    print("=" * 80)
