"""
Feature engineering module for demand estimation.
Handles creation of dummy variables and feature matrices for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class FeatureEngineer:
    """
    Class to handle feature engineering for demand estimation models.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with dataframe.
        
        Args:
            df: Input dataframe with all required columns
        """
        self.df = df
        
    def create_product_dummies(self) -> Dict[str, pd.DataFrame]:
        """
        Create dummy variables for product attributes.
        
        Returns:
            Dictionary with dummy variable DataFrames for each product attribute
        """
        dummies = {}
        dummies['brand'] = pd.get_dummies(self.df['brand'], prefix='BRAND', drop_first=True)
        dummies['package'] = pd.get_dummies(self.df['package'], prefix='PKG', drop_first=True)
        dummies['flavor'] = pd.get_dummies(self.df['flavorscent'], prefix='FLV', drop_first=True)
        dummies['fat'] = pd.get_dummies(self.df['fatcontent'], prefix='FAT', drop_first=True)
        dummies['cook'] = pd.get_dummies(self.df['cookingmethod'], prefix='CM', drop_first=True)
        dummies['salt'] = pd.get_dummies(self.df['saltsodiumcontent'], prefix='SALT', drop_first=True)
        dummies['cut'] = pd.get_dummies(self.df['typeofcut'], prefix='CUT', drop_first=True)
        return dummies
    
    def create_promotion_dummies(self) -> Dict[str, pd.DataFrame]:
        """
        Create dummy variables for promotion variables.
        
        Returns:
            Dictionary with dummy variable DataFrames for each promotion type
        """
        dummies = {}
        dummies['feature'] = pd.get_dummies(self.df['f'], prefix='F', drop_first=True)
        dummies['display'] = pd.get_dummies(self.df['d'], prefix='D', drop_first=True)
        dummies['promo'] = pd.get_dummies(self.df['pr'], prefix='PR', drop_first=True)
        return dummies
    
    def create_fixed_effect_dummies(self) -> Dict[str, pd.DataFrame]:
        """
        Create dummy variables for fixed effects (store, week, product).
        
        Returns:
            Dictionary with dummy variable DataFrames for each fixed effect
        """
        dummies = {}
        dummies['store'] = pd.get_dummies(self.df['iri_key'], prefix='STORE', drop_first=True)
        dummies['week'] = pd.get_dummies(self.df['numweek'], prefix='WK', drop_first=True)
        dummies['product'] = pd.get_dummies(self.df['colupc'])
        return dummies
    
    def create_all_dummies(self) -> Dict[str, pd.DataFrame]:
        """
        Create all dummy variables at once.
        
        Returns:
            Dictionary with all dummy variable DataFrames
        """
        all_dummies = {}
        all_dummies.update(self.create_product_dummies())
        all_dummies.update(self.create_promotion_dummies())
        all_dummies.update(self.create_fixed_effect_dummies())
        return all_dummies
    
    def get_continuous_variables(self) -> Dict[str, pd.Series]:
        """
        Extract continuous variables from dataframe.
        
        Returns:
            Dictionary with continuous variable Series
        """
        continuous = {}
        continuous['logprice'] = self.df['logprice']
        continuous['price'] = self.df['price']
        continuous['voleq'] = self.df['vol_eq']
        return continuous
    
    def get_target_variables(self) -> Dict[str, pd.Series]:
        """
        Extract target variables from dataframe.
        
        Returns:
            Dictionary with target variable Series
        """
        targets = {}
        targets['logunits'] = self.df['logunits']
        targets['units'] = self.df['units']
        return targets
    
    def get_identifier_variables(self) -> Dict[str, pd.Series]:
        """
        Extract identifier variables from dataframe.
        
        Returns:
            Dictionary with identifier variable Series
        """
        identifiers = {}
        identifiers['product'] = self.df['colupc']
        identifiers['store'] = self.df['iri_key']
        identifiers['week'] = self.df['week']
        return identifiers


def build_feature_matrix_standard(df: pd.DataFrame, 
                                  include_product_fe: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build standard feature matrix for non-choice models (using log units as target).
    
    Args:
        df: Input dataframe
        include_product_fe: Whether to include product fixed effects instead of product attributes
        
    Returns:
        Tuple of (X feature matrix, y target variable)
    """
    fe = FeatureEngineer(df)
    
    # Get all components
    continuous = fe.get_continuous_variables()
    product_dummies = fe.create_product_dummies()
    promo_dummies = fe.create_promotion_dummies()
    fixed_effects = fe.create_fixed_effect_dummies()
    targets = fe.get_target_variables()
    
    # Build X matrix
    if include_product_fe:
        # Use product fixed effects instead of product attributes
        X = pd.concat([
            continuous['logprice'],
            fixed_effects['store'],
            fixed_effects['week'],
            promo_dummies['feature'],
            promo_dummies['display'],
            promo_dummies['promo'],
            fixed_effects['product']
        ], axis=1)
    else:
        # Use product attributes
        X = pd.concat([
            continuous['logprice'],
            fixed_effects['store'],
            fixed_effects['week'],
            promo_dummies['feature'],
            promo_dummies['display'],
            promo_dummies['promo'],
            product_dummies['brand'],
            continuous['voleq'],
            product_dummies['package'],
            product_dummies['flavor'],
            product_dummies['fat'],
            product_dummies['cook'],
            product_dummies['salt'],
            product_dummies['cut']
        ], axis=1)
    
    y = targets['logunits']
    
    return X, y


def build_feature_matrix_choice(df: pd.DataFrame,
                               share_column: str = 'sharedp') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix for choice models (using share difference as target).
    
    Args:
        df: Input dataframe with share variables already computed
        share_column: Name of the share difference column (log(share) - log(outside_share))
        
    Returns:
        Tuple of (X feature matrix, y target variable)
    """
    fe = FeatureEngineer(df)
    
    # Get all components
    continuous = fe.get_continuous_variables()
    product_dummies = fe.create_product_dummies()
    promo_dummies = fe.create_promotion_dummies()
    fixed_effects = fe.create_fixed_effect_dummies()
    
    # Build X matrix (same as standard model)
    X = pd.concat([
        continuous['logprice'],
        fixed_effects['store'],
        fixed_effects['week'],
        promo_dummies['feature'],
        promo_dummies['display'],
        promo_dummies['promo'],
        product_dummies['brand'],
        continuous['voleq'],
        product_dummies['package'],
        product_dummies['flavor'],
        product_dummies['fat'],
        product_dummies['cook'],
        product_dummies['salt'],
        product_dummies['cut']
    ], axis=1)
    
    # Use share difference as target
    y = df[share_column]
    
    return X, y


def create_intercept(n_rows: int) -> pd.DataFrame:
    """
    Create intercept column for models that require it.
    
    Args:
        n_rows: Number of rows
        
    Returns:
        DataFrame with single intercept column
    """
    return pd.DataFrame(np.ones(n_rows), columns=['Intercept'])
