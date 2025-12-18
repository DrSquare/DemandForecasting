"""
Model training module for demand estimation.
Encapsulates training and prediction logic for different model types.
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import statsmodels.api as sm
from typing import Dict, Any, Tuple


class BaselineModel:
    """Base class for baseline demand models."""
    
    def __init__(self, name: str):
        """Initialize model with name."""
        self.name = name
        self.model = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model."""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict method")


class LinearRegressionModel(BaselineModel):
    """OLS Linear Regression model."""
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize linear regression model.
        
        Args:
            fit_intercept: Whether to fit intercept
        """
        super().__init__("Linear Regression")
        self.fit_intercept = fit_intercept
        self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit linear regression model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class MixedEffectsModel(BaselineModel):
    """Linear Mixed Effects Model."""
    
    def __init__(self, group_col: str = 'colupc', 
                 random_effect_vars: list = None):
        """
        Initialize mixed effects model.
        
        Args:
            group_col: Column name for grouping (e.g., product ID)
            random_effect_vars: Variables to include as random effects
        """
        super().__init__("Mixed Effects Model")
        self.group_col = group_col
        self.random_effect_vars = random_effect_vars or ['logprice']
        self.fitted_model = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            groups: pd.Series):
        """
        Fit mixed effects model.
        
        Args:
            X_train: Training features
            y_train: Training target
            groups: Grouping variable (e.g., product IDs)
        """
        # Prepare data for statsmodels
        exog = X_train.copy()
        
        # Extract random effects variable
        if 'Intercept' not in exog.columns:
            exog['Intercept'] = 1
        
        re_vars = exog[self.random_effect_vars[0]] if len(self.random_effect_vars) == 1 else exog[self.random_effect_vars]
        
        # Fit model
        lm = sm.MixedLM(y_train, exog, groups=groups, exog_re=re_vars)
        self.fitted_model = lm.fit()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        exog_test = X.copy()
        if 'Intercept' not in exog_test.columns:
            exog_test['Intercept'] = 1
            
        return self.fitted_model.predict(exog=exog_test)
    
    def summary(self):
        """Print model summary."""
        if self.fitted_model is not None:
            return self.fitted_model.summary()
        return "Model not fitted yet"


class HomogeneousLogitModel(BaselineModel):
    """Homogeneous Logit model with outside share."""
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize homogeneous logit model.
        
        Args:
            fit_intercept: Whether to fit intercept
        """
        super().__init__("Homogeneous Logit")
        self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit logit model.
        
        Args:
            X_train: Training features
            y_train: Training target (share difference: log(share) - log(outside_share))
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions in share difference space."""
        return self.model.predict(X)
    
    def predict_units(self, X: pd.DataFrame, df_test: pd.DataFrame,
                     mp_test: pd.Series) -> np.ndarray:
        """
        Convert share predictions to unit predictions.
        
        Args:
            X: Test features
            df_test: Test dataframe with store/week info
            mp_test: Market potential for test observations
            
        Returns:
            Predicted units
        """
        # Predict share ratio
        y_pred_share = self.predict(X)
        share_ratio_pred = pd.DataFrame(np.exp(y_pred_share), columns=['ShareRatio_pred'])
        
        # Combine with store/week information
        share_ratio_comb = pd.concat([
            share_ratio_pred,
            df_test['iri_key'].reset_index(drop=True),
            df_test['week'].reset_index(drop=True)
        ], axis=1)
        
        # Calculate sum of share ratios by store-week
        share_sum = share_ratio_comb.groupby(['iri_key', 'week'])['ShareRatio_pred'].sum().reset_index()
        share_sum.rename(columns={'ShareRatio_pred': 'ShareRatio_sum'}, inplace=True)
        
        # Merge back and calculate outside share
        share_ratio_comb = share_ratio_comb.merge(share_sum, how='left', on=['iri_key', 'week'])
        share_ratio_comb['Osh'] = 1 / (share_ratio_comb['ShareRatio_sum'] + 1)
        
        # Convert to units
        units_pred = (share_ratio_comb['ShareRatio_pred'] * 
                     share_ratio_comb['Osh'] * 
                     mp_test.reset_index(drop=True))
        
        return units_pred.values


class XGBoostModel(BaselineModel):
    """XGBoost Regression model."""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            **kwargs: Parameters for XGBRegressor
        """
        super().__init__("XGBoost")
        self.model = XGBRegressor(**kwargs)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit XGBoost model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class RandomForestModel(BaselineModel):
    """Random Forest Regression model."""
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            **kwargs: Parameters for RandomForestRegressor
        """
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(**kwargs)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Random Forest model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class LightGBMModel(BaselineModel):
    """LightGBM Regression model."""
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM model.
        
        Args:
            **kwargs: Parameters for LGBMRegressor
        """
        super().__init__("LightGBM")
        self.model = lgb.LGBMRegressor(**kwargs)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit LightGBM model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


def train_and_evaluate_standard_model(model: BaselineModel,
                                     data_dict: Dict,
                                     in_log_space: bool = True) -> Dict[str, Any]:
    """
    Train and evaluate a standard model (non-choice).
    
    Args:
        model: BaselineModel instance
        data_dict: Dictionary with train/test data from prepare_standard_model_data
        in_log_space: Whether predictions are in log space
        
    Returns:
        Dictionary with model, predictions, and data
    """
    from evaluation_metrics import evaluate_model
    from data_preparation import create_counterfactual_features
    
    # Train model
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test'])
    
    # Create counterfactual predictions (10% price decrease)
    X_test_cf = create_counterfactual_features(data_dict['X_test'], price_multiplier=0.9)
    y_pred_cf = model.predict(X_test_cf)
    
    # Convert to actual space if in log space
    if in_log_space:
        actual = np.exp(data_dict['y_test'])
        forecast = np.exp(y_pred)
        forecast_cf = np.exp(y_pred_cf)
    else:
        actual = data_dict['y_test']
        forecast = y_pred
        forecast_cf = y_pred_cf
    
    # Calculate weights for WMAPE (revenue = units * price)
    weights = actual * data_dict['df_test']['price'].values
    
    # Evaluate
    metrics = evaluate_model(actual, forecast, weights=weights, 
                            forecast_counterfactual=forecast_cf)
    
    return {
        'model': model,
        'predictions': y_pred,
        'predictions_cf': y_pred_cf,
        'metrics': metrics
    }


def train_and_evaluate_choice_model(model: BaselineModel,
                                   data_dict: Dict) -> Dict[str, Any]:
    """
    Train and evaluate a choice model (logit).
    
    Args:
        model: BaselineModel instance (should be HomogeneousLogitModel)
        data_dict: Dictionary with train/test data from prepare_choice_model_data
        
    Returns:
        Dictionary with model, predictions, and data
    """
    from evaluation_metrics import evaluate_model
    from data_preparation import create_counterfactual_features
    
    # Train model
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    # Make predictions (convert to units)
    if isinstance(model, HomogeneousLogitModel):
        y_pred_units = model.predict_units(
            data_dict['X_test'], 
            data_dict['df_test'],
            data_dict['mp_test']
        )
        
        # Counterfactual predictions
        X_test_cf = create_counterfactual_features(data_dict['X_test'], price_multiplier=0.9)
        y_pred_share_cf = model.predict(X_test_cf)
        y_pred_units_cf = np.exp(y_pred_share_cf)
    else:
        # For regular models trained on share difference
        y_pred_share = model.predict(data_dict['X_test'])
        y_pred_units = np.exp(y_pred_share)
        
        X_test_cf = create_counterfactual_features(data_dict['X_test'], price_multiplier=0.9)
        y_pred_units_cf = np.exp(model.predict(X_test_cf))
    
    # Get actual units
    actual = np.exp(data_dict['df_test']['logunits'])
    
    # Calculate weights for WMAPE
    weights = actual * data_dict['df_test']['price'].values
    
    # Evaluate
    metrics = evaluate_model(actual, y_pred_units, weights=weights,
                            forecast_counterfactual=y_pred_units_cf)
    
    return {
        'model': model,
        'predictions': y_pred_units,
        'predictions_cf': y_pred_units_cf,
        'metrics': metrics
    }
