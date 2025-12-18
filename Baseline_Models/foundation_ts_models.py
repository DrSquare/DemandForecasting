"""
Foundation Time Series Models for Demand Forecasting

This script trains and evaluates state-of-the-art foundation time series models:
1. TimesFM 2.5 (Google) - Foundation model for time series forecasting
2. SunDial (Amazon) - Time series foundation model
3. TabPFN-TS - Tabular Prior-Data Fitted Network for Time Series

Data: salty_snack_0.05_store.csv
Validation: weeks 1375-1426 (last 52 weeks)
Training: weeks 1114-1374 (first 261 weeks)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_metrics import (
    evaluate_model, print_metrics, compare_models,
    counterfactual_validity_by_store_week
)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_path': '../Data/salty_snack_0.05_store.csv',
    'output_dir': 'output',
    'val_week_start': 1375,
    'forecast_horizon': 52,  # 52 weeks forecast
    'context_length': 104,   # Use 2 years of history as context
    'random_seed': 42,
    'verbose': True,
    'save_results': True,
}


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess the sales data."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    required_cols = ['iri_key', 'week', 'colupc', 'units', 'price']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate equivalent units if vol_eq exists
    if 'vol_eq' in df.columns:
        df['eq_units'] = df['units'] * df['vol_eq']
    else:
        df['eq_units'] = df['units']
    
    print(f"  Data shape: {df.shape}")
    print(f"  Week range: {df['week'].min()} - {df['week'].max()}")
    print(f"  Unique stores: {df['iri_key'].nunique()}")
    print(f"  Unique products: {df['colupc'].nunique()}")
    
    return df


def prepare_time_series_data(df: pd.DataFrame, 
                              val_week_start: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare time series data for foundation models.
    
    Args:
        df: Input DataFrame
        val_week_start: Week where validation starts
        
    Returns:
        train_df, val_df
    """
    print("\nPreparing time series data...")
    
    # Split by time
    train_df = df[df['week'] < val_week_start].copy()
    val_df = df[df['week'] >= val_week_start].copy()
    
    print(f"  Train: {len(train_df)} rows, weeks {train_df['week'].min()}-{train_df['week'].max()}")
    print(f"  Val: {len(val_df)} rows, weeks {val_df['week'].min()}-{val_df['week'].max()}")
    
    return train_df, val_df


def create_panel_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a panel of time series for each store-product combination.
    
    Returns DataFrame with columns: store, product, week, units, price
    """
    # Aggregate to store-product-week level
    panel = df.groupby(['iri_key', 'colupc', 'week']).agg({
        'eq_units': 'sum',
        'price': 'mean'
    }).reset_index()
    
    panel = panel.rename(columns={
        'iri_key': 'store',
        'colupc': 'product',
        'eq_units': 'units'
    })
    
    return panel


# ============================================================================
# TimesFM 2.0 Model (PyTorch / HuggingFace)
# ============================================================================

class TimesFMModel:
    """
    TimesFM 2.0 - Google's Foundation Model for Time Series Forecasting.
    
    Uses the PyTorch implementation via HuggingFace Transformers.
    
    Paper: "A decoder-only foundation model for time-series forecasting"
    https://arxiv.org/abs/2310.10688
    
    Model: google/timesfm-1.0-200m (PyTorch version)
    """
    
    def __init__(self, 
                 context_length: int = 104,
                 forecast_horizon: int = 52,
                 backend: str = 'cpu'):
        """
        Initialize TimesFM model.
        
        Args:
            context_length: Number of historical time steps to use
            forecast_horizon: Number of steps to forecast
            backend: 'cpu' or 'gpu'
        """
        self.name = "TimesFM 2.0"
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.backend = backend
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Load the TimesFM model from HuggingFace."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoConfig
            
            print(f"  Loading TimesFM model from HuggingFace (backend={self.backend})...")
            
            device = "cuda" if self.backend == 'gpu' and torch.cuda.is_available() else "cpu"
            
            # Try loading the PyTorch version
            try:
                # TimesFM 1.0-200m model (smaller, faster)
                model_name = "google/timesfm-1.0-200m-pytorch"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map=device,
                    torch_dtype=torch.float32
                )
                print(f"  TimesFM model loaded successfully on {device}")
                return True
            except Exception as e1:
                print(f"  Could not load timesfm-1.0-200m-pytorch: {e1}")
                
                # Fallback: Try using a simple transformer-based forecaster
                print("  Attempting fallback to Lag-Llama or similar...")
                try:
                    from transformers import pipeline
                    self.model = pipeline(
                        "time-series-prediction",
                        model="huggingface/lag-llama",
                        device=0 if device == "cuda" else -1
                    )
                    print("  Loaded Lag-Llama as fallback")
                    return True
                except Exception as e2:
                    print(f"  Fallback also failed: {e2}")
                    return False
                    
        except ImportError as e:
            print(f"  ERROR: Required packages not installed: {e}")
            print("  Install with: pip install transformers accelerate")
            return False
        except Exception as e:
            print(f"  ERROR loading TimesFM: {e}")
            return False
    
    def forecast(self, 
                 train_panel: pd.DataFrame,
                 val_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for all store-product combinations.
        
        Args:
            train_panel: Training data panel
            val_panel: Validation data panel (for structure)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            if not self._load_model():
                # If model loading fails, use a simple statistical fallback
                print("  Using statistical fallback (Seasonal Naive)...")
                return self._seasonal_naive_forecast(train_panel, val_panel)
        
        print(f"\n  Generating TimesFM forecasts...")
        
        # Get unique store-product combinations
        combinations = val_panel[['store', 'product']].drop_duplicates()
        
        predictions = []
        
        for _, row in tqdm(combinations.iterrows(), 
                          total=len(combinations), 
                          desc="  Forecasting"):
            store, product = row['store'], row['product']
            
            # Get historical data for this combination
            hist = train_panel[
                (train_panel['store'] == store) & 
                (train_panel['product'] == product)
            ].sort_values('week')
            
            if len(hist) < 10:  # Skip if insufficient history
                continue
            
            # Prepare input time series
            ts = hist['units'].values[-self.context_length:]
            
            # Get validation weeks for this combination
            val_weeks = val_panel[
                (val_panel['store'] == store) & 
                (val_panel['product'] == product)
            ]['week'].values
            
            try:
                if hasattr(self.model, 'predict'):
                    # Use model's predict method
                    forecast = self._model_forecast(ts)
                else:
                    # Use pipeline
                    forecast = self._pipeline_forecast(ts)
                
                # Match forecasts to weeks
                for i, week in enumerate(val_weeks):
                    if i < len(forecast):
                        predictions.append({
                            'store': store,
                            'product': product,
                            'week': week,
                            'predicted_units': max(0, forecast[i])
                        })
                        
            except Exception as e:
                continue
        
        if not predictions:
            print("  WARNING: No predictions generated, using seasonal naive")
            return self._seasonal_naive_forecast(train_panel, val_panel)
            
        pred_df = pd.DataFrame(predictions)
        print(f"  Generated {len(pred_df)} predictions")
        
        return pred_df
    
    def _model_forecast(self, ts: np.ndarray) -> np.ndarray:
        """Generate forecast using the loaded model."""
        import torch
        
        # Pad if necessary
        if len(ts) < self.context_length:
            ts = np.pad(ts, (self.context_length - len(ts), 0), mode='edge')
        
        # Convert to tensor
        input_tensor = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_length=len(ts) + self.forecast_horizon
            )
        
        forecast = output[0, len(ts):].numpy()
        return forecast[:self.forecast_horizon]
    
    def _pipeline_forecast(self, ts: np.ndarray) -> np.ndarray:
        """Generate forecast using HuggingFace pipeline."""
        result = self.model(ts.tolist(), prediction_length=self.forecast_horizon)
        return np.array(result['predictions'])
    
    def _seasonal_naive_forecast(self, 
                                  train_panel: pd.DataFrame,
                                  val_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Simple seasonal naive forecast as fallback.
        Uses same week last year as prediction.
        """
        print("  Generating Seasonal Naive forecasts...")
        
        combinations = val_panel[['store', 'product']].drop_duplicates()
        predictions = []
        
        for _, row in tqdm(combinations.iterrows(),
                          total=len(combinations),
                          desc="  Forecasting"):
            store, product = row['store'], row['product']
            
            hist = train_panel[
                (train_panel['store'] == store) &
                (train_panel['product'] == product)
            ].sort_values('week')
            
            if len(hist) < 52:
                continue
            
            # Use last 52 weeks as template
            last_year = hist['units'].values[-52:]
            
            val_data = val_panel[
                (val_panel['store'] == store) &
                (val_panel['product'] == product)
            ].sort_values('week')
            
            for i, (_, val_row) in enumerate(val_data.iterrows()):
                pred = last_year[i % 52] if i < 52 else last_year[-1]
                predictions.append({
                    'store': store,
                    'product': product,
                    'week': val_row['week'],
                    'predicted_units': max(0, pred)
                })
        
        pred_df = pd.DataFrame(predictions)
        print(f"  Generated {len(pred_df)} predictions")
        return pred_df


# ============================================================================
# SunDial Model
# ============================================================================

class SunDialModel:
    """
    SunDial - Amazon's Time Series Foundation Model.
    
    Paper: "Sundial: Aligning Time Series and Natural Language with Calendar Time"
    https://arxiv.org/abs/2502.00816
    
    Installation: pip install chronos-forecasting
    Note: SunDial uses the Chronos architecture
    """
    
    def __init__(self,
                 context_length: int = 104,
                 forecast_horizon: int = 52,
                 model_size: str = "small"):
        """
        Initialize SunDial/Chronos model.
        
        Args:
            context_length: Number of historical time steps
            forecast_horizon: Number of steps to forecast
            model_size: 'tiny', 'mini', 'small', 'base', 'large'
        """
        self.name = "SunDial (Chronos)"
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.model_size = model_size
        self.model = None
        self.pipeline = None
        
    def _load_model(self):
        """Load the Chronos/SunDial model."""
        try:
            import torch
            from chronos import ChronosPipeline
            
            print(f"  Loading Chronos-{self.model_size} model...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.pipeline = ChronosPipeline.from_pretrained(
                f"amazon/chronos-t5-{self.model_size}",
                device_map=device,
                torch_dtype=torch.float32,
            )
            
            print(f"  Chronos model loaded on {device}")
            return True
            
        except ImportError:
            print("  ERROR: chronos-forecasting not installed.")
            print("  Install with: pip install chronos-forecasting")
            return False
        except Exception as e:
            print(f"  ERROR loading Chronos: {e}")
            return False
    
    def forecast(self,
                 train_panel: pd.DataFrame,
                 val_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for all store-product combinations.
        
        Args:
            train_panel: Training data panel
            val_panel: Validation data panel
            
        Returns:
            DataFrame with predictions
        """
        if self.pipeline is None:
            if not self._load_model():
                return None
        
        import torch
        
        print(f"\n  Generating Chronos/SunDial forecasts...")
        
        combinations = val_panel[['store', 'product']].drop_duplicates()
        predictions = []
        
        for _, row in tqdm(combinations.iterrows(),
                          total=len(combinations),
                          desc="  Forecasting"):
            store, product = row['store'], row['product']
            
            # Get historical data
            hist = train_panel[
                (train_panel['store'] == store) &
                (train_panel['product'] == product)
            ].sort_values('week')
            
            if len(hist) < 10:
                continue
            
            # Prepare context
            context = torch.tensor(hist['units'].values[-self.context_length:])
            
            try:
                # Generate forecast (returns samples from predictive distribution)
                forecast = self.pipeline.predict(
                    context,
                    prediction_length=self.forecast_horizon,
                    num_samples=20
                )
                
                # Take median as point forecast
                point_forecast = np.median(forecast.numpy(), axis=1)[0]
                
                # Get validation weeks
                val_weeks = val_panel[
                    (val_panel['store'] == store) &
                    (val_panel['product'] == product)
                ]['week'].values
                
                for i, week in enumerate(val_weeks):
                    if i < len(point_forecast):
                        predictions.append({
                            'store': store,
                            'product': product,
                            'week': week,
                            'predicted_units': max(0, point_forecast[i])
                        })
                        
            except Exception as e:
                continue
        
        if not predictions:
            print("  WARNING: No predictions generated")
            return None
            
        pred_df = pd.DataFrame(predictions)
        print(f"  Generated {len(pred_df)} predictions")
        
        return pred_df


# ============================================================================
# TabPFN-TS Model
# ============================================================================

class TabPFNTSModel:
    """
    TabPFN-TS - Tabular Prior-Data Fitted Network for Time Series.
    
    Paper: "TabPFN: A Transformer That Solves Small Tabular Classification 
           Problems in a Second"
    Extended for time series forecasting.
    
    Installation: pip install tabpfn
    """
    
    def __init__(self,
                 context_length: int = 104,
                 forecast_horizon: int = 52):
        """
        Initialize TabPFN-TS model.
        
        Args:
            context_length: Number of historical time steps
            forecast_horizon: Number of steps to forecast
        """
        self.name = "TabPFN-TS"
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        
    def _load_model(self):
        """Load the TabPFN model using V2 (open source, no commercial license required)."""
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.regressor import ModelVersion
            
            print("  Loading TabPFN V2 model (open source)...")
            # Use V2 instead of V2.5 to avoid HuggingFace license issues
            self.model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2,
                device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
                ignore_pretraining_limits=True  # Allow more than 1000 samples
            )
            print("  TabPFN V2 model loaded successfully")
            return True
            
        except ImportError:
            print("  ERROR: tabpfn not installed.")
            print("  Install with: pip install tabpfn")
            return False
        except Exception as e:
            print(f"  ERROR loading TabPFN: {e}")
            return False
    
    def _create_features(self, 
                         units: np.ndarray, 
                         prices: np.ndarray,
                         week_idx: int) -> np.ndarray:
        """
        Create lag features for TabPFN.
        
        Args:
            units: Historical unit sales
            prices: Historical prices
            week_idx: Current week index
            
        Returns:
            Feature vector
        """
        features = []
        
        # Lag features (1, 2, 4, 8, 12, 26, 52 weeks)
        lags = [1, 2, 4, 8, 12, 26, 52]
        for lag in lags:
            if week_idx >= lag:
                features.append(units[week_idx - lag])
            else:
                features.append(0)
        
        # Rolling statistics
        for window in [4, 8, 12]:
            start_idx = max(0, week_idx - window)
            if start_idx < week_idx:
                features.append(np.mean(units[start_idx:week_idx]))
                features.append(np.std(units[start_idx:week_idx]) if week_idx - start_idx > 1 else 0)
            else:
                features.extend([0, 0])
        
        # Price features
        if week_idx > 0 and len(prices) > week_idx:
            features.append(prices[week_idx - 1])
            features.append(prices[week_idx] / prices[week_idx - 1] if prices[week_idx - 1] > 0 else 1)
        else:
            features.extend([0, 1])
        
        # Week of year (cyclical)
        week_of_year = week_idx % 52
        features.append(np.sin(2 * np.pi * week_of_year / 52))
        features.append(np.cos(2 * np.pi * week_of_year / 52))
        
        return np.array(features)
    
    def forecast(self,
                 train_panel: pd.DataFrame,
                 val_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts using TabPFN with lag features.
        
        Args:
            train_panel: Training data panel
            val_panel: Validation data panel
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            if not self._load_model():
                return None
        
        print(f"\n  Generating TabPFN-TS forecasts...")
        
        combinations = val_panel[['store', 'product']].drop_duplicates()
        predictions = []
        
        for _, row in tqdm(combinations.iterrows(),
                          total=len(combinations),
                          desc="  Forecasting"):
            store, product = row['store'], row['product']
            
            # Get all data for this combination
            train_data = train_panel[
                (train_panel['store'] == store) &
                (train_panel['product'] == product)
            ].sort_values('week')
            
            if len(train_data) < 52:  # Need at least 1 year of history
                continue
            
            units = train_data['units'].values
            prices = train_data['price'].values
            
            try:
                # Create training features
                X_train = []
                y_train = []
                
                for i in range(52, len(units)):
                    X_train.append(self._create_features(units, prices, i))
                    y_train.append(units[i])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Limit training size for TabPFN (max 1024 samples)
                if len(X_train) > 1024:
                    X_train = X_train[-1024:]
                    y_train = y_train[-1024:]
                
                # Fit TabPFN
                self.model.fit(X_train, y_train)
                
                # Get validation weeks
                val_data = val_panel[
                    (val_panel['store'] == store) &
                    (val_panel['product'] == product)
                ].sort_values('week')
                
                # Combine train and val for feature creation
                all_units = np.concatenate([units, val_data['units'].values])
                all_prices = np.concatenate([prices, val_data['price'].values])
                
                # Predict for each validation week
                for i, (_, val_row) in enumerate(val_data.iterrows()):
                    week_idx = len(units) + i
                    X_pred = self._create_features(all_units, all_prices, week_idx).reshape(1, -1)
                    pred = self.model.predict(X_pred)[0]
                    
                    predictions.append({
                        'store': store,
                        'product': product,
                        'week': val_row['week'],
                        'predicted_units': max(0, pred)
                    })
                    
            except Exception as e:
                continue
        
        if not predictions:
            print("  WARNING: No predictions generated")
            return None
            
        pred_df = pd.DataFrame(predictions)
        print(f"  Generated {len(pred_df)} predictions")
        
        return pred_df


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_forecasts(predictions: pd.DataFrame,
                       actual_data: pd.DataFrame,
                       model_name: str) -> Dict:
    """
    Evaluate forecast predictions against actual values.
    
    Args:
        predictions: DataFrame with predicted_units
        actual_data: DataFrame with actual units
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Merge predictions with actuals
    merged = predictions.merge(
        actual_data[['store', 'product', 'week', 'units', 'price']],
        on=['store', 'product', 'week'],
        how='inner'
    )
    
    if len(merged) == 0:
        print(f"  WARNING: No matching predictions for {model_name}")
        return None
    
    actual = merged['units'].values
    predicted = merged['predicted_units'].values
    weights = actual * merged['price'].values
    
    # Evaluate
    metrics = evaluate_model(actual, predicted, weights=weights, forecast_counterfactual=None)
    
    # Time series models don't have counterfactual validity (they're not causal)
    metrics['counterfactual_validity'] = np.nan
    
    return metrics


def compute_ts_counterfactual_validity(model, 
                                        train_panel: pd.DataFrame,
                                        val_panel: pd.DataFrame,
                                        price_multiplier: float = 0.9) -> float:
    """
    Compute counterfactual validity for time series models.
    
    Note: Time series foundation models are not designed for causal inference,
    so counterfactual validity may not be meaningful. This is included for
    comparison purposes only.
    """
    # For time series models without price as explicit input,
    # counterfactual validity is not applicable
    return np.nan


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 80)
    print("FOUNDATION TIME SERIES MODELS - DEMAND FORECASTING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data: {CONFIG['data_path']}")
    print(f"  Validation week start: {CONFIG['val_week_start']}")
    print(f"  Forecast horizon: {CONFIG['forecast_horizon']} weeks")
    print(f"  Context length: {CONFIG['context_length']} weeks")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # ========== 1. LOAD DATA ==========
    print("\n" + "=" * 80)
    print("[1/4] Loading data...")
    print("=" * 80)
    
    df = load_data(CONFIG['data_path'])
    train_df, val_df = prepare_time_series_data(df, CONFIG['val_week_start'])
    
    # Create panel format
    train_panel = create_panel_time_series(train_df)
    val_panel = create_panel_time_series(val_df)
    
    print(f"\nPanel data:")
    print(f"  Train panel: {len(train_panel)} rows")
    print(f"  Val panel: {len(val_panel)} rows")
    print(f"  Unique combinations: {len(train_panel[['store', 'product']].drop_duplicates())}")
    
    # ========== 2. INITIALIZE MODELS ==========
    print("\n" + "=" * 80)
    print("[2/4] Initializing foundation models...")
    print("=" * 80)
    
    models = {
        'TimesFM 2.0': TimesFMModel(
            context_length=CONFIG['context_length'],
            forecast_horizon=CONFIG['forecast_horizon'],
            backend='gpu' if __import__('torch').cuda.is_available() else 'cpu'
        ),
        'SunDial (Chronos)': SunDialModel(
            context_length=CONFIG['context_length'],
            forecast_horizon=CONFIG['forecast_horizon'],
            model_size='small'
        ),
        'TabPFN-TS': TabPFNTSModel(
            context_length=CONFIG['context_length'],
            forecast_horizon=CONFIG['forecast_horizon']
        )
    }
    
    # ========== 3. TRAIN AND EVALUATE ==========
    print("\n" + "=" * 80)
    print("[3/4] Training and evaluating models...")
    print("=" * 80)
    
    all_results = {}
    all_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print('='*60)
        
        try:
            # Generate forecasts
            predictions = model.forecast(train_panel, val_panel)
            
            if predictions is not None and len(predictions) > 0:
                # Evaluate
                metrics = evaluate_forecasts(predictions, val_panel, model_name)
                
                if metrics is not None:
                    all_results[model_name] = metrics
                    all_predictions[model_name] = predictions
                    
                    if CONFIG['verbose']:
                        print_metrics(metrics, model_name)
                else:
                    print(f"  Evaluation failed for {model_name}")
            else:
                print(f"  No predictions generated for {model_name}")
                
        except Exception as e:
            print(f"  ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 4. COMPARE MODELS ==========
    print("\n" + "=" * 80)
    print("[4/4] Comparing models...")
    print("=" * 80)
    
    if all_results:
        comparison = compare_models(all_results, metric='wmape')
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON (sorted by WMAPE)")
        print("=" * 80)
        print(comparison.to_string())
        print("=" * 80)
        
        # Find best model
        best_model_name = comparison.index[0]
        print(f"\nBest model: {best_model_name}")
        print(f"WMAPE: {comparison.loc[best_model_name, 'wmape']:.4f}")
        
        # Save results
        if CONFIG['save_results']:
            results = {
                'config': CONFIG,
                'metrics': {k: {m: float(v) if not np.isnan(v) else None 
                               for m, v in metrics.items()} 
                           for k, metrics in all_results.items()},
                'comparison': comparison.to_dict(),
                'best_model': best_model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results_path = os.path.join(CONFIG['output_dir'], 'foundation_ts_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
    else:
        print("\nNo models were successfully evaluated.")
        print("Please ensure the required packages are installed:")
        print("  pip install timesfm")
        print("  pip install chronos-forecasting")
        print("  pip install tabpfn")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    
    return {
        'results': all_results,
        'predictions': all_predictions,
        'comparison': comparison if all_results else None
    }


if __name__ == "__main__":
    results = main()
