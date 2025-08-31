"""Bayesian metric calculation functions with proper uncertainty propagation."""

import numpy as np
import arviz as az
from typing import Dict, Any


def calculate_r2_vectorized(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate R-squared for multiple predictions simultaneously.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (n_times,)
    predicted : np.ndarray
        Predicted values, shape (n_samples, n_times) where n_samples = n_chains × n_draws
        
    Returns
    -------
    np.ndarray
        R-squared values, shape (n_samples,)
    """
    # Handle NaNs
    mask = ~np.isnan(actual)
    actual_clean = actual[mask]
    predicted_clean = predicted[:, mask]
    
    if len(actual_clean) < 3:
        return np.full(predicted.shape[0], np.nan)
    
    actual_mean = np.mean(actual_clean)
    ss_tot = np.sum((actual_clean - actual_mean) ** 2)
    
    if ss_tot == 0:
        return np.full(predicted.shape[0], np.nan)
    
    # Compute residuals for each sample
    residuals = predicted_clean - actual_clean[None, :]
    ss_res = np.sum(residuals ** 2, axis=1)
    
    return 1 - ss_res / ss_tot


def calculate_mape_vectorized(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate Mean Absolute Percentage Error for multiple predictions.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (n_times,)
    predicted : np.ndarray
        Predicted values, shape (n_samples, n_times)
        
    Returns
    -------
    np.ndarray
        MAPE values as percentages, shape (n_samples,)
    """
    # Create mask for valid values (not NaN and not too close to zero)
    mask = ~np.isnan(actual) & (np.abs(actual) > 1e-3)
    
    if mask.sum() == 0:
        return np.full(predicted.shape[0], np.nan)
    
    actual_clean = actual[mask]
    predicted_clean = predicted[:, mask]
    
    # Calculate MAPE for each sample
    abs_pct_errors = np.abs((predicted_clean - actual_clean[None, :]) / actual_clean[None, :])
    mape_values = np.mean(abs_pct_errors, axis=1) * 100
    
    return mape_values


def calculate_bias_vectorized(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate bias for multiple predictions.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (n_times,)
    predicted : np.ndarray
        Predicted values, shape (n_samples, n_times)
        
    Returns
    -------
    np.ndarray
        Bias values, shape (n_samples,)
    """
    # Handle NaNs
    mask = ~np.isnan(actual)
    actual_clean = actual[mask]
    predicted_clean = predicted[:, mask]
    
    if len(actual_clean) == 0:
        return np.full(predicted.shape[0], np.nan)
    
    return np.mean(predicted_clean - actual_clean[None, :], axis=1)


def calculate_srmse_vectorized(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate standardized root mean squared error for multiple predictions.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (n_times,)
    predicted : np.ndarray
        Predicted values, shape (n_samples, n_times)
        
    Returns
    -------
    np.ndarray
        SRMSE values as fractions, shape (n_samples,)
    """
    # Handle NaNs
    mask = ~np.isnan(actual)
    actual_clean = actual[mask]
    predicted_clean = predicted[:, mask]
    
    if len(actual_clean) == 0:
        return np.full(predicted.shape[0], np.nan)
    
    mean_actual = np.mean(actual_clean)
    
    if mean_actual == 0:
        return np.full(predicted.shape[0], np.nan)
    
    # Calculate RMSE for each sample
    squared_errors = (predicted_clean - actual_clean[None, :]) ** 2
    rmse_values = np.sqrt(np.mean(squared_errors, axis=1))
    
    return rmse_values / np.abs(mean_actual)


def calculate_durbin_watson_vectorized(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate Durbin-Watson statistic for multiple predictions.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (n_times,)
    predicted : np.ndarray
        Predicted values, shape (n_samples, n_times)
        
    Returns
    -------
    np.ndarray
        Durbin-Watson values, shape (n_samples,)
    """
    from statsmodels.stats.stattools import durbin_watson
    
    # Handle NaNs
    mask = ~np.isnan(actual)
    actual_clean = actual[mask]
    predicted_clean = predicted[:, mask]
    
    if len(actual_clean) < 3:
        return np.full(predicted.shape[0], np.nan)
    
    # Calculate Durbin-Watson for each sample
    dw_values = np.zeros(predicted.shape[0])
    for i in range(predicted.shape[0]):
        residuals = actual_clean - predicted_clean[i, :]
        dw_values[i] = durbin_watson(residuals)
    
    return dw_values


def compute_summary_stats(metric_array: np.ndarray, hdi_prob: float = 0.90) -> Dict[str, float]:
    """Compute summary statistics for a metric across posterior samples using HDI.
    
    Parameters
    ----------
    metric_array : np.ndarray
        Array of metric values across posterior samples
    hdi_prob : float
        HDI probability (default 0.90 for 90% HDI)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with summary statistics including HDI bounds
    """
    # Remove NaN values for statistics
    valid_values = metric_array[~np.isnan(metric_array)]
    
    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'hdi_lower': np.nan,
            'hdi_upper': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    # Compute HDI using ArviZ
    hdi_bounds = az.hdi(valid_values, hdi_prob=hdi_prob)
    
    return {
        'mean': float(np.mean(valid_values)),
        'std': float(np.std(valid_values)),
        'median': float(np.median(valid_values)),
        'hdi_lower': float(hdi_bounds[0]),
        'hdi_upper': float(hdi_bounds[1]),
        'min': float(np.min(valid_values)),
        'max': float(np.max(valid_values))
    }


def format_metric_with_hdi(stats: Dict[str, float], precision: int = 2) -> str:
    """Format a metric with HDI (Highest Density Interval).
    
    Parameters
    ----------
    stats : Dict[str, float]
        Dictionary with summary statistics including HDI bounds
    precision : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted string like "0.95 ± 0.02 [0.91, 0.98]"
    """
    if np.isnan(stats['mean']):
        return "N/A"
    
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(stats['mean'])} ± {fmt.format(stats['std'])} [{fmt.format(stats['hdi_lower'])}, {fmt.format(stats['hdi_upper'])}]"