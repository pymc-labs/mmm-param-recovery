"""Model evaluation functions for performance metrics."""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from meridian.analysis import analyzer


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R-squared.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        R-squared value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) < 3:
        return np.nan
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - ss_res / ss_tot


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error. Only takes data points that are not nan and not 0.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE as percentage
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0)
    
    if mask.sum() == 0:
        return np.nan
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate bias. Only takes data points that are not nan.
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    return np.mean(predicted - actual)


def calculate_srmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate standardised root mean squared error. Only takes data points that are not nan.
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    return np.sqrt(np.mean((actual - predicted) ** 2)) / np.mean(actual)


def calculate_durbin_watson(residuals: np.ndarray) -> float:
    """Calculate Durbin-Watson statistic.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals (actual - predicted)
        
    Returns
    -------
    float
        Durbin-Watson statistic
    """
    mask = ~np.isnan(residuals)
    residuals = residuals[mask]
    
    if len(residuals) < 3:
        return np.nan
    
    return durbin_watson(residuals)


def evaluate_meridian_fit(
    meridian_model: Any,
    data_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Evaluate Meridian model fit.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    data_df : pd.DataFrame
        Original dataset
        
    Returns
    -------
    List[Dict[str, Any]]
        Performance metrics per geo
    """
    results = []
    model_analysis = analyzer.Analyzer(meridian_model)
    fit_data = model_analysis.expected_vs_actual_data()
    
    for geo in fit_data.geo.values:
        geo_label = "geo_a" if geo == "national_geo" else geo
        
        expected = fit_data['expected'].sel(geo=geo, metric="mean").values
        actual = fit_data['actual'].sel(geo=geo).values
        
        r2 = calculate_r2(actual, expected)
        mape = calculate_mape(actual, expected)
        bias = calculate_bias(actual, expected)
        srmse = calculate_srmse(actual, expected)
        dw = calculate_durbin_watson(actual - expected)
        
        results.append({
            "Library": "Meridian",
            "Geo": geo_label,
            "R²": round(r2, 4) if not np.isnan(r2) else None,
            "MAPE (%)": round(mape, 2) if not np.isnan(mape) else None,
            "Bias": round(bias, 4) if not np.isnan(bias) else None,
            "SRMSE": round(srmse, 4) if not np.isnan(srmse) else None,
            "Durbin-Watson": round(dw, 3) if not np.isnan(dw) else None
        })
    
    return results


def evaluate_pymc_fit(
    pymc_model: Any,
    data_df: pd.DataFrame,
    sampler: str
) -> List[Dict[str, Any]]:
    """Evaluate PyMC-Marketing model fit.
    
    Parameters
    ----------
    pymc_model : MMM
        Fitted PyMC-Marketing model
    data_df : pd.DataFrame
        Original dataset
    sampler : str
        Sampler name
        
    Returns
    -------
    List[Dict[str, Any]]
        Performance metrics per geo
    """
    results = []
    
    for geo in pymc_model.model.coords["geo"]:
        geo_label = "geo_a" if geo == "Local" else geo
        
        expected = pymc_model.idata["posterior_predictive"].y_original_scale.mean(
            ['chain', 'draw']
        ).sel(geo=geo).values
        
        actual = data_df.loc[data_df["geo"] == geo, "y"].values
        
        r2 = calculate_r2(actual, expected)
        mape = calculate_mape(actual, expected)
        bias = calculate_bias(actual, expected)
        srmse = calculate_srmse(actual, expected)
        dw = calculate_durbin_watson(actual - expected)
        
        results.append({
            "Library": f"PyMC-Marketing - {sampler}",
            "Geo": geo_label,
            "R²": round(r2, 4) if not np.isnan(r2) else None,
            "MAPE (%)": round(mape, 2) if not np.isnan(mape) else None,
            "Bias": round(bias, 4) if not np.isnan(bias) else None,
            "SRMSE": round(srmse, 4) if not np.isnan(srmse) else None,
            "Durbin-Watson": round(dw, 3) if not np.isnan(dw) else None
        })
    
    return results


def create_performance_summary(
    all_performance_rows: List[Dict[str, Any]],
    dataset_names: List[str]
) -> pd.DataFrame:
    """Create performance summary dataframe.
    
    Parameters
    ----------
    all_performance_rows : List[Dict]
        All performance metric rows
    dataset_names : List[str]
        List of dataset names for ordering
        
    Returns
    -------
    pd.DataFrame
        Performance summary in wide format
    """
    performance_df = pd.DataFrame(all_performance_rows)
    
    dataset_order = {name: i for i, name in enumerate(dataset_names)}
    performance_df["dataset_order"] = performance_df["Dataset"].map(dataset_order)
    performance_df = performance_df.sort_values(
        by=["dataset_order", "Geo", "Library"]
    ).drop(columns="dataset_order")
    
    melted_df = performance_df.melt(
        id_vars=["Dataset", "Geo", "Library"],
        value_vars=["R²", "MAPE (%)", "Durbin-Watson"],
        var_name="Metric",
        value_name="Value"
    )
    
    final_df = melted_df.pivot_table(
        index=["Dataset", "Geo", "Metric"],
        columns="Library",
        values="Value"
    ).reset_index()
    
    final_df.columns.name = None
    
    return final_df


def calculate_contribution_recovery(
    ground_truth: pd.DataFrame,
    meridian_model: Any,
    pymc_model: Any,
    channel_columns: List[str]
) -> Dict[str, float]:
    """Calculate contribution recovery metrics.
    
    Parameters
    ----------
    ground_truth : pd.DataFrame
        Ground truth contributions
    meridian_model : Any
        Fitted Meridian model
    pymc_model : Any
        Fitted PyMC model
    channel_columns : List[str]
        Channel column names
        
    Returns
    -------
    Dict[str, float]
        Contribution recovery metrics
    """
    # This is a placeholder - implement actual contribution recovery logic
    # based on the notebook implementation
    return {
        "meridian_recovery_r2": 0.0,
        "pymc_recovery_r2": 0.0
    }