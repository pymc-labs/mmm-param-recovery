"""Model evaluation functions for performance metrics."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import xarray as xr
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
    """Calculate Mean Absolute Percentage Error. Only takes data points that are not nan and close to 0.
    
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
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (np.abs(actual) > 1e-3)
    
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
    
    Returns sRMSE as a fraction (not percentage).
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return np.nan
    
    mean_actual = np.mean(actual)
    
    if mean_actual == 0:
        return np.nan
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse / np.abs(mean_actual)


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
            "Model": "Meridian",
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
            "Model": f"PyMC-Marketing - {sampler}",
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
        by=["dataset_order", "Geo", "Model"]
    ).drop(columns="dataset_order")
    
    melted_df = performance_df.melt(
        id_vars=["Dataset", "Geo", "Model"],
        value_vars=["R²", "MAPE (%)", "Durbin-Watson"],
        var_name="Metric",
        value_name="Value"
    )
    
    final_df = melted_df.pivot_table(
        index=["Dataset", "Geo", "Metric"],
        columns="Model",
        values="Value"
    ).reset_index()
    
    final_df.columns.name = None
    
    return final_df


def extract_meridian_channel_contributions(
    meridian_model: Any,
    channel_columns: List[str]
) -> pd.DataFrame:
    """Extract channel contributions from Meridian model.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    channel_columns : List[str]
        Channel column names
        
    Returns
    -------
    pd.DataFrame
        Channel contributions with MultiIndex (date, geo) and columns for each channel
    """
    import tensorflow as tf
    
    model_analysis = analyzer.Analyzer(meridian_model)
    
    # Get incremental outcomes per geo
    # IMPORTANT: aggregate_geos defaults to True, we need False for per-geo contributions
    incremental_outcomes = model_analysis.incremental_outcome(
        aggregate_times=False,
        aggregate_geos=False,  # Critical: get per-geo contributions
        use_kpi=False,  # Use revenue, not KPI
        use_posterior=True
    )
    
    # Take mean across chains and draws
    # Shape should be (n_chains, n_draws, n_geos, n_times, n_channels) when aggregate_geos=False
    mean_contributions = tf.reduce_mean(incremental_outcomes, axis=[0, 1])
    
    # Convert to numpy array - should be (n_geos, n_times, n_channels)
    mean_contributions_np = mean_contributions.numpy()
    
    # Get geo and time coordinates from the model
    geos = meridian_model.input_data.geo.coords["geo"].values
    times = meridian_model.input_data.kpi.coords["time"].values
    n_geos = len(geos)
    n_times = len(times)
    
    # Map Meridian geo names to dataset geo names
    # Meridian uses 'national_geo' for single-geo models, but dataset uses actual geo name
    if n_geos == 1 and geos[0] == 'national_geo':
        # For single-geo models, use 'Local' which is the common geo name in small_business
        geos = ['Local']
    
    # Validate shape
    if len(mean_contributions_np.shape) == 2:
        # Shape is (n_times, n_channels) - single geo without geo dimension
        n_channels = mean_contributions_np.shape[1]
        # Reshape to add geo dimension
        mean_contributions_np = mean_contributions_np.reshape(1, n_times, n_channels)
    elif len(mean_contributions_np.shape) == 3:
        # Shape is (n_geos, n_times, n_channels) - expected shape for multi-geo
        if mean_contributions_np.shape[0] != n_geos:
            raise ValueError(f"Geo dimension mismatch: got {mean_contributions_np.shape[0]}, expected {n_geos}")
        if mean_contributions_np.shape[1] != n_times:
            raise ValueError(f"Time dimension mismatch: got {mean_contributions_np.shape[1]}, expected {n_times}")
    else:
        raise ValueError(f"Unexpected shape for mean_contributions: {mean_contributions_np.shape}")
    
    # Create DataFrame with proper structure
    contrib_list = []
    for geo_idx in range(n_geos):
        geo = geos[geo_idx]
        
        # Create MultiIndex for this geo
        index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(times).strftime('%Y-%m-%d'), [geo] * n_times],
            names=['date', 'geo']
        )
        
        geo_df = pd.DataFrame(
            mean_contributions_np[geo_idx, :, :],
            columns=channel_columns[:mean_contributions_np.shape[2]],
            index=index
        )
        contrib_list.append(geo_df)
    
    return pd.concat(contrib_list)


def extract_pymc_channel_contributions(
    pymc_model: Any,
    channel_columns: List[str]
) -> pd.DataFrame:
    """Extract channel contributions from PyMC-Marketing model.
    
    Parameters
    ----------
    pymc_model : MMM
        Fitted PyMC-Marketing model
    channel_columns : List[str]
        Channel column names
        
    Returns
    -------
    pd.DataFrame
        Channel contributions with MultiIndex (date, geo) and columns for each channel
    """
    # Get channel contributions - shape: (chain, draw, date, geo, channel)
    contribution_da = pymc_model.idata["posterior"]["channel_contribution_original_scale"]
    
    # Take mean across chains and draws
    mean_contrib = contribution_da.mean(['chain', 'draw'])
    
    # Convert to DataFrame with proper structure
    contrib_list = []
    for geo_idx, geo in enumerate(mean_contrib.geo.values):
        # Convert dates to string format for consistency
        dates = pd.to_datetime(mean_contrib.date.values).strftime('%Y-%m-%d')
        
        # Create MultiIndex for this geo
        index = pd.MultiIndex.from_arrays(
            [dates, [geo] * len(dates)],
            names=['date', 'geo']
        )
        
        geo_df = pd.DataFrame(
            mean_contrib.sel(geo=geo).values,
            columns=channel_columns,
            index=index
        )
        contrib_list.append(geo_df)
    
    return pd.concat(contrib_list)


def evaluate_channel_contributions(
    true_contributions: pd.DataFrame,
    predicted_contributions: pd.DataFrame,
    channel_columns: List[str],
    model_name: str,
    dataset_name: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate channel contribution reconstruction per channel-region combination.
    
    Parameters
    ----------
    true_contributions : pd.DataFrame
        Ground truth contributions with columns like 'contribution_x1'
    predicted_contributions : pd.DataFrame
        Predicted contributions with columns like 'x1'
    channel_columns : List[str]
        Channel column names
    model_name : str
        Name of the model (e.g., "Meridian", "PyMC-Marketing - pymc")
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - DataFrame with per-channel-region metrics (for CSV export)
        - Dictionary with averaged metrics across channels and regions
    """
    channel_metrics = []
    
    # Fix index structure for true_contributions if needed
    if 'time' in true_contributions.columns and 'geo' in true_contributions.columns:
        # truth_df was reset_index, so recreate the MultiIndex
        true_contributions['time'] = true_contributions['time'].astype(str)
        true_contributions = true_contributions.set_index(['time', 'geo'])
        # Rename index to match predicted
        true_contributions.index.names = ['date', 'geo']
    
    # Get unique regions from the data
    regions = predicted_contributions.index.get_level_values('geo').unique()
    
    for region in regions:
        # Filter data for this region
        true_region = true_contributions[true_contributions.index.get_level_values('geo') == region]
        pred_region = predicted_contributions[predicted_contributions.index.get_level_values('geo') == region]
        
        for channel in channel_columns:
            # Get true contribution column name
            true_col = f"contribution_{channel}"
            
            if true_col not in true_region.columns:
                continue
                
            if channel not in pred_region.columns:
                continue
                
            # Get values for this channel-region combination
            true_values = true_region[true_col].values
            pred_values = pred_region[channel].values
            
            # Ensure same length (should be after filtering by region)
            if len(true_values) != len(pred_values):
                print(f"Warning: Length mismatch for {channel} in {region}: true={len(true_values)}, pred={len(pred_values)}")
                continue
            
            # Calculate metrics for this channel-region combination
            bias = calculate_bias(true_values, pred_values)
            srmse = calculate_srmse(true_values, pred_values)
            r2 = calculate_r2(true_values, pred_values)
            mape = calculate_mape(true_values, pred_values)
            
            channel_metrics.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Channel": channel,
                "Region": region,
                "Bias": bias,
                "SRMSE": srmse,
                "R²": r2,
                "MAPE (%)": mape
            })
    
    # Create DataFrame for per-channel-region results
    channel_df = pd.DataFrame(channel_metrics)
    
    # Calculate averages across all channel-region combinations
    if len(channel_df) > 0:
        avg_metrics = {
            "Bias": channel_df["Bias"].mean(),
            "SRMSE": channel_df["SRMSE"].mean(),
            "R²": channel_df["R²"].mean(),
            "MAPE (%)": channel_df["MAPE (%)"].mean()
        }
    else:
        avg_metrics = {
            "Bias": np.nan,
            "SRMSE": np.nan,
            "R²": np.nan,
            "MAPE (%)": np.nan
        }
    
    return channel_df, avg_metrics


def evaluate_meridian_channel_contributions(
    meridian_model: Any,
    truth_df: pd.DataFrame,
    channel_columns: List[str],
    dataset_name: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate Meridian channel contribution reconstruction.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    truth_df : pd.DataFrame
        Ground truth dataframe with contribution columns
    channel_columns : List[str]
        Channel column names
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Per-channel metrics DataFrame and averaged metrics
    """
    predicted_contrib = extract_meridian_channel_contributions(
        meridian_model, channel_columns
    )
    
    return evaluate_channel_contributions(
        truth_df, predicted_contrib, channel_columns,
        "Meridian", dataset_name
    )


def evaluate_pymc_channel_contributions(
    pymc_model: Any,
    truth_df: pd.DataFrame,
    channel_columns: List[str],
    sampler: str,
    dataset_name: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate PyMC-Marketing channel contribution reconstruction.
    
    Parameters
    ----------
    pymc_model : MMM
        Fitted PyMC-Marketing model
    truth_df : pd.DataFrame
        Ground truth dataframe with contribution columns
    channel_columns : List[str]
        Channel column names
    sampler : str
        Sampler name
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Per-channel metrics DataFrame and averaged metrics
    """
    predicted_contrib = extract_pymc_channel_contributions(
        pymc_model, channel_columns
    )
    
    return evaluate_channel_contributions(
        truth_df, predicted_contrib, channel_columns,
        f"PyMC-Marketing - {sampler}", dataset_name
    )