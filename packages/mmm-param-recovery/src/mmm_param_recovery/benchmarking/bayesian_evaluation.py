# Copyright 2025 PyMC Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bayesian evaluation functions for proper uncertainty propagation."""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from meridian.analysis import analyzer

from . import bayesian_metrics


def extract_meridian_posterior_contributions(
    meridian_model: Any,
    channel_columns: List[str]
) -> np.ndarray:
    """Extract full posterior of channel contributions from Meridian model.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    channel_columns : List[str]
        Channel column names
        
    Returns
    -------
    np.ndarray
        Channel contributions, shape (n_samples, n_geos, n_times, n_channels)
        where n_samples = n_chains × n_draws
    """
    model_analysis = analyzer.Analyzer(meridian_model)
    
    # Get incremental outcomes per geo
    # Shape: (n_chains, n_draws, n_geos, n_times, n_channels)
    incremental_outcomes = model_analysis.incremental_outcome(
        aggregate_times=False,
        aggregate_geos=False,  # Critical: get per-geo contributions
        use_kpi=False,
        use_posterior=True
    )
    
    # Convert to numpy and reshape to (n_samples, n_geos, n_times, n_channels)
    contrib_np = incremental_outcomes.numpy()
    n_chains, n_draws, n_geos, n_times, n_channels = contrib_np.shape
    
    # Reshape to combine chains and draws
    contrib_reshaped = contrib_np.reshape(n_chains * n_draws, n_geos, n_times, n_channels)
    
    return contrib_reshaped


def extract_pymc_posterior_contributions(
    pymc_model: Any,
    channel_columns: List[str]
) -> np.ndarray:
    """Extract full posterior of channel contributions from PyMC model.
    
    Parameters
    ----------
    pymc_model : MMM
        Fitted PyMC-Marketing model
    channel_columns : List[str]
        Channel column names
        
    Returns
    -------
    np.ndarray
        Channel contributions, shape (n_samples, n_geos, n_times, n_channels)
        where n_samples = n_chains × n_draws
    """
    # Get channel contributions - shape: (chain, draw, date, geo, channel)
    contribution_da = pymc_model.idata["posterior"]["channel_contribution_original_scale"]
    
    # Convert to numpy array
    contrib_np = contribution_da.values
    n_chains, n_draws, n_times, n_geos, n_channels = contrib_np.shape
    
    # Reshape to (n_samples, n_geos, n_times, n_channels)
    # Note: PyMC has different dimension order than Meridian
    contrib_reshaped = contrib_np.reshape(n_chains * n_draws, n_times, n_geos, n_channels)
    contrib_reshaped = contrib_reshaped.transpose(0, 2, 1, 3)  # Reorder to match Meridian
    
    return contrib_reshaped


def extract_meridian_posterior_predictions(meridian_model: Any) -> np.ndarray:
    """Extract full posterior predictive from Meridian model.
    
    Returns
    -------
    np.ndarray
        Predictions, shape (n_samples, n_geos, n_times)
    """
    model_analysis = analyzer.Analyzer(meridian_model)
    
    # Get full posterior predictions using expected_outcome
    # This returns shape: (n_chains, n_draws, n_geos, n_times)
    expected_outcomes = model_analysis.expected_outcome(
        use_posterior=True,
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=True,  # Use KPI scale (same as y in data)
        batch_size=100
    )
    
    # Convert to numpy array
    predictions = expected_outcomes.numpy()
    n_chains, n_draws, n_geos, n_times = predictions.shape
    
    # Reshape to (n_samples, n_geos, n_times)
    predictions_reshaped = predictions.reshape(n_chains * n_draws, n_geos, n_times)
    
    return predictions_reshaped


def extract_pymc_posterior_predictions(pymc_model: Any) -> np.ndarray:
    """Extract full posterior predictive from PyMC model.
    
    Returns
    -------
    np.ndarray
        Predictions, shape (n_samples, n_geos, n_times)
    """
    # Shape: (chain, draw, date, geo)
    posterior_pred = pymc_model.idata["posterior_predictive"].y_original_scale.values
    n_chains, n_draws, n_times, n_geos = posterior_pred.shape
    
    # Reshape to (n_samples, n_geos, n_times)
    pred_reshaped = posterior_pred.reshape(n_chains * n_draws, n_times, n_geos)
    pred_reshaped = pred_reshaped.transpose(0, 2, 1)  # Reorder to (n_samples, n_geos, n_times)
    
    return pred_reshaped


def evaluate_revenue_bayesian(
    model: Any,
    data_df: pd.DataFrame,
    model_type: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate revenue predictions with proper Bayesian uncertainty.
    
    Parameters
    ----------
    model : Any
        Fitted model (Meridian or PyMC)
    data_df : pd.DataFrame
        Original dataset with actual values
    model_type : str
        'meridian' or 'pymc'
        
    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Nested dict: {geo: {metric_name: summary_stats}}
    """
    # Extract posterior predictions
    if model_type == 'meridian':
        predictions = extract_meridian_posterior_predictions(model)
        geo_names = model.input_data.geo.coords["geo"].values
        # Map Meridian geo names
        if len(geo_names) == 1 and geo_names[0] == 'national_geo':
            geo_names = ['Local']
    else:  # pymc
        predictions = extract_pymc_posterior_predictions(model)
        geo_names = model.model.coords["geo"]
    
    results = {}
    
    for geo_idx, geo in enumerate(geo_names):
        # Get actual values for this geo
        actual = data_df[data_df['geo'] == geo]['y'].values
        
        # Get predictions for this geo - shape: (n_samples, n_times)
        predicted = predictions[:, geo_idx, :]
        
        # Calculate metrics for all posterior samples
        r2_values = bayesian_metrics.calculate_r2_vectorized(actual, predicted)
        mape_values = bayesian_metrics.calculate_mape_vectorized(actual, predicted)
        bias_values = bayesian_metrics.calculate_bias_vectorized(actual, predicted)
        srmse_values = bayesian_metrics.calculate_srmse_vectorized(actual, predicted)
        dw_values = bayesian_metrics.calculate_durbin_watson_vectorized(actual, predicted)
        
        # Also calculate MAPE on posterior mean (traditional approach)
        # This matches how traditional metrics are calculated
        posterior_mean = np.mean(predicted, axis=0)
        from . import evaluation  # Import traditional MAPE calculator
        mape_posterior_mean = evaluation.calculate_mape(actual, posterior_mean)
        
        # Compute summary statistics
        results[geo] = {
            'R²': bayesian_metrics.compute_summary_stats(r2_values),
            'MAPE (%)': bayesian_metrics.compute_summary_stats(mape_values),
            'MAPE_posterior_mean (%)': mape_posterior_mean,  # Traditional-style MAPE
            'Bias': bayesian_metrics.compute_summary_stats(bias_values),
            'SRMSE': bayesian_metrics.compute_summary_stats(srmse_values),
            'Durbin-Watson': bayesian_metrics.compute_summary_stats(dw_values)
        }
    
    return results


def evaluate_contributions_bayesian(
    model: Any,
    truth_df: pd.DataFrame,
    channel_columns: List[str],
    model_type: str
) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    """Evaluate channel contributions with proper Bayesian uncertainty.
    
    Parameters
    ----------
    model : Any
        Fitted model (Meridian or PyMC)
    truth_df : pd.DataFrame
        Ground truth dataframe with contribution columns
    channel_columns : List[str]
        Channel column names
    model_type : str
        'meridian' or 'pymc'
        
    Returns
    -------
    Tuple[Dict, Dict]
        - Per-channel-geo metrics: {(geo, channel): {metric_name: summary_stats}}
        - Aggregated metrics across all channels/geos
    """
    # Extract posterior contributions
    if model_type == 'meridian':
        contributions = extract_meridian_posterior_contributions(model, channel_columns)
        geo_names = model.input_data.geo.coords["geo"].values
        if len(geo_names) == 1 and geo_names[0] == 'national_geo':
            geo_names = ['Local']
    else:  # pymc
        contributions = extract_pymc_posterior_contributions(model, channel_columns)
        geo_names = model.model.coords["geo"]
    
    # Fix truth_df index if needed
    if 'time' in truth_df.columns and 'geo' in truth_df.columns:
        truth_df['time'] = truth_df['time'].astype(str)
        truth_df = truth_df.set_index(['time', 'geo'])
        truth_df.index.names = ['date', 'geo']
    
    detailed_results = {}
    all_metric_arrays = defaultdict(list)  # For aggregation
    
    for geo_idx, geo in enumerate(geo_names):
        # Filter truth data for this geo
        truth_geo = truth_df[truth_df.index.get_level_values('geo') == geo]
        
        for channel_idx, channel in enumerate(channel_columns):
            # Get true contribution column
            true_col = f"contribution_{channel}"
            
            if true_col not in truth_geo.columns:
                continue
            
            # Get actual values
            actual = truth_geo[true_col].values
            
            # Get predictions for this geo/channel - shape: (n_samples, n_times)
            predicted = contributions[:, geo_idx, :, channel_idx]
            
            # Ensure same length
            if len(actual) != predicted.shape[1]:
                print(f"Warning: Length mismatch for {channel} in {geo}: actual={len(actual)}, pred={predicted.shape[1]}")
                continue
            
            # Calculate metrics for all posterior samples
            r2_values = bayesian_metrics.calculate_r2_vectorized(actual, predicted)
            mape_values = bayesian_metrics.calculate_mape_vectorized(actual, predicted)
            bias_values = bayesian_metrics.calculate_bias_vectorized(actual, predicted)
            srmse_values = bayesian_metrics.calculate_srmse_vectorized(actual, predicted)
            
            # Store detailed results
            detailed_results[(geo, channel)] = {
                'R²': bayesian_metrics.compute_summary_stats(r2_values),
                'MAPE (%)': bayesian_metrics.compute_summary_stats(mape_values),
                'Bias': bayesian_metrics.compute_summary_stats(bias_values),
                'SRMSE': bayesian_metrics.compute_summary_stats(srmse_values)
            }
            
            # Collect for aggregation
            all_metric_arrays['R²'].extend(r2_values)
            all_metric_arrays['MAPE (%)'].extend(mape_values)
            all_metric_arrays['Bias'].extend(bias_values)
            all_metric_arrays['SRMSE'].extend(srmse_values)
    
    # Compute aggregated statistics
    aggregated_results = {}
    for metric_name, values_list in all_metric_arrays.items():
        if values_list:
            aggregated_results[metric_name] = bayesian_metrics.compute_summary_stats(np.array(values_list))
        else:
            aggregated_results[metric_name] = bayesian_metrics.compute_summary_stats(np.array([np.nan]))
    
    return detailed_results, aggregated_results


def create_summary_dataframe(
    results_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metric_names: List[str] = ['R²', 'MAPE (%)']
) -> pd.DataFrame:
    """Create a summary DataFrame from Bayesian evaluation results.
    
    Parameters
    ----------
    results_dict : Dict
        Nested dict with structure {dataset: {model: {geo: {metric: stats}}}}
    metric_names : List[str]
        Metrics to include in summary
        
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with formatted confidence intervals
    """
    rows = []
    
    for dataset, models in results_dict.items():
        for model_name, geos in models.items():
            # Average across geos for summary
            avg_stats = defaultdict(list)
            
            for geo, metrics in geos.items():
                for metric_name in metric_names:
                    if metric_name in metrics:
                        # Collect means across geos for averaging
                        avg_stats[metric_name].append(metrics[metric_name]['mean'])
            
            row = {
                'Dataset': dataset,
                'Model': model_name
            }
            
            for metric_name in metric_names:
                if metric_name in avg_stats and avg_stats[metric_name]:
                    # For the summary, show average of geo means
                    # Note: This is simplified - could do full posterior aggregation
                    geo_avg = np.mean([v for v in avg_stats[metric_name] if not np.isnan(v)])
                    
                    # Get the stats from first geo as representative (simplified)
                    first_geo = list(geos.keys())[0]
                    if metric_name in geos[first_geo]:
                        stats = geos[first_geo][metric_name]
                        # Use geo average for mean, but keep uncertainty from posterior
                        stats_with_avg = dict(stats)
                        stats_with_avg['mean'] = geo_avg
                        
                        precision = 2 if 'MAPE' in metric_name else 3
                        row[metric_name] = bayesian_metrics.format_metric_with_hdi(stats_with_avg, precision)
            
            rows.append(row)
    
    return pd.DataFrame(rows)