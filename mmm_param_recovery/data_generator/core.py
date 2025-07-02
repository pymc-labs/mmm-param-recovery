"""
Core data generation functions for MMM Dataset Generator.

This module contains the main data generation function and core logic for
creating synthetic MMM datasets with known ground truth parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig
from .channels import generate_channel_spend
from .regions import generate_regional_baseline
from .transforms import apply_transformations
from .ground_truth import calculate_ground_truth
from .validation import validate_config, validate_output_schema


def generate_mmm_dataset(
    config: Optional[MMMDataConfig] = None,
    **kwargs
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Generate a synthetic MMM dataset with known ground truth parameters.
    
    This is the main entry point for the MMM Dataset Generator. It creates
    a comprehensive dataset including channel spend data, regional sales data,
    transformed variables, and ground truth parameters for validation.
    
    Parameters
    ----------
    config : MMMDataConfig, optional
        Configuration object specifying all data generation parameters.
        If None, uses DEFAULT_CONFIG.
    **kwargs
        Additional parameters to override in the configuration.
        
    Returns
    -------
    Dict[str, Union[pd.DataFrame, Dict]]
        Dictionary containing:
        - 'data': Main dataset with spend, sales, and transformed variables
        - 'ground_truth': Dictionary with true parameters and contributions
        - 'config': The configuration used for generation
        
    Raises
    ------
    ValueError
        If configuration parameters are invalid or inconsistent.
    RuntimeError
        If data generation fails due to numerical issues.
        
    Examples
    --------
    >>> from mmm_data_generator import generate_mmm_dataset, DEFAULT_CONFIG
    >>> 
    >>> # Use default configuration
    >>> result = generate_mmm_dataset()
    >>> data = result['data']
    >>> ground_truth = result['ground_truth']
    >>> 
    >>> # Custom configuration
    >>> config = MMMDataConfig(
    ...     n_periods=52,
    ...     channels=[ChannelConfig(name="x_seasonal", base_spend=2000.0)],
    ...     seed=123
    ... )
    >>> result = generate_mmm_dataset(config)
    """
    
    # Set up configuration
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown parameter '{key}' ignored")
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
    
    try:
        # Generate time index
        time_index = _generate_time_index(config)
        
        # Generate channel spend data
        spend_data = _generate_channel_spend_data(config, time_index)
        
        # Generate regional baseline sales
        baseline_data = _generate_baseline_data(config, time_index)
        
        # Generate control variables
        control_data = _generate_control_variables(config, time_index)
        
        # Apply transformations to spend data
        transformed_data = _apply_transformations_to_data(
            config, spend_data, time_index
        )
        
        # Calculate channel contributions and sales
        contributions_data = _calculate_channel_contributions(
            config, transformed_data, baseline_data
        )
        
        # Combine all data
        combined_data = _combine_data(
            config, time_index, spend_data, baseline_data, 
            control_data, transformed_data, contributions_data
        )
        
        # Calculate ground truth
        ground_truth = calculate_ground_truth(
            config, spend_data, transformed_data, contributions_data
        )
        
        # Validate output schema
        schema_errors = validate_output_schema(combined_data, config)
        if schema_errors:
            error_message = "Output schema validation failed:\n" + "\n".join(f"- {error}" for error in schema_errors)
            raise ValueError(error_message)
        
        # Prepare output
        result = {
            'data': combined_data,
            'ground_truth': ground_truth,
            'config': config
        }
        
        # Add optional outputs
        if config.include_raw_data:
            result['raw_spend'] = spend_data
            result['raw_baseline'] = baseline_data
        
        if config.include_transformed_data:
            result['transformed_spend'] = transformed_data
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Data generation failed: {e}")


def _generate_time_index(config: MMMDataConfig) -> pd.DatetimeIndex:
    """Generate time index for the dataset."""
    if config.start_date:
        start_date = pd.to_datetime(config.start_date)
    else:
        start_date = pd.Timestamp('2020-01-01')
    
    # Generate weekly periods
    time_index = pd.date_range(
        start=start_date,
        periods=config.n_periods,
        freq='W'
    )
    
    return time_index


def _generate_channel_spend_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate spend data for all channels across all regions."""
    spend_data = {}
    
    for channel in config.channels:
        for region_idx, region_name in enumerate(config.regions.region_names):
            # Generate base spend pattern
            base_spend = generate_channel_spend(
                channel, time_index, config.seed
            )
            
            # Add regional variation if multiple regions
            if config.regions.n_regions > 1:
                regional_factor = _generate_regional_channel_factor(
                    config, channel, region_idx
                )
                base_spend = base_spend * regional_factor
            
            # Add spend noise
            spend_noise = np.random.normal(
                0, channel.spend_volatility * base_spend
            )
            final_spend = np.maximum(0, base_spend + spend_noise)
            
            column_name = f"{channel.name}_{region_name}_spend"
            spend_data[column_name] = final_spend
    
    return pd.DataFrame(spend_data, index=time_index)


def _generate_regional_channel_factor(
    config: MMMDataConfig, 
    channel: ChannelConfig, 
    region_idx: int
) -> np.ndarray:
    """Generate regional variation factor for channel effectiveness."""
    # Use correlated random numbers for regional similarity
    if region_idx == 0:
        base_factor = np.random.normal(
            1.0, channel.regional_effectiveness_variation
        )
    else:
        # Correlate with previous regions
        correlation = config.regions.channel_effectiveness_correlation
        base_factor = (
            correlation * np.random.normal(1.0, 0.1) +
            (1 - correlation) * np.random.normal(
                1.0, channel.regional_effectiveness_variation
            )
        )
    
    # Ensure factor is positive
    return np.maximum(0.1, base_factor)


def _generate_baseline_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate baseline sales data for all regions."""
    baseline_data = {}
    
    for region_idx, region_name in enumerate(config.regions.region_names):
        baseline_sales = generate_regional_baseline(
            config.regions, time_index, region_idx, config.seed
        )
        
        column_name = f"{region_name}_baseline_sales"
        baseline_data[column_name] = baseline_sales
    
    return pd.DataFrame(baseline_data, index=time_index)


def _generate_control_variables(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate control variables data."""
    control_data = {}
    
    for var_name, var_config in config.control_variables.items():
        base_value = var_config['base_value']
        volatility = var_config['volatility']
        
        # Generate time series with trend and noise
        trend = var_config.get('trend', 0.0)
        seasonal_amplitude = var_config.get('seasonal_amplitude', 0.0)
        
        # Base trend
        time_factor = np.arange(len(time_index))
        trend_component = base_value * (1 + trend * time_factor / len(time_index))
        
        # Seasonal component
        seasonal_component = 0
        if seasonal_amplitude > 0:
            seasonal_component = (
                seasonal_amplitude * base_value * 
                np.sin(2 * np.pi * time_factor / 52)  # Annual seasonality
            )
        
        # Noise component
        noise = np.random.normal(0, volatility * base_value, len(time_index))
        
        # Combine components
        control_values = trend_component + seasonal_component + noise
        control_data[var_name] = np.maximum(0, control_values)
    
    return pd.DataFrame(control_data, index=time_index)


def _apply_transformations_to_data(
    config: MMMDataConfig,
    spend_data: pd.DataFrame,
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Apply adstock and saturation transformations to spend data."""
    transformed_data = {}
    
    for column in spend_data.columns:
        channel_name = column.split('_')[0]  # Extract channel name from column
        
        # Find channel config
        channel_config = next(
            (ch for ch in config.channels if ch.name == channel_name),
            None
        )
        
        if channel_config is None:
            warnings.warn(f"No config found for channel {channel_name}")
            transformed_data[column.replace('_spend', '_transformed')] = spend_data[column]
            continue
        
        # Apply transformations
        transformed_spend = apply_transformations(
            spend_data[column].values,
            config.transforms,
            channel_config
        )
        
        transformed_column = column.replace('_spend', '_transformed')
        transformed_data[transformed_column] = transformed_spend
    
    return pd.DataFrame(transformed_data, index=time_index)


def _calculate_channel_contributions(
    config: MMMDataConfig,
    transformed_data: pd.DataFrame,
    baseline_data: pd.DataFrame
) -> pd.DataFrame:
    """Calculate channel contributions to sales."""
    contributions_data = {}
    
    for region_name in config.regions.region_names:
        region_contributions = {}
        
        for channel in config.channels:
            # Get transformed spend for this channel and region
            transformed_column = f"{channel.name}_{region_name}_transformed"
            if transformed_column not in transformed_data.columns:
                continue
            
            transformed_spend = transformed_data[transformed_column]
            
            # Calculate effectiveness for this channel and region
            effectiveness = _calculate_channel_effectiveness(
                config, channel, region_name
            )
            
            # Calculate contribution
            contribution = transformed_spend * effectiveness
            region_contributions[f"{channel.name}_{region_name}_contribution"] = contribution
        
        # Add total contribution column
        if region_contributions:
            total_contribution = sum(region_contributions.values())
            region_contributions[f"{region_name}_total_contribution"] = total_contribution
        
        contributions_data.update(region_contributions)
    
    return pd.DataFrame(contributions_data, index=transformed_data.index)


def _calculate_channel_effectiveness(
    config: MMMDataConfig,
    channel: ChannelConfig,
    region_name: str
) -> float:
    """Calculate channel effectiveness for a specific region."""
    # Base effectiveness
    effectiveness = channel.base_effectiveness
    
    # Add trend over time
    if channel.effectiveness_trend != 0:
        # This would need to be calculated per time period
        # For now, use average trend
        effectiveness *= (1 + channel.effectiveness_trend / 2)
    
    # Add regional variation
    region_idx = config.regions.region_names.index(region_name)
    if config.regions.n_regions > 1:
        regional_factor = _generate_regional_channel_factor(
            config, channel, region_idx
        )
        effectiveness *= regional_factor
    
    return effectiveness


def _combine_data(
    config: MMMDataConfig,
    time_index: pd.DatetimeIndex,
    spend_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    control_data: pd.DataFrame,
    transformed_data: pd.DataFrame,
    contributions_data: pd.DataFrame
) -> pd.DataFrame:
    """Combine all data into final dataset with required schema."""
    # Create list to store all rows
    all_rows = []
    
    # Generate all date-geo combinations
    for date in time_index:
        for region_name in config.regions.region_names:
            row_data = {'date': date, 'geo': region_name}
            
            # Add channel spend data (x1, x2, x3, ...)
            for i, channel in enumerate(config.channels):
                spend_col = f"{channel.name}_{region_name}_spend"
                if spend_col in spend_data.columns:
                    # Use consistent x{i+1} format for channel columns
                    column_name = f'x{i+1}'
                    row_data[column_name] = spend_data.loc[date, spend_col]
                else:
                    column_name = f'x{i+1}'
                    row_data[column_name] = 0.0
            
            # Add control variables (c1, c2, c3, ...)
            for i, (var_name, var_config) in enumerate(config.control_variables.items()):
                if var_name in control_data.columns:
                    row_data[f'c{i+1}'] = control_data.loc[date, var_name]
                else:
                    row_data[f'c{i+1}'] = 0.0
            
            # Calculate total sales (y)
            baseline_col = f"{region_name}_baseline_sales"
            total_contribution_col = f"{region_name}_total_contribution"
            
            baseline_sales = 0.0
            if baseline_col in baseline_data.columns:
                baseline_sales = baseline_data.loc[date, baseline_col]
            
            total_contribution = 0.0
            if total_contribution_col in contributions_data.columns:
                total_contribution = contributions_data.loc[date, total_contribution_col]
            
            row_data['y'] = baseline_sales + total_contribution
            
            all_rows.append(row_data)
    
    # Create DataFrame and sort by date, then geo
    combined_data = pd.DataFrame(all_rows)
    combined_data = combined_data.sort_values(['date', 'geo']).reset_index(drop=True)
    
    return combined_data 