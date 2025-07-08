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
from .regions import generate_regional_baseline, generate_regional_channel_variations, generate_regional_transform_variations
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
        
        # Generate baseline data for each region first
        baseline_data = _generate_baseline_data(config, time_index)
        
        # Generate channel spend data (without geo in column names)
        spend_data = _generate_channel_spend_data(config, time_index)
        
        # Generate control variables
        control_data = _generate_control_variables(config, time_index)
        
        # Apply transformations to spend data
        transformed_data = _apply_transformations_to_data(
            config, spend_data, time_index
        )
        
        # Combine all data
        combined_data = pd.concat([spend_data, control_data], axis=1)
        # Calculate total sales
        combined_data["y"] = pd.concat([baseline_data, control_data, transformed_data], axis=1).sum(axis=1)
        combined_data = combined_data.reset_index()

        # Calculate ground truth
        ground_truth = calculate_ground_truth(
            config, spend_data, transformed_data, baseline_data
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


def _generate_baseline_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate baseline sales data for all regions."""
    baseline_data = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        baseline_sales = generate_regional_baseline(
            config.regions, time_index, region_idx, config.seed
        )
        
        # Create Series with MultiIndex (time_index, region_name)
        region_series = pd.Series(
            baseline_sales, 
            index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']),
            name="baseline_sales"
        )
        baseline_data.append(region_series)
    
    return pd.concat(baseline_data, axis=0)


def _generate_channel_spend_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate spend data for all channels across all regions."""
    # Create a list to store regional DataFrames
    regional_dataframes = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        regional_channels = generate_regional_channel_variations(
            config.regions, config.channels, region_idx, config.seed
        )
        
        region_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']))
        
        for i, channel in enumerate(regional_channels):
                # Generate independent spend pattern for this region
                # Use region-specific seed for independence
                region_seed = config.seed + region_idx * 1000 if config.seed is not None else None
                
                base_spend = generate_channel_spend(
                    channel, time_index, region_seed
                )
                
                # Use consistent x{i+1} format for channel columns
                column_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
                region_data[column_name] = base_spend
            
        
        regional_dataframes.append(region_data)
    
    # Concatenate all regional DataFrames on axis=0
    spend_data = pd.concat(regional_dataframes, axis=0)
    
    return spend_data



def _generate_control_variables(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Placeholder function for generating control variables data.
    
    Returns a DataFrame with multi-index (time_index, geo region) containing
    columns of zeros for each control variable.
    """
    # Create multi-index for all regions and time periods
    multi_index = pd.MultiIndex.from_product(
        [time_index, config.regions.region_names], 
        names=['date', 'geo']
    )
    
    # Create DataFrame with zeros for each control variable
    control_data = {}
    for idx, var_name in enumerate(config.control_variables.keys()):
        control_data[f"c{idx+1}"] = 0.0

    return pd.DataFrame(control_data, index=multi_index)


def _apply_transformations_to_data(
    config: MMMDataConfig,
    spend_data: pd.DataFrame,
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply adstock and saturation transformations to spend data."""
    # Create a list to store regional DataFrames
    regional_dataframes = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        # Get regional transform configuration
        regional_transform = generate_regional_transform_variations(
            config.regions,
            config.transforms,
            region_idx,
            config.seed
        )
        
        # Get regional channels configuration
        regional_channels = generate_regional_channel_variations(
            config.regions, config.channels, region_idx, config.seed
        )
        
        # Get spend data for this region
        region_mask = spend_data.index.get_level_values('geo') == region_name
        region_spend_data = spend_data[region_mask].copy()
        
        # Create DataFrame for this region with same structure
        region_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']))
        
        # Apply transformations to each channel
        for i, channel in enumerate(regional_channels):
            # Find the corresponding column in the spend data
            # Use consistent x{i+1} format for channel columns
            column_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
            
            if column_name in region_spend_data.columns:
                # Apply transformations with channel index and regional transform config
                max_spend = region_spend_data[column_name].max()
                transformed_spend = apply_transformations(
                    region_spend_data[column_name].values / max_spend,
                    regional_transform,
                    channel,
                    channel_idx=i
                )
                
                # Store in the region DataFrame with consistent column naming
                region_data["contribution_" + column_name] = transformed_spend * config.regions.base_sales_rate
            else:
                # If column not found, use zeros
                region_data[f"contribution_{column_name}"] = 0.0
        
        regional_dataframes.append(region_data)
    
    # Concatenate all regional DataFrames on axis=0
    transformed_data = pd.concat(regional_dataframes, axis=0)
    
    return transformed_data