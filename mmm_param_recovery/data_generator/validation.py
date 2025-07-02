"""
Validation and error handling for MMM Dataset Generator.

This module provides comprehensive validation functions for configuration
parameters and data quality checks with helpful error messages and warnings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import warnings
from dataclasses import fields

from .config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


class MMMValidationError(Exception):
    """Custom exception for MMM validation errors."""
    pass


class MMMWarning(UserWarning):
    """Custom warning for MMM-related issues."""
    pass


def validate_config(config: MMMDataConfig) -> None:
    """
    Validate the complete MMM configuration.
    
    Parameters
    ----------
    config : MMMConfig
        Configuration to validate
        
    Raises
    ------
    MMMValidationError
        If configuration is invalid
    """
    errors = []
    warnings_list = []
    
    # Basic parameter validation
    errors.extend(_validate_basic_parameters(config))
    
    # Channel validation
    errors.extend(_validate_channels(config.channels))
    
    # Region validation
    errors.extend(_validate_regions(config.regions))
    
    # Transformation validation
    errors.extend(_validate_transforms(config.transforms))
    
    # Raise errors if any found
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise MMMValidationError(error_message)
    
    # Issue warnings if any found
    for warning in warnings_list:
        warnings.warn(warning, MMMWarning)


def _validate_basic_parameters(config: MMMDataConfig) -> List[str]:
    """Validate basic configuration parameters."""
    errors = []
    
    # Time parameters
    if config.n_periods < 1:
        errors.append("n_periods must be positive")
    
    # Start date validation
    if config.start_date is not None:
        try:
            pd.to_datetime(config.start_date)
        except ValueError:
            errors.append("start_date must be in YYYY-MM-DD format")
    
    # Seed validation
    if config.seed is not None:
        if not isinstance(config.seed, int):
            errors.append("seed must be an integer")
        elif config.seed < 0 or config.seed > 2**32 - 1:
            errors.append("seed must be between 0 and 2^32 - 1")
    
    return errors


def _validate_channels(channels: List[ChannelConfig]) -> List[str]:
    """Validate channel configurations."""
    errors = []
    
    if not channels:
        errors.append("At least one channel must be specified")
        return errors
    
    # Check for duplicate channel names
    channel_names = [ch.name for ch in channels]
    if len(channel_names) != len(set(channel_names)):
        errors.append("Channel names must be unique")
    
    # Validate individual channels
    for i, channel in enumerate(channels):
        channel_errors = _validate_single_channel(channel, i)
        errors.extend(channel_errors)
    
    return errors


def _validate_single_channel(channel: ChannelConfig, index: int) -> List[str]:
    """Validate a single channel configuration."""
    errors = []
    
    # Name validation
    if not channel.name or not channel.name.strip():
        errors.append(f"Channel {index}: name cannot be empty")
    
    # Pattern-specific validation
    if channel.pattern == "seasonal":
        if channel.seasonal_amplitude < 0 or channel.seasonal_amplitude > 1:
            errors.append(f"Channel {channel.name}: seasonal_amplitude must be between 0 and 1")
    
    elif channel.pattern == "delayed_start":
        if channel.start_period is not None and channel.start_period < 0:
            errors.append(f"Channel {channel.name}: start_period must be non-negative")
        if channel.ramp_up_periods < 1:
            errors.append(f"Channel {channel.name}: ramp_up_periods must be at least 1")
    
    elif channel.pattern == "on_off":
        if channel.activation_probability < 0 or channel.activation_probability > 1:
            errors.append(f"Channel {channel.name}: activation_probability must be between 0 and 1")
        if channel.min_active_periods < 1:
            errors.append(f"Channel {channel.name}: min_active_periods must be at least 1")
        if channel.max_active_periods < channel.min_active_periods:
            errors.append(f"Channel {channel.name}: max_active_periods must be >= min_active_periods")
    
    elif channel.pattern == "custom":
        if channel.custom_pattern_func is None:
            errors.append(f"Channel {channel.name}: custom_pattern_func must be provided for custom pattern")
    
    # General parameter validation
    if channel.base_spend < 0:
        errors.append(f"Channel {channel.name}: base_spend must be non-negative")
    
    if channel.spend_volatility < 0:
        errors.append(f"Channel {channel.name}: spend_volatility must be non-negative")
    
    if channel.base_effectiveness < 0:
        errors.append(f"Channel {channel.name}: base_effectiveness must be non-negative")
    
    if channel.regional_effectiveness_variation < 0:
        errors.append(f"Channel {channel.name}: regional_effectiveness_variation must be non-negative")
    
    return errors


def _validate_regions(regions: RegionConfig) -> List[str]:
    """Validate region configuration."""
    errors = []
    
    if regions.n_regions < 1:
        errors.append("n_regions must be at least 1")
    
    if regions.base_sales_rate <= 0:
        errors.append("base_sales_rate must be positive")
    
    if regions.sales_volatility < 0:
        errors.append("sales_volatility must be non-negative")
    
    if regions.seasonal_amplitude < 0 or regions.seasonal_amplitude > 1:
        errors.append("seasonal_amplitude must be between 0 and 1")
    
    if regions.regional_sales_variation < 0:
        errors.append("regional_sales_variation must be non-negative")
    
    if regions.regional_trend_variation < 0:
        errors.append("regional_trend_variation must be non-negative")
    
    if regions.region_correlation < -1 or regions.region_correlation > 1:
        errors.append("region_correlation must be between -1 and 1")
    
    if regions.channel_effectiveness_correlation < -1 or regions.channel_effectiveness_correlation > 1:
        errors.append("channel_effectiveness_correlation must be between -1 and 1")
    
    # Region names validation
    if regions.region_names is not None:
        if len(regions.region_names) != regions.n_regions:
            errors.append("Number of region_names must match n_regions")
        else:
            # Check for duplicate names
            if len(regions.region_names) != len(set(regions.region_names)):
                errors.append("Region names must be unique")
    
    return errors


def _validate_transforms(transforms: TransformConfig) -> List[str]:
    """Validate transformation configuration."""
    errors = []
    
    # Adstock validation
    if transforms.adstock_alpha < 0 or transforms.adstock_alpha > 1:
        errors.append("adstock_alpha must be between 0 and 1")
    
    if transforms.adstock_lam <= 0:
        errors.append("adstock_lam must be positive")
    
    if transforms.adstock_k <= 0:
        errors.append("adstock_k must be positive")
    
    # Saturation validation
    if transforms.saturation_ec50 <= 0:
        errors.append("saturation_ec50 must be positive")
    
    if transforms.saturation_slope <= 0:
        errors.append("saturation_slope must be positive")
    
    if transforms.saturation_max <= 0:
        errors.append("saturation_max must be positive")
    
    # Custom function validation
    if transforms.adstock_type == "custom" and transforms.custom_adstock_func is None:
        errors.append("custom_adstock_func must be provided for custom adstock type")
    
    if transforms.saturation_type == "custom" and transforms.custom_saturation_func is None:
        errors.append("custom_saturation_func must be provided for custom saturation type")
    
    return errors


def check_data_quality(data: pd.DataFrame, config: MMMDataConfig) -> Dict[str, Any]:
    """
    Check the quality of generated data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Generated dataset
    config : MMMConfig
        Configuration used for generation
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with quality metrics and issues found
    """
    quality_report = {
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        quality_report['issues'].append(f"Found {missing_counts.sum()} missing values")
        quality_report['metrics']['missing_values'] = missing_counts.to_dict()
    
    # Check for negative values in spend columns
    spend_columns = [col for col in data.columns if col.endswith('_spend')]
    for col in spend_columns:
        negative_count = (data[col] < 0).sum()
        if negative_count > 0:
            quality_report['issues'].append(f"Found {negative_count} negative values in {col}")
    
    # Check for negative values in sales columns
    sales_columns = [col for col in data.columns if col.endswith('_sales')]
    for col in sales_columns:
        negative_count = (data[col] < 0).sum()
        if negative_count > 0:
            quality_report['issues'].append(f"Found {negative_count} negative values in {col}")
    
    # Check for zero variance in important columns
    for col in spend_columns + sales_columns:
        if data[col].var() == 0:
            quality_report['warnings'].append(f"Zero variance in {col}")
    
    # Check for extreme values
    for col in spend_columns + sales_columns:
        q99 = data[col].quantile(0.99)
        q01 = data[col].quantile(0.01)
        if q99 / q01 > 100:
            quality_report['warnings'].append(f"High variance in {col} (99th percentile / 1st percentile > 100)")
    
    # Calculate basic statistics
    quality_report['metrics']['summary_stats'] = {}
    for col in spend_columns + sales_columns:
        quality_report['metrics']['summary_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'cv': data[col].std() / data[col].mean() if data[col].mean() > 0 else 0
        }
    
    return quality_report


def validate_ground_truth(ground_truth: Dict[str, Any], config: MMMDataConfig) -> List[str]:
    """
    Validate ground truth parameters for consistency.
    
    Parameters
    ----------
    ground_truth : Dict[str, Any]
        Ground truth parameters
    config : MMMConfig
        Configuration used for generation
        
    Returns
    -------
    List[str]
        List of validation issues found
    """
    issues = []
    
    # Check that all expected keys are present
    expected_keys = ['parameters', 'contributions', 'roas', 'attribution']
    for key in expected_keys:
        if key not in ground_truth:
            issues.append(f"Missing ground truth key: {key}")
    
    # Check parameter consistency
    if 'parameters' in ground_truth:
        params = ground_truth['parameters']
        
        # Check that all channels have parameters
        for channel in config.channels:
            if channel.name not in params:
                issues.append(f"Missing parameters for channel: {channel.name}")
        
        # Check parameter ranges
        for channel_name, channel_params in params.items():
            if 'effectiveness' in channel_params:
                if channel_params['effectiveness'] < 0:
                    issues.append(f"Negative effectiveness for channel: {channel_name}")
    
    return issues


def suggest_fixes(config: MMMDataConfig, issues: List[str]) -> Dict[str, str]:
    """
    Suggest fixes for common configuration issues.
    
    Parameters
    ----------
    config : MMMConfig
        Current configuration
    issues : List[str]
        List of issues found
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping issues to suggested fixes
    """
    suggestions = {}
    
    for issue in issues:
        if "n_periods must be positive" in issue:
            suggestions[issue] = "Increase n_periods to a positive value"
        
        elif "base_spend must be non-negative" in issue:
            suggestions[issue] = "Set base_spend to a positive value"
        
        elif "adstock_alpha must be between 0 and 1" in issue:
            suggestions[issue] = "Set adstock_alpha to a value between 0 and 1"
        
        elif "seed must be between 0 and 2^32 - 1" in issue:
            suggestions[issue] = "Use a seed value between 0 and 4294967295"
        
        elif "n_regions must be at least 1" in issue:
            suggestions[issue] = "Set n_regions to a positive value"
        
        elif "base_sales_rate must be positive" in issue:
            suggestions[issue] = "Set base_sales_rate to a positive value"
        
        elif "sales_volatility must be non-negative" in issue:
            suggestions[issue] = "Set sales_volatility to a non-negative value"
    
    return suggestions


def validate_output_schema(data: pd.DataFrame, config: MMMDataConfig) -> List[str]:
    """
    Validate that the output dataframe follows the required schema.
    
    Parameters
    ----------
    data : pd.DataFrame
        The generated dataset to validate
    config : MMMDataConfig
        The configuration used for generation
        
    Returns
    -------
    List[str]
        List of validation errors found
    """
    errors = []
    
    # Check required columns
    required_columns = ['date', 'geo', 'y']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check channel columns (x1, x2, x3, ...)
    expected_channel_columns = [f'x{i+1}' for i in range(len(config.channels))]
    actual_channel_columns = [col for col in data.columns if col.startswith('x') and col[1:].isdigit()]
    missing_channel_columns = [col for col in expected_channel_columns if col not in actual_channel_columns]
    if missing_channel_columns:
        errors.append(f"Missing channel columns: {missing_channel_columns}")
    
    # Check control variable columns (c1, c2, c3, ...)
    expected_control_columns = [f'c{i+1}' for i in range(len(config.control_variables))]
    actual_control_columns = [col for col in data.columns if col.startswith('c') and col[1:].isdigit()]
    missing_control_columns = [col for col in expected_control_columns if col not in actual_control_columns]
    if missing_control_columns:
        errors.append(f"Missing control variable columns: {missing_control_columns}")
    
    # Check data types
    if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
        errors.append("'date' column must be datetime type")
    
    if 'geo' in data.columns and not pd.api.types.is_string_dtype(data['geo']):
        errors.append("'geo' column must be string type")
    
    # Check for unique date-geo combinations
    if 'date' in data.columns and 'geo' in data.columns:
        duplicates = data.duplicated(subset=['date', 'geo'])
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate date-geo combinations")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    if not columns_with_missing.empty:
        errors.append(f"Found missing values in columns: {list(columns_with_missing.index)}")
    
    # Check for negative values in numeric columns (except control variables which can be negative)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Exclude control variables from negative value check as they can be negative
    non_control_numeric = [col for col in numeric_columns if not col.startswith('c')]
    if non_control_numeric:
        negative_values = (data[non_control_numeric] < 0).sum()
        columns_with_negative = negative_values[negative_values > 0]
        if not columns_with_negative.empty:
            errors.append(f"Found negative values in columns: {list(columns_with_negative.index)}")
    
    # Check for infinite values
    infinite_values = np.isinf(data.select_dtypes(include=[np.number])).sum()
    columns_with_infinite = infinite_values[infinite_values > 0]
    if not columns_with_infinite.empty:
        errors.append(f"Found infinite values in columns: {list(columns_with_infinite.index)}")
    
    # Check for zero variance in channel columns
    channel_columns = [col for col in data.columns if col.startswith('x') and col[1:].isdigit()]
    for col in channel_columns:
        if data[col].var() == 0:
            errors.append(f"Zero variance in channel column {col}")
    
    # Check that sales column (y) is positive
    if 'y' in data.columns:
        negative_sales = (data['y'] < 0).sum()
        if negative_sales > 0:
            errors.append(f"Found {negative_sales} negative sales values")
    
    # Check that all geo values are in the expected region names
    if 'geo' in data.columns and config.regions.region_names:
        unexpected_geos = set(data['geo'].unique()) - set(config.regions.region_names)
        if unexpected_geos:
            errors.append(f"Found unexpected geo values: {list(unexpected_geos)}")
    
    return errors 