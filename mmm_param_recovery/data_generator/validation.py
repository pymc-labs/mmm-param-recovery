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
    
    # Region names validation
    if regions.region_names is not None:
        if len(regions.region_names) != regions.n_regions:
            errors.append("Number of region_names must match n_regions")
        else:
            # Check for duplicate names
            if len(regions.region_names) != len(set(regions.region_names)):
                errors.append("Region names must be unique")
    
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
    Validate ground truth parameters for consistency and completeness.
    
    Parameters
    ----------
    ground_truth : Dict[str, Any]
        Ground truth parameters
    config : MMMDataConfig
        Configuration used for generation
        
    Returns
    -------
    List[str]
        List of validation issues found
    """
    issues = []
    
    # Check that all expected keys are present
    expected_keys = ['transformation_parameters', 'baseline_components', 'roas_values', 'attribution_percentages', 'transformed_spend']
    for key in expected_keys:
        if key not in ground_truth:
            issues.append(f"Missing ground truth key: {key}")
    
    # Validate transformation parameters
    if 'transformation_parameters' in ground_truth:
        transform_params = ground_truth['transformation_parameters']
        issues.extend(_validate_transformation_parameters(transform_params, config))
    
    # Validate baseline components
    if 'baseline_components' in ground_truth:
        baseline_data = ground_truth['baseline_components']
        issues.extend(_validate_baseline_components(baseline_data, config))
    
    # Validate ROAS values
    if 'roas_values' in ground_truth:
        roas_values = ground_truth['roas_values']
        issues.extend(_validate_roas_values(roas_values, config))
    
    # Validate attribution percentages
    if 'attribution_percentages' in ground_truth:
        attribution_values = ground_truth['attribution_percentages']
        issues.extend(_validate_attribution_percentages(attribution_values, config))
    
    # Validate transformed spend data
    if 'transformed_spend' in ground_truth:
        transformed_data = ground_truth['transformed_spend']
        issues.extend(_validate_transformed_spend(transformed_data, config))
    
    return issues


def _validate_transformation_parameters(transform_params: Dict[str, Any], config: MMMDataConfig) -> List[str]:
    """Validate transformation parameters structure and values."""
    issues = []
    
    if 'channels' not in transform_params:
        issues.append("Missing 'channels' key in transformation_parameters")
        return issues
    
    channels_params = transform_params['channels']
    
    # Check that all channels have parameters
    for channel in config.channels:
        if channel.name not in channels_params:
            issues.append(f"Missing transformation parameters for channel: {channel.name}")
    
    # Check that all regions have parameters for each channel
    for channel_name, channel_regions in channels_params.items():
        for region_name in config.regions.region_names: # type: ignore
            if region_name not in channel_regions:
                issues.append(f"Missing transformation parameters for channel {channel_name} in region {region_name}")
    
    # Validate parameter values
    for channel_name, channel_regions in channels_params.items():
        for region_name, region_params in channel_regions.items():
            # Check for required transformation parameters
            required_params = ['adstock_rate', 'saturation_params']
            for param in required_params:
                if param not in region_params:
                    issues.append(f"Missing {param} for channel {channel_name} in region {region_name}")
            
            # Validate adstock rate range
            if 'adstock_rate' in region_params:
                adstock_rate = region_params['adstock_rate']
                if not (0 <= adstock_rate <= 1):
                    issues.append(f"Invalid adstock_rate {adstock_rate} for channel {channel_name} in region {region_name} (must be between 0 and 1)")
            
            # Validate saturation parameters
            if 'saturation_params' in region_params:
                sat_params = region_params['saturation_params']
                if 'half_max_effective' in sat_params:
                    half_max = sat_params['half_max_effective']
                    if half_max <= 0:
                        issues.append(f"Invalid half_max_effective {half_max} for channel {channel_name} in region {region_name} (must be positive)")
    
    return issues


def _validate_baseline_components(baseline_data: pd.DataFrame, config: MMMDataConfig) -> List[str]:
    """Validate baseline components structure and values."""
    issues = []
    
    # Check required columns
    required_columns = ['base_sales', 'trend', 'seasonal', 'baseline_sales']
    missing_columns = [col for col in required_columns if col not in baseline_data.columns]
    if missing_columns:
        issues.append(f"Missing baseline component columns: {missing_columns}")
    
    # Check that all regions are present
    if 'geo' in baseline_data.index.names:
        expected_regions = set(config.regions.region_names) # type: ignore
        actual_regions = set(baseline_data.index.get_level_values('geo').unique())
        missing_regions = expected_regions - actual_regions
        if missing_regions:
            issues.append(f"Missing baseline data for regions: {list(missing_regions)}")
    
    # Validate baseline sales values
    if 'baseline_sales' in baseline_data.columns:
        negative_sales = (baseline_data['baseline_sales'] < 0).sum()
        if negative_sales > 0:
            issues.append(f"Found {negative_sales} negative baseline sales values")
    
    # Check for infinite values
    numeric_columns = baseline_data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        infinite_values = np.isinf(baseline_data[numeric_columns]).sum().sum()
        if infinite_values > 0:
            issues.append(f"Found {infinite_values} infinite values in baseline components")
    
    return issues


def _validate_roas_values(roas_values: Dict[str, Dict[str, float]], config: MMMDataConfig) -> List[str]:
    """Validate ROAS values structure and consistency."""
    issues = []
    
    # Check that all regions are present
    expected_regions = set(config.regions.region_names) # type: ignore
    actual_regions = set(roas_values.keys())
    missing_regions = expected_regions - actual_regions
    if missing_regions:
        issues.append(f"Missing ROAS values for regions: {list(missing_regions)}")
    
    # Check that all channels are present for each region
    for region_name, region_roas in roas_values.items():
        expected_channels = set(channel.name for channel in config.channels)
        actual_channels = set(region_roas.keys())
        missing_channels = expected_channels - actual_channels
        if missing_channels:
            issues.append(f"Missing ROAS values for channels {list(missing_channels)} in region {region_name}")
    
    # Validate ROAS value ranges
    for region_name, region_roas in roas_values.items():
        for channel_name, roas_value in region_roas.items():
            if not isinstance(roas_value, (int, float)):
                issues.append(f"Invalid ROAS value type for channel {channel_name} in region {region_name}")
            elif roas_value < 0:
                issues.append(f"Negative ROAS value {roas_value} for channel {channel_name} in region {region_name}")
            elif np.isinf(roas_value):
                issues.append(f"Infinite ROAS value for channel {channel_name} in region {region_name}")
    
    return issues


def _validate_attribution_percentages(attribution_values: Dict[str, Dict[str, float]], config: MMMDataConfig) -> List[str]:
    """Validate attribution percentages structure and consistency."""
    issues = []
    
    # Check that all regions are present
    expected_regions = set(config.regions.region_names) # type: ignore
    actual_regions = set(attribution_values.keys())
    missing_regions = expected_regions - actual_regions
    if missing_regions:
        issues.append(f"Missing attribution percentages for regions: {list(missing_regions)}")
    
    # Check that all channels are present for each region
    for region_name, region_attribution in attribution_values.items():
        expected_channels = set(channel.name for channel in config.channels)
        actual_channels = set(region_attribution.keys())
        missing_channels = expected_channels - actual_channels
        if missing_channels:
            issues.append(f"Missing attribution percentages for channels {list(missing_channels)} in region {region_name}")
    
    # Validate attribution percentage ranges and sum
    for region_name, region_attribution in attribution_values.items():
        total_attribution = sum(region_attribution.values())
        
        # Check if attribution percentages sum to approximately 100% (allowing for small numerical errors)
        if not (99.5 <= total_attribution <= 100.5):
            issues.append(f"Attribution percentages for region {region_name} sum to {total_attribution:.2f}% (should be ~100%)")
        
        # Check individual percentage ranges
        for channel_name, attribution_pct in region_attribution.items():
            if not isinstance(attribution_pct, (int, float)):
                issues.append(f"Invalid attribution percentage type for channel {channel_name} in region {region_name}")
            elif attribution_pct < 0:
                issues.append(f"Negative attribution percentage {attribution_pct} for channel {channel_name} in region {region_name}")
            elif attribution_pct > 100:
                issues.append(f"Attribution percentage {attribution_pct} > 100% for channel {channel_name} in region {region_name}")
            elif np.isinf(attribution_pct):
                issues.append(f"Infinite attribution percentage for channel {channel_name} in region {region_name}")
    
    return issues


def _validate_transformed_spend(transformed_data: pd.DataFrame, config: MMMDataConfig) -> List[str]:
    """Validate transformed spend data structure and values."""
    issues = []
    
    # Check that all regions are present
    if 'geo' in transformed_data.index.names:
        expected_regions = set(config.regions.region_names) # type: ignore
        actual_regions = set(transformed_data.index.get_level_values('geo').unique())
        missing_regions = expected_regions - actual_regions
        if missing_regions:
            issues.append(f"Missing transformed spend data for regions: {list(missing_regions)}")
    
    # Check for contribution columns
    contribution_cols = [col for col in transformed_data.columns if col.startswith('contribution_')]
    if not contribution_cols:
        issues.append("No contribution columns found in transformed spend data")
    
    # Validate contribution values
    for col in contribution_cols:
        negative_contributions = (transformed_data[col] < 0).sum()
        if negative_contributions > 0:
            issues.append(f"Found {negative_contributions} negative contribution values in column {col}")
        
        infinite_contributions = np.isinf(transformed_data[col]).sum()
        if infinite_contributions > 0:
            issues.append(f"Found {infinite_contributions} infinite contribution values in column {col}")
    
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
        
        elif "adstock_kwargs alpha must be between 0 and 1" in issue:
            suggestions[issue] = "Set adstock_kwargs alpha to a value between 0 and 1"
        
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
    expected_channel_columns = [f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}' for i, channel in enumerate(config.channels)]
    actual_channel_columns = [col for col in data.columns if col.startswith('x')]
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
    
    # Check for negative values in numeric columns (except control variables and baseline components which can be negative)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Exclude control variables and only 'trend' and 'seasonal' from negative value check as they can be negative
    excluded_columns = ['trend', 'seasonal'] + [col for col in numeric_columns if col.startswith('c')]
    non_control_numeric = [col for col in numeric_columns if col not in excluded_columns]
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