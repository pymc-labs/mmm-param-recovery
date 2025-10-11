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


def comprehensive_data_validation(
    data: pd.DataFrame, 
    ground_truth: Dict[str, Any], 
    config: MMMDataConfig,
    raise_on_error: bool = False
) -> Dict[str, Any]:
    """
    Perform comprehensive data validation with warning systems.
    
    This function integrates all validation checks and provides a comprehensive
    report with warnings and suggestions for improvement.
    
    Parameters
    ----------
    data : pd.DataFrame
        Generated dataset to validate
    ground_truth : Dict[str, Any]
        Ground truth data to validate
    config : MMMDataConfig
        Configuration used for generation
    raise_on_error : bool, default False
        Whether to raise exceptions on validation errors (False = return warnings)
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive validation report with:
        - 'errors': List of critical errors found
        - 'warnings': List of warnings that don't prevent usage
        - 'suggestions': Dictionary of issues to suggested fixes
        - 'quality_metrics': Data quality metrics
        - 'overall_score': Overall data quality score (0-100)
        - 'passed': Boolean indicating if all critical checks passed
    """
    validation_report = {
        'errors': [],
        'warnings': [],
        'suggestions': {},
        'quality_metrics': {},
        'overall_score': 0,
        'passed': True
    }
    
    # 1. Schema validation
    schema_errors = validate_output_schema(data, config)
    validation_report['errors'].extend(schema_errors)
    
    # 2. Ground truth validation
    ground_truth_errors = validate_ground_truth(ground_truth, config)
    validation_report['errors'].extend(ground_truth_errors)
    
    # 3. Data quality checks
    quality_report = check_data_quality(data, config)
    validation_report['warnings'].extend(quality_report['warnings'])
    validation_report['quality_metrics'] = quality_report['metrics']
    
    # 4. Additional comprehensive checks
    additional_checks = _perform_additional_validation_checks(data, ground_truth, config)
    validation_report['errors'].extend(additional_checks['errors'])
    validation_report['warnings'].extend(additional_checks['warnings'])
    
    # 5. Calculate overall score
    validation_report['overall_score'] = _calculate_validation_score(
        data, validation_report['errors'], validation_report['warnings']
    )
    
    # 6. Generate suggestions
    all_issues = validation_report['errors'] + validation_report['warnings']
    validation_report['suggestions'] = suggest_fixes(config, all_issues)
    
    # 7. Determine if validation passed
    validation_report['passed'] = len(validation_report['errors']) == 0
    
    # 8. Issue warnings (if not raising exceptions)
    if not raise_on_error:
        for warning in validation_report['warnings']:
            warnings.warn(warning, MMMWarning)
    
    # 9. Raise exception if requested and errors found
    if raise_on_error and validation_report['errors']:
        error_message = "Data validation failed:\n" + "\n".join(f"- {error}" for error in validation_report['errors'])
        raise MMMValidationError(error_message)
    
    return validation_report


def _perform_additional_validation_checks(
    data: pd.DataFrame, 
    ground_truth: Dict[str, Any], 
    config: MMMDataConfig
) -> Dict[str, List[str]]:
    """Perform additional validation checks beyond basic schema and quality checks."""
    errors = []
    warnings = []
    
    # Check for data consistency between data and ground truth
    if 'roas_values' in ground_truth:
        roas_values = ground_truth['roas_values']
        
        # Check that ROAS values are reasonable (not too high or too low)
        for region_name, region_roas in roas_values.items():
            for channel_name, roas_value in region_roas.items():
                if roas_value > 10:
                    warnings.append(f"High ROAS value {roas_value:.2f} for {channel_name} in {region_name} (may indicate unrealistic effectiveness)")
                elif roas_value < 0.1:
                    warnings.append(f"Low ROAS value {roas_value:.2f} for {channel_name} in {region_name} (may indicate poor effectiveness)")
    
    # Check for temporal consistency
    if 'date' in data.columns:
        data_sorted = data.sort_values('date')
        date_gaps = data_sorted['date'].diff().dt.days
        if date_gaps.max() > 7:
            warnings.append(f"Large time gap detected: {date_gaps.max()} days between consecutive observations")
    
    # Check for regional consistency
    if 'geo' in data.columns and len(data['geo'].unique()) > 1:
        # Check if regions have similar sales patterns
        sales_by_region = data.groupby('geo')['y']
        sales_means = sales_by_region.mean()
        sales_cvs = sales_by_region.std() / sales_by_region.mean()
        
        if sales_cvs.max() / sales_cvs.min() > 5:
            warnings.append("High variation in sales volatility across regions")
        
        if sales_means.max() / sales_means.min() > 10:
            warnings.append("High variation in average sales across regions")
    
    # Check for channel effectiveness consistency
    channel_cols = [col for col in data.columns if col.startswith('x') and col[1:].split('_')[0].isdigit()]
    if channel_cols:
        # Check for channels with very low or zero spend
        for col in channel_cols:
            zero_spend_periods = (data[col] == 0).sum()
            total_periods = len(data)
            if zero_spend_periods / total_periods > 0.8:
                warnings.append(f"Channel {col} has very low activity ({zero_spend_periods}/{total_periods} periods with zero spend)")
    
    # Check for seasonality patterns
    if 'date' in data.columns and len(data) > 52:  # At least a year of data
        data_sorted = data.sort_values('date')
        # Simple seasonality check using autocorrelation
        sales_series = data_sorted['y'].values
        if len(sales_series) > 12:
            # Calculate autocorrelation at lag 12 (monthly if weekly data)
            autocorr = np.corrcoef(sales_series[:-12], sales_series[12:])[0, 1]
            if abs(autocorr) < 0.1:
                warnings.append("Low seasonality detected in sales data")
    
    return {'errors': errors, 'warnings': warnings}


def _calculate_validation_score(
    data: pd.DataFrame, 
    errors: List[str], 
    warnings: List[str]
) -> int:
    """Calculate overall validation score (0-100)."""
    base_score = 100
    
    # Deduct points for errors (critical issues)
    error_penalty = len(errors) * 20
    base_score -= error_penalty
    
    # Deduct points for warnings (minor issues)
    warning_penalty = len(warnings) * 5
    base_score -= warning_penalty
    
    # Bonus points for good data characteristics
    if len(data) > 100:
        base_score += 5  # Bonus for large dataset
    
    if data.isnull().sum().sum() == 0:
        base_score += 5  # Bonus for no missing values
    
    if (data['y'] >= 0).all():
        base_score += 5  # Bonus for no negative sales
    
    # Ensure score is within bounds
    return max(0, min(100, base_score))


def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Parameters
    ----------
    report : Dict[str, Any]
        Validation report from comprehensive_data_validation
    """
    print("=" * 60)
    print("MMM DATASET VALIDATION REPORT")
    print("=" * 60)
    
    # Overall status
    status = "✅ PASSED" if report['passed'] else "❌ FAILED"
    print(f"Overall Status: {status}")
    print(f"Quality Score: {report['overall_score']}/100")
    print()
    
    # Errors
    if report['errors']:
        print("🚨 CRITICAL ERRORS:")
        for error in report['errors']:
            print(f"  • {error}")
        print()
    
    # Warnings
    if report['warnings']:
        print("⚠️  WARNINGS:")
        for warning in report['warnings']:
            print(f"  • {warning}")
        print()
    
    # Suggestions
    if report['suggestions']:
        print("💡 SUGGESTIONS:")
        for issue, suggestion in report['suggestions'].items():
            print(f"  • {issue}")
            print(f"    → {suggestion}")
        print()
    
    # Quality metrics summary
    if report['quality_metrics']:
        print("📊 QUALITY METRICS:")
        for metric_name, metric_value in report['quality_metrics'].items():
            if isinstance(metric_value, dict):
                print(f"  {metric_name}:")
                for sub_metric, value in metric_value.items():
                    print(f"    {sub_metric}: {value}")
            else:
                print(f"  {metric_name}: {metric_value}")
        print()
    
    print("=" * 60) 