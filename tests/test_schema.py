#!/usr/bin/env python3
"""
Comprehensive test script to verify the MMM output schema implementation.
Combines validation, compliance checks, and detailed reporting.
"""

import pandas as pd
from typing import Dict, Any

from mmm_param_recovery.data_generator import generate_mmm_dataset
from mmm_param_recovery.data_generator.config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


def test_mmm_schema():
    """Test the MMM output schema implementation with comprehensive validation."""
    print("Testing MMM output schema implementation...")
    
    # Create a configuration with control variables
    config = MMMDataConfig(
        n_periods=50,  # Minimum allowed for validation
        channels=[
            ChannelConfig(name='tv', base_spend=1000.0),
            ChannelConfig(name='digital', base_spend=500.0)
        ],
        regions=RegionConfig(n_regions=2, region_names=['North', 'South']),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 1000.0}
        ),
        control_variables={
            'price': {'base_value': 10.0, 'volatility': 0.1},
            'promotion': {'base_value': 0.2, 'volatility': 0.05}
        },
        seed=123
    )
    
    # Generate dataset
    result: Dict[str, Any] = generate_mmm_dataset(config)
    data: pd.DataFrame = result['data']
    
    print(f"Generated dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nSample data:")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isnull().sum())
    print(f"\nUnique date-geo combinations: {data[['date', 'geo']].drop_duplicates().shape[0]} out of {len(data)}")
    
    # Check for required schema compliance
    print("\n=== Schema Compliance Check ===")
    
    # Required columns
    required_columns = ['date', 'geo', 'y']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        print(f"❌ Missing required columns: {missing_required}")
        raise AssertionError(f"Missing required columns: {missing_required}")
    else:
        print("✅ All required columns present")
    
    # Channel columns (x1, x2, ...)
    expected_channel_columns = [f'x{i+1}' for i in range(len(config.channels))]
    actual_channel_columns = [col for col in data.columns if col.startswith('x') and col[1:].isdigit()]
    missing_channel_columns = [col for col in expected_channel_columns if col not in actual_channel_columns]
    if missing_channel_columns:
        print(f"❌ Missing channel columns: {missing_channel_columns}")
        raise AssertionError(f"Missing channel columns: {missing_channel_columns}")
    else:
        print("✅ All expected channel columns present")
    
    # Control variable columns (c1, c2, ...)
    expected_control_columns = [f'c{i+1}' for i in range(len(config.control_variables))]
    actual_control_columns = [col for col in data.columns if col.startswith('c') and col[1:].isdigit()]
    missing_control_columns = [col for col in expected_control_columns if col not in actual_control_columns]
    if missing_control_columns:
        print(f"❌ Missing control variable columns: {missing_control_columns}")
        raise AssertionError(f"Missing control variable columns: {missing_control_columns}")
    else:
        print("✅ All expected control variable columns present")
    
    # Data types
    if 'date' in data.columns and data['date'].dtype == 'datetime64[ns]':
        print("✅ Date column has correct datetime type")
    else:
        print("❌ Date column should be datetime type")
        raise AssertionError("Date column should be datetime type")
    
    if 'geo' in data.columns and data['geo'].dtype == 'object':
        print("✅ Geo column has correct string type")
    else:
        print("❌ Geo column should be string type")
        raise AssertionError("Geo column should be string type")
    
    # Unique date-geo combinations
    duplicates = data.duplicated(subset=['date', 'geo'])
    if not duplicates.any():
        print("✅ No duplicate date-geo combinations")
    else:
        print(f"❌ Found {duplicates.sum()} duplicate date-geo combinations")
        raise AssertionError(f"Found {duplicates.sum()} duplicate date-geo combinations")
    
    # No missing values
    missing_count = data.isnull().sum().sum()
    if missing_count == 0:
        print("✅ No missing values")
    else:
        print(f"❌ Found {missing_count} missing values")
        raise AssertionError(f"Found {missing_count} missing values")
    
    # No negative values in non-control columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_control_numeric = [col for col in numeric_columns if not col.startswith('c')]
    negative_count = (data[non_control_numeric] < 0).sum().sum()
    if negative_count == 0:
        print("✅ No negative values in non-control columns")
    else:
        print(f"❌ Found {negative_count} negative values in non-control columns")
        raise AssertionError(f"Found {negative_count} negative values in non-control columns")
    
    # Check expected number of rows
    expected_rows = config.n_periods * config.regions.n_regions
    if len(data) == expected_rows:
        print(f"✅ Expected {expected_rows} rows, got {len(data)}")
    else:
        print(f"❌ Expected {expected_rows} rows, got {len(data)}")
        raise AssertionError(f"Expected {expected_rows} rows, got {len(data)}")
    
    # Check geo values
    if config.regions.region_names is not None:
        expected_geos = set(config.regions.region_names)
        actual_geos = set(data['geo'].unique())
        if actual_geos == expected_geos:
            print(f"✅ Expected geos {expected_geos}, got {actual_geos}")
        else:
            print(f"❌ Expected geos {expected_geos}, got {actual_geos}")
            raise AssertionError(f"Expected geos {expected_geos}, got {actual_geos}")
    else:
        print("⚠️  No region names specified in config")
    
    print("\n=== Final Summary ===")
    print("✅ All schema tests passed!")
    print(f"Dataset has {len(data)} rows with {len(data.columns)} columns")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Geos: {sorted(data['geo'].unique())}")
    print(f"Channels: {len(config.channels)}")
    print(f"Control variables: {len(config.control_variables)}")


if __name__ == "__main__":
    test_mmm_schema()
