#!/usr/bin/env python3
"""
Comprehensive test script to check the default configuration output schema and control variables.
"""

import pandas as pd
from typing import Dict, Any

from mmm_param_recovery.data_generator import generate_mmm_dataset, DEFAULT_CONFIG

def test_default_config():
    """Test the default configuration output schema and control variables."""
    print("Testing default configuration output schema and control variables...")
    
    # Generate dataset with default configuration
    result: Dict[str, Any] = generate_mmm_dataset()
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
    else:
        print("✅ All required columns present")
    
    # Channel columns (x1, x2, ...)
    expected_channel_columns = [f'x{i+1}' for i in range(len(DEFAULT_CONFIG.channels))]
    actual_channel_columns = [col for col in data.columns if col.startswith('x') and col[1:].isdigit()]
    missing_channel_columns = [col for col in expected_channel_columns if col not in actual_channel_columns]
    if missing_channel_columns:
        print(f"❌ Missing channel columns: {missing_channel_columns}")
    else:
        print("✅ All expected channel columns present")
    
    # Control variable columns (c1, c2, ...)
    expected_control_columns = [f'c{i+1}' for i in range(len(DEFAULT_CONFIG.control_variables))]
    actual_control_columns = [col for col in data.columns if col.startswith('c') and col[1:].isdigit()]
    missing_control_columns = [col for col in expected_control_columns if col not in actual_control_columns]
    if missing_control_columns:
        print(f"❌ Missing control variable columns: {missing_control_columns}")
    else:
        print("✅ All expected control variable columns present")
    
    # Data types
    if 'date' in data.columns and data['date'].dtype == 'datetime64[ns]':
        print("✅ Date column has correct datetime type")
    else:
        print("❌ Date column should be datetime type")
    
    if 'geo' in data.columns and data['geo'].dtype == 'object':
        print("✅ Geo column has correct string type")
    else:
        print("❌ Geo column should be string type")
    
    # Unique date-geo combinations
    duplicates = data.duplicated(subset=['date', 'geo'])
    if not duplicates.any():
        print("✅ No duplicate date-geo combinations")
    else:
        print(f"❌ Found {duplicates.sum()} duplicate date-geo combinations")
    
    # No missing values
    missing_count = data.isnull().sum().sum()
    if missing_count == 0:
        print("✅ No missing values")
    else:
        print(f"❌ Found {missing_count} missing values")
    
    # No negative values in non-control columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_control_numeric = [col for col in numeric_columns if not col.startswith('c')]
    negative_count = (data[non_control_numeric] < 0).sum().sum()
    if negative_count == 0:
        print("✅ No negative values in non-control columns")
    else:
        print(f"❌ Found {negative_count} negative values in non-control columns")
    
    # Check if default config has control variables
    print(f"\nDefault config control variables: {len(DEFAULT_CONFIG.control_variables)}")
    if len(DEFAULT_CONFIG.control_variables) == 0:
        print("⚠️  Default configuration has no control variables")
    else:
        print("✅ Default configuration has control variables")
    
    # Control variables verification
    print("\n=== Control Variables Verification ===")
    
    if len(DEFAULT_CONFIG.control_variables) > 0:
        # Check for specific expected control columns
        expected_control_columns = ['c1', 'c2']
        actual_control_columns = [col for col in data.columns if col.startswith('c') and col[1:].isdigit()]
        
        print(f"Expected control columns: {expected_control_columns}")
        print(f"Actual control columns: {actual_control_columns}")
        
        # Verify control variables are present
        for expected_col in expected_control_columns:
            if expected_col in actual_control_columns:
                print(f"✅ Control column {expected_col} is present")
            else:
                print(f"❌ Missing control column: {expected_col}")
        
        # Verify control variables have data
        for col in actual_control_columns:
            if data[col].notna().all():
                print(f"✅ Control column {col} has no missing values")
            else:
                print(f"❌ Control column {col} has missing values")
            
            if len(data[col].unique()) > 1:
                print(f"✅ Control column {col} has variation")
            else:
                print(f"❌ Control column {col} has no variation")
        
        print("✅ Control variables verification completed")
    else:
        print("⚠️  Skipping control variables verification - no control variables in default config")

if __name__ == "__main__":
    test_default_config() 