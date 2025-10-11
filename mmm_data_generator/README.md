# MMM Dataset Generator

A lightweight Python package for generating reproducible datasets for benchmarking Marketing Mix Models (MMM) with known ground truth parameters.

## Overview

The MMM Dataset Generator creates synthetic marketing data with configurable channel patterns, geographic regions, adstock/saturation transformations, control variables, and comprehensive ground truth information for model validation. It provides a complete toolkit for MMM research, testing, and benchmarking.

## Key Features

### ✅ **Core Data Generation**
- **Main API**: Single `generate_mmm_dataset()` function as the primary entry point
- **Configuration System**: Comprehensive dataclasses for all parameters (`MMMDataConfig`, `ChannelConfig`, `RegionConfig`, `TransformConfig`, `ControlConfig`)
- **Reproducibility**: Complete seed management and random number generation utilities
- **Data Validation**: Comprehensive parameter validation and data quality checks
- **Output Schema**: Strict compliance with required column structure (`date`, `geo`, `y`, `x1-*`, `x2-*`, ..., `c1-*`, `c2-*`, ...)

### ✅ **Channel Pattern Generation**
- **Linear Trend**: Configurable linear trend channels with customizable spend growth/decline
- **Seasonal**: Annual seasonality patterns with adjustable amplitude and phase
- **Delayed Start**: Channels with configurable start times and ramp-up periods
- **On/Off**: Random activation patterns with controllable probability and duration
- **Custom**: Support for user-defined channel patterns via function inputs

### ✅ **Control Variables**
- **Price Variables**: Configurable price patterns with trends and volatility
- **Promotion Variables**: Seasonal and on/off promotion patterns
- **External Factors**: Custom control variables with various patterns
- **Regional Variations**: Control variables can vary by geographic region

### ✅ **Adstock and Saturation Transformations**
- **PyMC Marketing Integration**: Wrapper for applying adstock and saturation transforms from PyMC Marketing
- **Flexible Combinations**: Support for different adstock/saturation combinations per channel
- **Parameter Tracking**: Complete tracking of transformation parameters for ground truth

### ✅ **Geographic Region Management**
- **Multi-Region Support**: Generate data for 1-50 geographic regions
- **Regional Variations**: Different baseline sales rates and channel effectiveness per region
- **Region Similarity Controls**: Configurable similarity levels between regions
- **Regional Parameter Validation**: Region-specific parameter validation

### ✅ **Ground Truth Calculation and ROAS**
- **True Parameter Tracking**: Complete tracking of betas, alphas, kappas, etc.
- **Channel Contribution Calculation**: True channel contributions over time
- **ROAS Computation**: True ROAS values for each channel and region
- **Attribution Percentages**: True attribution percentage calculations
- **Baseline Component Tracking**: Intercept, trend, seasonality, and control variable tracking

### ✅ **Configuration Presets**
- **Basic**: Simple configuration for learning and testing
- **Seasonal**: Strong seasonal patterns for seasonal business analysis
- **Multi-Region**: Multi-regional analysis with regional variations
- **Small Business**: Small business with limited budget and channels
- **Medium Business**: Medium business with more budget and channels
- **Large Business**: Large business with many channels and regions

### ✅ **Visualization and Validation Functions**
- **Channel Spend Visualization**: Plot channel spend patterns over time
- **Channel Contribution Plots**: Visualize channel contributions over time
- **ROAS Visualization**: ROAS comparison charts across channels
- **Regional Comparison**: Regional comparison visualization functions
- **Data Quality Diagnostics**: Comprehensive data validation plots

### ✅ **Data Quality and Validation**
- **Parameter Validation**: Comprehensive validation of all configuration parameters
- **Data Integrity**: Ensures unique date-geo combinations and no missing values
- **Quality Checks**: Non-negative values and reasonable range validation
- **Warning System**: Helpful warnings for potentially problematic configurations
- **Comprehensive Validation**: End-to-end data validation with detailed reporting

## Installation

```bash
pip install mmm-data-generator
```

### Dependencies

The package requires:
- Python >= 3.10
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pymc-marketing >= 0.16.0,<0.17

## Quick Start

### Basic Usage

```python
from mmm_data_generator import generate_mmm_dataset, DEFAULT_CONFIG

# Generate dataset with default configuration
result = generate_mmm_dataset()
data = result['data']
ground_truth = result['ground_truth']

print(f"Generated {len(data)} rows of data")
print(f"Columns: {list(data.columns)}")
print("Sample of the generated data:")
print(data.head())
```

### Using Presets

```python
from mmm_data_generator import get_preset_config, generate_mmm_dataset, list_available_presets

# See available presets
print("Available presets:", list_available_presets())

# Use a preset configuration
config = get_preset_config('seasonal')
result = generate_mmm_dataset(config)

# Customize a preset
from mmm_data_generator import customize_preset
config = customize_preset('basic', n_periods=104, seed=123)
result = generate_mmm_dataset(config)
```

### Custom Configuration

```python
from mmm_data_generator import MMMDataConfig, ChannelConfig, ControlConfig, generate_mmm_dataset

# Create custom configuration with channels and control variables
config = MMMDataConfig(
    n_periods=52,
    channels=[
        ChannelConfig(
            name="x-TV",
            pattern="seasonal",
            base_spend=3000.0,
            seasonal_amplitude=0.5,
            base_effectiveness=0.7
        ),
        ChannelConfig(
            name="x-Digital",
            pattern="linear_trend",
            base_spend=1500.0,
            spend_trend=0.08,
            base_effectiveness=0.6
        )
    ],
    control_variables=[
        ControlConfig(
            name="price",
            pattern="linear_trend",
            base_value=10.0,
            value_trend=0.05
        )
    ],
    seed=42
)

result = generate_mmm_dataset(config)
```

### Visualization

```python
from mmm_data_generator import plot_channel_spend, plot_data_quality

# Plot channel spend patterns
fig = plot_channel_spend(data, figsize=(12, 8))
fig.show()

# Plot data quality diagnostics
fig = plot_data_quality(data, figsize=(15, 12))
fig.show()
```

### Ground Truth Analysis

```python
from mmm_data_generator import calculate_roas_values, calculate_attribution_percentages

# Calculate ROAS values
roas_values = calculate_roas_values(spend_data, transformed_data, config)
print("ROAS values:", roas_values)

# Calculate attribution percentages
attribution = calculate_attribution_percentages(transformed_data, config)
print("Attribution percentages:", attribution)
```

## Output Schema

The generated dataset follows a strict schema structure:

### Required Columns
- `date`: datetime values representing the time period for each observation
- `geo`: string values representing the geographic region for each observation  
- `y`: numeric values representing total sales for the specific date and region combination

### Channel Columns
- `x1-*`, `x2-*`, `x3-*`, ...: numeric values representing spend for each advertising channel
- Channel columns are named sequentially starting with "x1"
- Names may include hyphenated suffixes for regional variations

### Control Variable Columns
- `c1-*`, `c2-*`, `c3-*`, ...: numeric values representing control variables (price, promotions, external factors)
- Control variable columns are named sequentially starting with "c1"
- Names may include hyphenated suffixes for regional variations

### Data Integrity Requirements
- Each row has a unique combination of `date` and `geo` values
- All numeric columns contain non-negative values
- No missing values are allowed in any column
- The dataframe is sorted by `date` and then by `geo` for consistent ordering

## Configuration Options

### Channel Patterns

#### Linear Trend
```python
ChannelConfig(
    name="x-TV",
    pattern="linear_trend",
    base_spend=2000.0,
    spend_trend=0.05,  # 5% growth per period
    base_effectiveness=0.6
)
```

#### Seasonal
```python
ChannelConfig(
    name="x-Digital",
    pattern="seasonal",
    base_spend=1500.0,
    seasonal_amplitude=0.4,  # 40% seasonal variation
    seasonal_phase=0.0,      # Phase shift in radians
    base_effectiveness=0.5
)
```

#### Delayed Start
```python
ChannelConfig(
    name="x-Print",
    pattern="delayed_start",
    base_spend=800.0,
    start_period=10,         # Start at period 10
    ramp_up_periods=4,       # Ramp up over 4 periods
    base_effectiveness=0.3
)
```

#### On/Off
```python
ChannelConfig(
    name="x-Radio",
    pattern="on_off",
    base_spend=1000.0,
    activation_probability=0.7,    # 70% chance of being active
    min_active_periods=2,          # Minimum 2 consecutive active periods
    max_active_periods=8,          # Maximum 8 consecutive active periods
    base_effectiveness=0.4
)
```

### Regional Configuration

```python
RegionConfig(
    n_regions=3,
    region_names=["geo_a", "geo_b", "geo_c"],
    base_sales_rate=10000.0,
    sales_trend=0.02,
    sales_volatility=0.15,
    seasonal_amplitude=0.2
)
```

### Control Variables Configuration

```python
ControlConfig(
    name="price",
    pattern="linear_trend",
    base_value=10.0,
    value_volatility=1.0,
    value_trend=0.05,
    base_effectiveness=0.3
)
```

### Transformation Configuration

```python
TransformConfig(
    adstock_fun="geometric_adstock",
    adstock_kwargs=[{"alpha": 0.6}, {"alpha": 0.7}, {"alpha": 0.8}],
    saturation_fun="hill_function",
    saturation_kwargs=[{"slope": 1.0, "kappa": 2000.0}, {"slope": 1.0, "kappa": 2500.0}, {"slope": 1.0, "kappa": 3000.0}]
)
```

## Available Presets

The package includes several pre-configured presets for common MMM scenarios:

- **`basic`**: Simple configuration for learning and testing
- **`seasonal`**: Strong seasonal patterns for seasonal business analysis  
- **`multi_region`**: Multi-regional analysis with regional variations
- **`small_business`**: Small business with limited budget and channels
- **`medium_business`**: Medium business with more budget and channels
- **`large_business`**: Large business with many channels and regions

```python
from mmm_data_generator import get_preset_config, list_available_presets

# List all available presets
presets = list_available_presets()
print(presets)

# Use a specific preset
config = get_preset_config('small_business')
result = generate_mmm_dataset(config)
```

## Visualization Functions

The package provides comprehensive visualization tools:

- **`plot_channel_spend()`**: Plot channel spend patterns over time
- **`plot_channel_contributions()`**: Visualize channel contributions over time  
- **`plot_roas_comparison()`**: ROAS comparison charts across channels
- **`plot_regional_comparison()`**: Regional comparison visualization functions
- **`plot_data_quality()`**: Comprehensive data validation plots

```python
from mmm_data_generator import plot_channel_spend, plot_data_quality

# Plot channel spend patterns
fig = plot_channel_spend(data, channels=['x-TV', 'x-Digital'], figsize=(12, 8))

# Plot comprehensive data quality diagnostics
fig = plot_data_quality(data, figsize=(15, 12))
```

## Ground Truth Analysis

The package provides comprehensive ground truth calculation and analysis:

- **`calculate_roas_values()`**: Calculate true ROAS values for each channel and region
- **`calculate_attribution_percentages()`**: Calculate true attribution percentages

```python
from mmm_data_generator import calculate_roas_values, calculate_attribution_percentages

# Calculate ROAS values
roas_values = calculate_roas_values(spend_data, transformed_data, config)

# Calculate attribution percentages  
attribution = calculate_attribution_percentages(transformed_data, config)
```

## Package Structure

The `mmm_data_generator` package is organized into several modules:

- **`core.py`**: Main data generation function (`generate_mmm_dataset`)
- **`config.py`**: Configuration classes (`MMMDataConfig`, `ChannelConfig`, `RegionConfig`, `TransformConfig`, `ControlConfig`)
- **`presets.py`**: Pre-configured settings for common scenarios
- **`channels.py`**: Channel pattern generation logic
- **`regions.py`**: Regional data generation and variations
- **`transforms.py`**: Adstock and saturation transformations
- **`ground_truth.py`**: Ground truth calculation and ROAS computation
- **`visualization.py`**: Plotting and visualization functions
- **`validation.py`**: Data validation and quality checks
- **`utils.py`**: Utility functions and seed management

## API Reference

### Main Functions

- **`generate_mmm_dataset(config=None, **kwargs)`**: Generate synthetic MMM dataset
- **`get_preset_config(preset_name, seed=2025_07_15)`**: Get preset configuration
- **`list_available_presets()`**: List all available presets
- **`customize_preset(preset_name, **overrides)`**: Customize a preset configuration

### Configuration Classes

- **`MMMDataConfig`**: Main configuration class
- **`ChannelConfig`**: Channel configuration
- **`RegionConfig`**: Regional configuration  
- **`TransformConfig`**: Transformation configuration
- **`ControlConfig`**: Control variable configuration

### Visualization Functions

- **`plot_channel_spend(data, **kwargs)`**: Plot channel spend patterns
- **`plot_channel_contributions(data, **kwargs)`**: Plot channel contributions
- **`plot_roas_comparison(data, **kwargs)`**: Plot ROAS comparisons
- **`plot_regional_comparison(data, **kwargs)`**: Plot regional comparisons
- **`plot_data_quality(data, **kwargs)`**: Plot data quality diagnostics

### Ground Truth Functions

- **`calculate_roas_values(spend_data, transformed_data, config)`**: Calculate ROAS values
- **`calculate_attribution_percentages(transformed_data, config)`**: Calculate attribution percentages

### Validation Functions

- **`validate_config(config)`**: Validate configuration parameters
- **`check_data_quality(data, config)`**: Check data quality
- **`validate_output_schema(data, config)`**: Validate output schema
