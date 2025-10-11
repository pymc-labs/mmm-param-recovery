# MMM Dataset Generator

A lightweight Python module for generating reproducible datasets for benchmarking Marketing Mix Models (MMM) with known ground truth parameters.

## Overview

The MMM Dataset Generator creates synthetic marketing data with configurable channel patterns, geographic regions, adstock/saturation transformations, and comprehensive ground truth information for model validation.

## Key Features

### ✅ Currently Implemented

#### Core Data Generation
- **Main API**: Single `generate_mmm_dataset()` function as the primary entry point
- **Configuration System**: Comprehensive dataclasses for all parameters (`MMMDataConfig`, `ChannelConfig`, `RegionConfig`, `TransformConfig`)
- **Reproducibility**: Complete seed management and random number generation utilities
- **Data Validation**: Comprehensive parameter validation and data quality checks
- **Output Schema**: Strict compliance with required column structure (`date`, `geo`, `y`, `x1-*`, `x2-*`, ..., `c1-*`, `c2-*`, ...)

#### Channel Pattern Generation
- **Linear Trend**: Configurable linear trend channels with customizable spend growth/decline
- **Seasonal**: Annual seasonality patterns with adjustable amplitude and phase
- **Delayed Start**: Channels with configurable start times and ramp-up periods
- **On/Off**: Random activation patterns with controllable probability and duration
- **Custom**: Support for user-defined channel patterns via function inputs

#### Adstock and Saturation Transformations
- **PyMC Marketing Integration**: Wrapper for applying adstock and saturation transforms from PyMC Marketing
- **Flexible Combinations**: Support for different adstock/saturation combinations per channel

#### Configuration Presets
- **Basic**: Simple configuration for learning and testing
- **Seasonal**: Strong seasonal patterns for seasonal business analysis
- **Multi-Region**: Multi-regional analysis with regional variations
- **High Frequency**: High-frequency digital marketing analysis
- **Digital Heavy**: Digital-heavy marketing mix
- **Traditional Media**: Traditional media-focused configurations
- **Small Business**: Small business scenarios
- **Enterprise**: Enterprise-level configurations
- **Research**: Research-focused settings
- **Demo**: Demonstration configurations

#### Data Quality and Validation
- **Parameter Validation**: Comprehensive validation of all configuration parameters
- **Data Integrity**: Ensures unique date-geo combinations and no missing values
- **Quality Checks**: Non-negative values and reasonable range validation
- **Warning System**: Helpful warnings for potentially problematic configurations

### 🚧 Planned Features (In Progress)

#### Geographic Region Management
- **Multi-Region Support**: Generate data for 1-50 geographic regions
- **Regional Variations**: Different baseline sales rates and channel effectiveness per region
- **Region Similarity Controls**: Configurable similarity levels between regions
- **Regional Parameter Validation**: Region-specific parameter validation

#### Ground Truth Calculation and ROAS
- **True Parameter Tracking**: Complete tracking of betas, alphas, kappas, etc.
- **Channel Contribution Calculation**: True channel contributions over time
- **ROAS Computation**: True ROAS values for each channel and region
- **Attribution Percentages**: True attribution percentage calculations
- **Baseline Component Tracking**: Intercept, trend, seasonality, and control variable tracking

#### Visualization and Validation Functions
- **Channel Spend Visualization**: Plot channel spend patterns over time
- **Channel Contribution Plots**: Visualize channel contributions over time
- **ROAS Visualization**: ROAS comparison charts across channels
- **Regional Comparison**: Regional comparison visualization functions
- **Data Quality Diagnostics**: Comprehensive data validation plots

#### Testing and Documentation
- **Unit Tests**: Complete test suite for all components
- **Integration Tests**: End-to-end pipeline testing
- **Performance Benchmarks**: Scalability testing
- **Comprehensive Documentation**: API documentation and usage examples

## Quick Start

### Basic Usage

```python
from mmm_param_recovery.data_generator import generate_mmm_dataset, DEFAULT_CONFIG

# Generate dataset with default configuration
result = generate_mmm_dataset()
data = result['data']
ground_truth = result['ground_truth']

print(f"Generated {len(data)} rows of data")
print(f"Columns: {list(data.columns)}")
print("Sample opf the generated data:")
print(data.head())
```

### Using Presets

```python
from mmm_param_recovery.data_generator import get_preset_config, generate_mmm_dataset

# Use a preset configuration
config = get_preset_config('seasonal')
result = generate_mmm_dataset(config)

# Customize a preset
from mmm_param_recovery.data_generator import customize_preset
config = customize_preset('basic', n_periods=104, seed=123)
result = generate_mmm_dataset(config)
```

### Custom Configuration

```python
from mmm_param_recovery.data_generator import MMMDataConfig, ChannelConfig, generate_mmm_dataset

# Create custom configuration
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
    seed=42
)

result = generate_mmm_dataset(config)
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

### Transformation Configuration

```python
TransformConfig(
    adstock_fun="geometric_adstock",
    adstock_kwargs={"alpha": 0.5},
    saturation_fun="hill_function", 
    saturation_kwargs={"slope": 1.0, "kappa": 1500.0}
)
```
