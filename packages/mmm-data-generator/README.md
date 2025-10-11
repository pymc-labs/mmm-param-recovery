# MMM Data Generator

A lightweight Python package for generating synthetic Marketing Mix Modeling (MMM) datasets with known ground truth parameters. This package provides tools to create reproducible datasets for benchmarking MMM models and parameter recovery studies.

## Features

- **Configurable Channel Patterns**: Create realistic spend patterns for various marketing channels
- **Geographic Variations**: Generate regional variations in baseline and channel effectiveness
- **Adstock & Saturation**: Apply realistic adstock decay and saturation transformations
- **Ground Truth**: Complete ground truth parameters for model validation
- **Preset Configurations**: Pre-built configurations for different business scenarios
- **Visualization**: Built-in plotting functions for data exploration

## Installation

```bash
pip install mmm-data-generator
```

## Quick Start

```python
from mmm_data_generator import generate_mmm_dataset, DEFAULT_CONFIG

# Generate a dataset with default configuration
data = generate_mmm_dataset(config=DEFAULT_CONFIG)

# Access the generated data
print(data.keys())
# ['spend_data', 'revenue_data', 'control_data', 'ground_truth', 'config']

# Plot the results
from mmm_data_generator import plot_channel_spend
plot_channel_spend(data['spend_data'])
```

## Using Presets

```python
from mmm_data_generator import get_preset_config, generate_mmm_dataset

# Get a preset configuration
config = get_preset_config('small_business')

# Generate data with preset
data = generate_mmm_dataset(config=config)
```

## Custom Configuration

```python
from mmm_data_generator import MMMDataConfig, ChannelConfig, generate_mmm_dataset

# Create custom configuration
config = MMMDataConfig(
    n_periods=52,
    n_regions=3,
    channels=[
        ChannelConfig(name='TV', spend_mean=1000, spend_std=200),
        ChannelConfig(name='Digital', spend_mean=500, spend_std=100),
    ]
)

data = generate_mmm_dataset(config=config)
```

## Available Presets

- `small_business`: Small business with limited channels
- `medium_business`: Medium business with moderate complexity
- `growing_business`: Growing business with increasing spend
- `enterprise`: Large enterprise with many channels and regions

## Dependencies

This package has minimal dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pymc-marketing (for transforms)

## Documentation

For detailed documentation, see the [main repository](https://github.com/pymc-labs/mmm-param-recovery).

## License

Apache License 2.0
