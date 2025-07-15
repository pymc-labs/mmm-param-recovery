"""
Configuration presets for common MMM use cases.

This module provides pre-configured settings for typical MMM scenarios,
making it easy for users to get started with common configurations.
"""

from typing import Dict, Any
from .config import ControlConfig, MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


def get_preset_config(preset_name: str) -> MMMDataConfig:
    """
    Get a preset configuration for common MMM use cases.
    
    Parameters
    ----------
    preset_name : str
        Name of the preset configuration
        
    Returns
    -------
    MMMConfig
        Pre-configured MMM configuration
        
    Raises
    ------
    ValueError
        If preset name is not recognized
    """
    presets = {
        'basic': _get_basic_preset(),
        'seasonal': _get_seasonal_preset(),
        'multi_region': _get_multi_region_preset(),
        'high_frequency': _get_high_frequency_preset(),
        'digital_heavy': _get_digital_heavy_preset(),
        'traditional_media': _get_traditional_media_preset(),
        'small_business': _get_small_business_preset(),
        'enterprise': _get_enterprise_preset(),
        'research': _get_research_preset(),
        'demo': _get_demo_preset()
    }
    
    if preset_name not in presets:
        available_presets = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    return presets[preset_name]


def _get_basic_preset() -> MMMDataConfig:
    """Basic preset for learning and testing."""
    return MMMDataConfig(
        n_periods=52,  # 1 year of weekly data
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="linear_trend",
                base_spend=2000.0,
                spend_trend=0.02,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="linear_trend",
                base_spend=1500.0,
                spend_trend=0.05,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=800.0,
                spend_trend=-0.01,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.5},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 1500.0}
        ),
        seed=42
    )


def _get_seasonal_preset() -> MMMDataConfig:
    """Preset with strong seasonal patterns for seasonal business analysis."""
    return MMMDataConfig(
        n_periods=104,  # 2 years for seasonal analysis
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="seasonal",
                base_spend=3000.0,
                seasonal_amplitude=0.6,
                seasonal_phase=0.0,
                base_effectiveness=0.7
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="seasonal",
                base_spend=2000.0,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,  # Offset phase
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Radio",
                pattern="seasonal",
                base_spend=1000.0,
                seasonal_amplitude=0.3,
                seasonal_phase=1.0,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"],
            seasonal_amplitude=0.3
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 2000.0}
        ),
        seed=123
    )


def _get_multi_region_preset() -> MMMDataConfig:
    """Preset for multi-regional analysis with regional variations."""
    return MMMDataConfig(
        n_periods=78,  # 1.5 years
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="linear_trend",
                base_spend=2500.0,
                spend_trend=0.01,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="linear_trend",
                base_spend=1800.0,
                spend_trend=0.08,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=800.0,
                spend_trend=-0.03,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=4,
            region_names=["geo_a", "geo_b", "geo_c", "geo_d"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.55},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 1800.0}
        ),
        seed=456
    )


def _get_high_frequency_preset() -> MMMDataConfig:
    """Preset for high-frequency digital marketing analysis."""
    return MMMDataConfig(
        n_periods=156,  # 3 years of weekly data
        channels=[
            ChannelConfig(
                name="x-Paid-Search",
                pattern="on_off",
                base_spend=1200.0,
                activation_probability=0.9,
                min_active_periods=1,
                max_active_periods=4,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="x-Social-Media",
                pattern="linear_trend",
                base_spend=800.0,
                spend_trend=0.12,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Display",
                pattern="on_off",
                base_spend=600.0,
                activation_probability=0.7,
                min_active_periods=2,
                max_active_periods=6,
                base_effectiveness=0.4
            ),
            ChannelConfig(
                name="x-Email",
                pattern="seasonal",
                base_spend=300.0,
                seasonal_amplitude=0.2,
                base_effectiveness=0.9
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.3},  # Lower adstock for digital channels
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 1000.0}
        ),
        seed=789
    )


def _get_digital_heavy_preset() -> MMMDataConfig:
    """Preset focused on digital marketing channels."""
    return MMMDataConfig(
        n_periods=104,
        channels=[
            ChannelConfig(
                name="x-Paid-Search",
                pattern="linear_trend",
                base_spend=2000.0,
                spend_trend=0.15,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="x-Social-Media",
                pattern="seasonal",
                base_spend=1500.0,
                seasonal_amplitude=0.4,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Display",
                pattern="on_off",
                base_spend=1000.0,
                activation_probability=0.8,
                base_effectiveness=0.4
            ),
            ChannelConfig(
                name="x-Video",
                pattern="linear_trend",
                base_spend=1200.0,
                spend_trend=0.2,
                base_effectiveness=0.7
            ),
            ChannelConfig(
                name="x-Email",
                pattern="seasonal",
                base_spend=500.0,
                seasonal_amplitude=0.3,
                base_effectiveness=0.9
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.4},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 1500.0}
        ),
        seed=321
    )


def _get_traditional_media_preset() -> MMMDataConfig:
    """Preset focused on traditional media channels."""
    return MMMDataConfig(
        n_periods=104,
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="seasonal",
                base_spend=5000.0,
                seasonal_amplitude=0.5,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="x-Radio",
                pattern="seasonal",
                base_spend=2000.0,
                seasonal_amplitude=0.3,
                base_effectiveness=0.4
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=1500.0,
                spend_trend=-0.05,
                base_effectiveness=0.3
            ),
            ChannelConfig(
                name="x-Outdoor",
                pattern="seasonal",
                base_spend=1000.0,
                seasonal_amplitude=0.2,
                base_effectiveness=0.5
            )
        ],
        regions=RegionConfig(
            n_regions=3,
            region_names=["geo_a", "geo_b", "geo_c"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.7},  # Higher adstock for traditional media
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 3000.0}
        ),
        seed=654
    )


def _get_small_business_preset() -> MMMDataConfig:
    """Preset for small business with limited budget and channels."""
    return MMMDataConfig(
        n_periods=104,
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.02,
                base_spend=100.0,
                spend_trend=0.004, # about 20% per year
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.05,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5
            ),
            ChannelConfig(
                name="Local-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.05,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.05,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=100.0,
                value_volatility=0.05,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=50.0,
                value_volatility=0.05,
                base_effectiveness=0.5,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["Local"],
            base_sales_rate=5000.0,
            sales_volatility=0.05,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": [0, 0.2, 0.4, 0.3]},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": [1, 1.5, 1, 2], "kappa": 0.7}
        ),
        seed=2025_07_15
    )



def _get_medium_business_preset() -> MMMDataConfig:
    """Preset for medium business with more budget and channels."""
    return MMMDataConfig(
        n_periods=131, # 2.5 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.02,
                base_spend=300.0,
                spend_trend=0.002, # about 20% per year
            ),
            ChannelConfig(
                name="Search-Ads-Brand",
                pattern="linear_trend",
                spend_volatility=0.01,
                base_spend=100.0,
                spend_trend=0.002, # about 20% per year
            ),
            ChannelConfig(
                name="Video",
                pattern="delayed_start",
                base_spend=500.0,
                spend_volatility=0.05,
                start_period=25,
                ramp_up_periods=10,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.05,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.05,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.05,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=500.0,
                value_volatility=0.05,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=25.0,
                value_volatility=0.5,
                base_effectiveness=1.5,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=2,
            region_names=["geo_a", "geo_b"],
            base_sales_rate=10000.0,
            sales_volatility=0.05,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": [0, 0.2, 0.4, 0.3]},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": [1, 1.5, 1, 2], "kappa": 0.7}
        ),
        seed=2025_07_15
    )

def _get_enterprise_preset() -> MMMDataConfig:
    """Preset for enterprise-level analysis with many channels and regions."""
    return MMMDataConfig(
        n_periods=156,
        channels=[
            ChannelConfig(
                name="x-TV-National",
                pattern="seasonal",
                base_spend=8000.0,
                seasonal_amplitude=0.4,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="x-TV-Regional",
                pattern="seasonal",
                base_spend=4000.0,
                seasonal_amplitude=0.3,
                base_effectiveness=0.7
            ),
            ChannelConfig(
                name="x-Digital-Display",
                pattern="linear_trend",
                base_spend=3000.0,
                spend_trend=0.1,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Paid-Search",
                pattern="linear_trend",
                base_spend=2500.0,
                spend_trend=0.15,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="x-Social-Media",
                pattern="seasonal",
                base_spend=2000.0,
                seasonal_amplitude=0.2,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=1500.0,
                spend_trend=-0.02,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=8,
            region_names=["geo_a", "geo_b", "geo_c", "geo_d", "geo_e", "geo_f", "geo_g", "geo_h"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 4000.0}
        ),
        seed=147
    )


def list_available_presets() -> Dict[str, str]:
    """
    Get a list of available presets with descriptions.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping preset names to descriptions
    """
    return {
        'basic': 'Simple configuration for learning and testing',
        'seasonal': 'Strong seasonal patterns for seasonal business analysis',
        'multi_region': 'Multi-regional analysis with regional variations',
        'high_frequency': 'High-frequency digital marketing analysis',
        'digital_heavy': 'Focused on digital marketing channels',
        'traditional_media': 'Focused on traditional media channels',
        'small_business': 'Small business with limited budget and channels',
        'enterprise': 'Enterprise-level analysis with many channels and regions',
        }


def customize_preset(preset_name: str, **overrides) -> MMMDataConfig:
    """
    Get a preset configuration with custom overrides.
    
    Parameters
    ----------
    preset_name : str
        Name of the preset configuration
    **overrides
        Parameters to override in the preset
        
    Returns
    -------
    MMMConfig
        Customized preset configuration
    """
    config = get_preset_config(preset_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to set nested attributes
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"Invalid override key: {key}")
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
            else:
                raise ValueError(f"Invalid override key: {key}")
    
    return config 