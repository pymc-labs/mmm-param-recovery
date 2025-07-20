"""
Configuration presets for common MMM use cases.

This module provides pre-configured settings for typical MMM scenarios,
making it easy for users to get started with common configurations.
"""

from math import pi
from typing import Dict, Any
from .config import ControlConfig, MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


def get_preset_config(preset_name: str, seed: int = 2025_07_15) -> MMMDataConfig:
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
        'basic': _get_basic_preset,
        'seasonal': _get_seasonal_preset,
        'multi_region': _get_multi_region_preset,
        'small_business': _get_small_business_preset,
        'medium_business': _get_medium_business_preset,
        'large_business': _get_large_business_preset,
    }
    
    if preset_name not in presets:
        available_presets = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    return presets[preset_name](seed)


def _get_basic_preset(seed: int) -> MMMDataConfig:
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
            saturation_kwargs={"slope": 1.0, "kappa": 0.15}
        ),
        seed=seed
    )


def _get_seasonal_preset(seed: int) -> MMMDataConfig:
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
            saturation_kwargs={"slope": 1.0, "kappa": 0.2}
        ),
        seed=seed
    )


def _get_multi_region_preset(seed: int) -> MMMDataConfig:
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
            saturation_kwargs={"slope": 1.0, "kappa": 0.8}
        ),
        seed=seed
    )


def _get_small_business_preset(seed: int) -> MMMDataConfig:
    """Preset for small business with limited budget and channels."""
    return MMMDataConfig(
        n_periods=104,
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=100.0,
                spend_trend=0.004, # about 20% per year
                base_effectiveness=1.5,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.3,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=1.2,
            ),
            ChannelConfig(
                name="Local-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.6,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=.9,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.5,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
                base_effectiveness=1.2,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=100.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=50.0,
                value_volatility=0.5,
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
            sales_volatility=0.4,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0}, 
                {"alpha": 0.2}, 
                {"alpha": 0.4}, 
                {"alpha": 0.3}
            ],
            saturation_fun="hill_function",
            saturation_kwargs =[
                {"slope": 1, "kappa": 1}, 
                {"slope": 1.5, "kappa": 0.8}, 
                {"slope": 1, "kappa": 1.5}, 
                {"slope": 2, "kappa": 0.5}
            ],
        ),
        seed=seed,
    )



def _get_medium_business_preset(seed: int) -> MMMDataConfig:
    """Preset for medium business with more budget and channels."""
    return MMMDataConfig(
        n_periods=131, # 2.5 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.7,
                base_spend=300.0,
                spend_trend=0.002, # about 20% per year
                base_effectiveness=0.3,
            ),
            ChannelConfig(
                name="Search-Ads-Brand",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=100.0,
                spend_trend=0.002, # about 20% per year
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Video",
                pattern="delayed_start",
                base_spend=500.0,
                spend_volatility=0.5,
                start_period=25,
                ramp_up_periods=10,
                base_effectiveness=1.4,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.6,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=0.7,
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.5,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=0.3,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.4,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
                base_effectiveness=0.4,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=500.0,
                value_volatility=0.5,
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
            base_sales_rate=5000.0,
            sales_volatility=0.7,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0}, 
                {"alpha": 0.2}, 
                {"alpha": 0.4}, 
                {"alpha": 0.3}
            ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 1, "kappa": 0.3},
                {"slope": 1.5, "kappa": 1.5},
                {"slope": 0.8, "kappa": 0.4},
                {"slope": 2, "kappa": 0.5},
                {"slope": .4, "kappa": 2}, # 
                {"slope": 1.5, "kappa": 1.25},
            ],
        ),
        seed=seed
    )

def _get_large_business_preset(seed: int) -> MMMDataConfig:
    """Preset for large business with many channels and regions."""
    return MMMDataConfig(
        n_periods=156, # 3 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=300.0,
                spend_trend=0.002, # about 20% per year
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Search-Ads-Brand",
                pattern="linear_trend",
                spend_volatility=1.1,
                base_spend=100.0,
                spend_trend=0.002, # about 20% per year
                base_effectiveness=1.2,
            ),
            ChannelConfig(
                name="Video",
                pattern="delayed_start",
                base_spend=500.0,
                spend_volatility=0.9,
                start_period=25,
                ramp_up_periods=10,
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Video-2",
                pattern="delayed_start",
                base_spend=1000.0,
                spend_volatility=1.2,
                start_period=75,
                end_period=125,
                base_effectiveness=0.9,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=1000.0,
                spend_volatility=1.2,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=0.5,
            ),
            ChannelConfig(
                name="Social-Media-2",
                pattern="seasonal",
                base_spend=1000.0,
                spend_volatility=0.9,
                seasonal_amplitude=0.2,
                seasonal_phase=0,
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.7,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=0.8,
            ),
            ChannelConfig(
                name="Influencer",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=1.1,
                activation_probability=0.1,
                min_active_periods=2,
                max_active_periods=6,
                base_effectiveness=1.3,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event-A",
                pattern="on_off",
                base_value=500.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Event-B",
                pattern="on_off",
                base_value=1000.0,
                value_volatility=0.5,
                base_effectiveness=0.2,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=8,
            ),
            ControlConfig(
                name="Linear",
                pattern="linear_trend",
                base_value=1000.0,
                value_trend=0.005,
                value_volatility=1,
                base_effectiveness=0.8,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=25.0,
                value_volatility=0.5,
                base_effectiveness=100.0,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=8,
            region_names=[
                "geo_a", "geo_b", "geo_c", "geo_d",
                "geo_e", "geo_f", "geo_g", "geo_h"
                ],
            base_sales_rate=1000000.0,
            sales_volatility=0.5,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0}, 
                {"alpha": 0.2}, 
                {"alpha": 0.4}, 
                {"alpha": 0.3},
                {"alpha": 0.2},
                {"alpha": 0.1},
                {"alpha": 0.05},
                {"alpha": 0.1},
                ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 1, "kappa": 1},
                {"slope": 1.5, "kappa": 0.4},
                {"slope": 0.8, "kappa": 0.55},
                {"slope": 2, "kappa": 0.7},
                {"slope": 1, "kappa": 0.4},
                {"slope": 0.8, "kappa": 1.2},
                {"slope": 0.5, "kappa": 0.5},
                {"slope": 1.2, "kappa": 0.9},
            ],
        ),
        seed=seed
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
        'small_business': 'Small business with limited budget and channels',
        'medium_business': 'Medium business with more budget and channels',
        'large_business': 'Large business with many channels and regions',
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