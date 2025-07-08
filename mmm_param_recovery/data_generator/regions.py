"""
Geographic region management for MMM Dataset Generator.

This module contains functions for managing regional data generation with simplified
variation patterns. Each region gets individual baseline patterns and slight variations
in channel parameters and transformation settings.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .config import RegionConfig, ChannelConfig, TransformConfig


def generate_regional_baseline(
    regions: RegionConfig,
    time_index: pd.DatetimeIndex,
    region_idx: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate baseline sales for a specific region with individual patterns.
    
    Creates baseline sales with trend, seasonality, and regional variation.
    
    Parameters
    ----------
    regions : RegionConfig
        Region configuration
    time_index : pd.DatetimeIndex
        Time index for the data
    region_idx : int
        Index of the region
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Baseline sales values for each time period
    """
    if seed is not None:
        np.random.seed(seed + region_idx)  # Different seed per region
    
    n_periods = len(time_index)
    
    # Generate regional baseline variation
    baseline_variation = 1.0 + np.random.uniform(
        -regions.baseline_variation, 
        regions.baseline_variation
    )
    
    # Create time-based baseline with trend
    time_array = np.arange(n_periods)
    baseline = regions.base_sales_rate * baseline_variation * (1 + regions.sales_trend * time_array)
    
    # Add seasonal component
    seasonal_period = 52  # Weekly data, annual seasonality
    seasonal_component = regions.seasonal_amplitude * np.sin(2 * np.pi * time_array / seasonal_period)
    baseline *= (1 + seasonal_component)
    
    # Add noise
    noise = np.random.normal(0, regions.sales_volatility, n_periods)
    baseline *= (1 + noise)
    
    return np.clip(baseline, 0, None)  # Ensure non-negative


def generate_regional_channel_variations(
    regions: RegionConfig,
    base_channels: list[ChannelConfig],
    region_idx: int,
    seed: Optional[int] = None
) -> list[ChannelConfig]:
    """
    Generate channel configurations with regional variations.
    
    Applies scaling factors to channel parameters to simulate regional variation.
    
    Parameters
    ----------
    regions : RegionConfig
        Region configuration
    base_channels : list[ChannelConfig]
        Base channel configurations
    region_idx : int
        Index of the region
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list[ChannelConfig]
        Channel configurations with regional variations
    """
    if seed is not None:
        np.random.seed(seed + region_idx * 1000)  # Different seed per region
    
    regional_channels = []
    
    for i, base_channel in enumerate(base_channels):
        # Create regional variation factors
        spend_scale = 1.0 + np.random.uniform(
            -regions.channel_scale_variation, 
            regions.channel_scale_variation
        )
        effectiveness_scale = 1.0 + np.random.uniform(
            -regions.effectiveness_variation, 
            regions.effectiveness_variation
        )
        
        # Create new channel config with regional variations
        regional_channel = ChannelConfig(
            name=base_channel.name,
            pattern=base_channel.pattern,
            base_spend=base_channel.base_spend * spend_scale,
            spend_trend=base_channel.spend_trend,
            spend_volatility=base_channel.spend_volatility,
            seasonal_amplitude=base_channel.seasonal_amplitude,
            seasonal_phase=base_channel.seasonal_phase,
            start_period=base_channel.start_period,
            ramp_up_periods=base_channel.ramp_up_periods,
            activation_probability=base_channel.activation_probability,
            min_active_periods=base_channel.min_active_periods,
            max_active_periods=base_channel.max_active_periods,
            custom_pattern_func=base_channel.custom_pattern_func,
            base_effectiveness=base_channel.base_effectiveness * effectiveness_scale,
            effectiveness_trend=base_channel.effectiveness_trend
        )
        
        regional_channels.append(regional_channel)
    
    return regional_channels


def generate_regional_transform_variations(
    regions: RegionConfig,
    base_transforms: TransformConfig,
    region_idx: int,
    seed: Optional[int] = None
) -> TransformConfig:
    """
    Generate transformation configurations with regional variations.
    
    Applies small variations to transformation parameters for regional differences.
    
    Parameters
    ----------
    regions : RegionConfig
        Region configuration
    base_transforms : TransformConfig
        Base transformation configuration
    region_idx : int
        Index of the region
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    TransformConfig
        Transformation configuration with regional variations
    """
    if seed is not None:
        np.random.seed(seed + region_idx * 2000)  # Different seed per region
    
    # Create regional variation factors for transformation parameters
    
    
    # Apply variations to adstock parameters
    regional_adstock_kwargs = {}
    if isinstance(base_transforms.adstock_kwargs, dict):
        for key, value in base_transforms.adstock_kwargs.items():
            if isinstance(value, float):
                transform_scale = 1.0 + np.random.uniform(
                    -regions.transform_variation, 
                    regions.transform_variation
                )
                regional_adstock_kwargs[key] = value * transform_scale
            else:
                regional_adstock_kwargs[key] = value
    else:
        regional_adstock_kwargs = base_transforms.adstock_kwargs
    
    # Apply variations to saturation parameters
    regional_saturation_kwargs = {}
    if isinstance(base_transforms.saturation_kwargs, dict):
        for key, value in base_transforms.saturation_kwargs.items():
            if isinstance(value, float):
                transform_scale = 1.0 + np.random.uniform(
                    -regions.transform_variation, 
                    regions.transform_variation
                )
                regional_saturation_kwargs[key] = value * transform_scale
            else:
                regional_saturation_kwargs[key] = value
    else:
        regional_saturation_kwargs = base_transforms.saturation_kwargs
    
    return TransformConfig(
        adstock_fun=base_transforms.adstock_fun,
        adstock_kwargs=regional_adstock_kwargs,
        saturation_fun=base_transforms.saturation_fun,
        saturation_kwargs=regional_saturation_kwargs
    ) 