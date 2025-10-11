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
) -> Dict[str, np.ndarray]:
    """
    Generate baseline sales components for a specific region with individual patterns.
    
    Creates separate components for base sales, trend, and seasonal components.
    
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
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'base_sales': Base sales component
        - 'trend': Trend component
        - 'seasonal': Seasonal component
        - 'total': Combined baseline sales values
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
    
    # Trend component
    trend =  regions.sales_trend * time_array
    
    # Seasonal component
    seasonal_period = 52  # Weekly data, annual seasonality
    seasonal = regions.seasonal_amplitude * np.sin(2 * np.pi * time_array / seasonal_period)

    # Add noise
    noise = np.random.normal(0, regions.sales_volatility, n_periods)
    total = regions.base_sales_rate * baseline_variation * (1 + trend + seasonal + noise)
    
    # Ensure non-negative
    total = np.clip(total, 0, None)
    
    return {
        'base_sales': regions.base_sales_rate * (1 + noise) ,
        'trend': trend,
        'seasonal': seasonal,
        'total': total
    }


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
        
        # Create new channel config with regional variations
        regional_channel = ChannelConfig(
            name=base_channel.name,
            pattern=base_channel.pattern,
            base_spend=base_channel.base_spend * _vary_param(regions.channel_param_variation),
            spend_trend=base_channel.spend_trend * _vary_param(regions.channel_param_variation),
            spend_volatility=base_channel.spend_volatility * _vary_param(regions.channel_param_variation),
            seasonal_amplitude=base_channel.seasonal_amplitude * _vary_param(regions.channel_param_variation),
            seasonal_phase=base_channel.seasonal_phase * _vary_param(regions.channel_param_variation),
            start_period=base_channel.start_period,
            ramp_up_periods=base_channel.ramp_up_periods,
            activation_probability=base_channel.activation_probability * _vary_param(regions.channel_param_variation),
            min_active_periods=base_channel.min_active_periods,
            max_active_periods=base_channel.max_active_periods,
            custom_pattern_func=base_channel.custom_pattern_func,
            base_effectiveness=base_channel.base_effectiveness * _vary_param(regions.channel_param_variation)
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
                regional_adstock_kwargs[key] = value * _vary_param(regions.transform_variation)
            else:
                regional_adstock_kwargs[key] = value
    else:
        regional_adstock_kwargs = base_transforms.adstock_kwargs
    
    # Apply variations to saturation parameters
    regional_saturation_kwargs = {}
    if isinstance(base_transforms.saturation_kwargs, dict):
        for key, value in base_transforms.saturation_kwargs.items():
            if isinstance(value, float):
                regional_saturation_kwargs[key] = value * _vary_param(regions.transform_variation)
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

def _vary_param(variation: float):
    return 1.0 + np.random.uniform(-variation, variation)