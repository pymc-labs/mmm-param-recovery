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
Configuration classes for MMM Dataset Generator.

This module defines the configuration dataclasses used to specify parameters
for data generation, including channel patterns, regional settings, and
transformation functions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Callable, Union
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for a single marketing channel."""
    
    name: str
    pattern: Literal["linear_trend", "seasonal", "delayed_start", "on_off", "custom"] = "linear_trend"
    
    # Spend pattern parameters
    base_spend: float = 1000.0
    spend_trend: float = 0.0  # Linear trend in spend over time
    spend_volatility: float = 0.1  # Coefficient of variation for spend noise
    
    # Seasonal parameters (for seasonal pattern)
    seasonal_amplitude: float = 0.3  # Amplitude of seasonal variation
    seasonal_phase: float = 0.0  # Phase shift in radians
    
    # Delayed start parameters
    start_period: int = 0  # When channel starts (None = immediate)
    ramp_up_periods: int = 4  # Number of periods to ramp up to full spend
    end_period: Optional[int] = None  # When channel ends (None = never ends)

    # On/off parameters
    activation_probability: float = 0.4  # Probability of being active in any period
    min_active_periods: int = 2  # Minimum consecutive active periods
    max_active_periods: int = 8  # Maximum consecutive active periods
    
    # Custom pattern function
    custom_pattern_func: Optional[Callable] = None
    
    # Effectiveness parameters
    base_effectiveness: float = 0.5  # Base effectiveness multiplier
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.base_spend < 0:
            raise ValueError("base_spend must be non-negative")
        if self.spend_volatility < 0:
            raise ValueError("spend_volatility must be non-negative")
        # Seasonal parameters
        if not 0 <= self.seasonal_amplitude:
            raise ValueError("seasonal_amplitude must be positive.")
        if not -2 * np.pi <= self.seasonal_phase <= 2 * np.pi:
            raise ValueError("seasonal_phase should be between -2π and 2π radians")
        # Delayed start parameters
        if self.start_period < 0:
            raise ValueError("start_period must be non-negative")
        if self.ramp_up_periods < 0:
            raise ValueError("ramp_up_periods must be non-negative")
        if self.end_period is not None and self.end_period < self.start_period:
            raise ValueError("end_period must not be before start_period")
        # On/off parameters
        if self.activation_probability < 0 or self.activation_probability > 1:
            raise ValueError("activation_probability must be between 0 and 1")
        if self.min_active_periods < 1:
            raise ValueError("min_active_periods must be at least 1")
        if self.max_active_periods < self.min_active_periods:
            raise ValueError("max_active_periods must be >= min_active_periods")
        # Effectiveness parameters
        if self.base_effectiveness < 0:
            raise ValueError("base_effectiveness must be non-negative")
        # Name validation
        if "_" in self.name:
            raise ValueError("channel name must not contain underscores (use hyphens instead)")


@dataclass
class TransformConfig:
    """Configuration for adstock and saturation transformations."""
    
    # Adstock parameters
    adstock_fun: str | list[str] | None = "geometric_adstock"
    adstock_kwargs: Dict[str, Any] | list[Dict[str, Any]] = field(default_factory=dict)
    
    # Saturation parameters
    saturation_fun: str | list[str] | None = "hill_function"
    saturation_kwargs: Dict[str, Any] | list[Dict[str, Any]] = field(default_factory=dict)
    
    def get_adstock_kwargs(self, channel_idx: int = 0) -> Dict[str, Any]:
        """Get adstock kwargs for a specific channel."""
        if isinstance(self.adstock_kwargs, list):
            # Cycle through the list if channel_idx exceeds length
            return self.adstock_kwargs[channel_idx % len(self.adstock_kwargs)]
        return self.adstock_kwargs
    
    def get_saturation_kwargs(self, channel_idx: int = 0) -> Dict[str, Any]:
        """Get saturation kwargs for a specific channel."""
        if isinstance(self.saturation_kwargs, list):
            # Cycle through the list if channel_idx exceeds length
            return self.saturation_kwargs[channel_idx % len(self.saturation_kwargs)]
        return self.saturation_kwargs


@dataclass
class RegionConfig:
    """Configuration for geographic regions."""
    
    n_regions: int = 1
    region_names: Optional[List[str]] = None
    
    # Baseline sales parameters
    base_sales_rate: float = 10000.0  # Base sales per period
    sales_trend: float = 0.00  # Linear trend in sales over time
    sales_volatility: float = 0.01  # Coefficient of variation for sales noise
    
    # Seasonal parameters
    seasonal_amplitude: float = 0.2  # Amplitude of seasonal variation in sales
    seasonal_phase: float = 0.0  # Phase shift in radians
    
    # Regional variation parameters
    baseline_variation: float = 0.1  # Factor for regional baseline variation (0.1 = ±10%)
    channel_param_variation: float = 0.1  # Factor for channel spend scaling variation (0.1 = ±10%)
    transform_variation: float = 0.1  # Factor for transformation parameter variation (0.1 = ±10%)
    
    def __post_init__(self):
        """Validate region configuration."""
        if self.n_regions < 1:
            raise ValueError("n_regions must be at least 1")
        if self.base_sales_rate <= 0:
            raise ValueError("base_sales_rate must be positive")
        if self.sales_volatility < 0:
            raise ValueError("sales_volatility must be non-negative")
        if not 0 <= self.seasonal_amplitude <= 1:
            raise ValueError("seasonal_amplitude must be between 0 and 1")
        
        # Validate variation factors
        if not 0 <= self.baseline_variation <= 1:
            raise ValueError("baseline_variation_factor must be between 0 and 1")
        if not 0 <= self.channel_param_variation <= 1:
            raise ValueError("channel_param_variation must be between 0 and 1")
        if not 0 <= self.transform_variation <= 1:
            raise ValueError("transform_variation must be between 0 and 1")
        
        # Generate region names if not provided
        if self.region_names is None:
            if self.n_regions == 1:
                self.region_names = ["geo_all"]
            else:
                self.region_names = [f"geo_{i+1}" for i in range(self.n_regions)]
        elif len(self.region_names) != self.n_regions:
            raise ValueError("region_names length must match n_regions")


@dataclass
class ControlConfig(ChannelConfig):
    """Configuration for a control variable."""
    # Control variable specific attributes
    base_value: float = 10.0
    value_volatility: float = 1.0
    value_trend: float = 0.1
    
    # Regional variation parameters (similar to channels)
    regional_value_variation: float = 0.05  # Factor for regional value variation (0.05 = ±5%)
    
    # Override defaults for control variables
    pattern: Literal["linear_trend", "seasonal", "delayed_start", "on_off", "custom"] = "on_off"
    
    @classmethod
    def from_channel_config(cls, channel_config: ChannelConfig) -> 'ControlConfig':
        """
        Create a ControlConfig from a ChannelConfig.
        
        This method maps ChannelConfig attributes to ControlConfig attributes,
        using the channel's spend parameters as the control's value parameters.
        
        Parameters
        ----------
        channel_config : ChannelConfig
            The channel configuration to convert
            
        Returns
        -------
        ControlConfig
            A new control configuration based on the channel configuration
        """
        return cls(
            name=channel_config.name,
            pattern=channel_config.pattern,
            base_value=channel_config.base_spend,
            value_volatility=channel_config.spend_volatility,
            value_trend=channel_config.spend_trend,
            seasonal_amplitude=channel_config.seasonal_amplitude,
            seasonal_phase=channel_config.seasonal_phase,
            start_period=channel_config.start_period,
            ramp_up_periods=channel_config.ramp_up_periods,
            end_period=channel_config.end_period,
            activation_probability=channel_config.activation_probability,
            min_active_periods=channel_config.min_active_periods,
            max_active_periods=channel_config.max_active_periods,
            custom_pattern_func=channel_config.custom_pattern_func,
            base_effectiveness=channel_config.base_effectiveness
        )
    
    def __post_init__(self):
        """Map value attributes to spend attributes and validate."""
        # Map value attributes to spend attributes
        self.base_spend = self.base_value
        self.spend_volatility = self.value_volatility
        self.spend_trend = self.value_trend
        
        if self.spend_volatility < 0:
            raise ValueError("value_volatility must be non-negative")
        # Seasonal parameters
        if not 0 <= self.seasonal_amplitude <= 1:
            raise ValueError("seasonal_amplitude must be between 0 and 1")
        if not -2 * np.pi <= self.seasonal_phase <= 2 * np.pi:
            raise ValueError("seasonal_phase should be between -2π and 2π radians")
        # On/off parameters
        if self.activation_probability < 0 or self.activation_probability > 1:
            raise ValueError("activation_probability must be between 0 and 1")
        if self.min_active_periods < 1:
            raise ValueError("min_active_periods must be at least 1")
        if self.max_active_periods < self.min_active_periods:
            raise ValueError("max_active_periods must be >= min_active_periods")
        # Regional variation validation
        if not 0 <= self.regional_value_variation:
            raise ValueError("regional_value_variation must non-negative")
        # Name validation
        if "_" in self.name:
            raise ValueError("control name must not contain underscores (use hyphens instead)")

    

@dataclass
class MMMDataConfig:
    """Main configuration for MMM dataset generation."""
    
    # Time parameters
    n_periods: int = 129  # Number of time periods (2.5 years of weekly data)
    start_date: Optional[str] = None  # Start date in YYYY-MM-DD format
    
    # Channel configuration
    channels: List[ChannelConfig] = field(default_factory=list)
    
    # Region configuration
    regions: RegionConfig = field(default_factory=RegionConfig)
    
    # Transformation configuration
    transforms: TransformConfig = field(default_factory=TransformConfig)
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    # Control variables
    control_variables: List[ControlConfig] = field(default_factory=list)
    
    # Output options
    include_ground_truth: bool = True
    include_transformed_data: bool = True
    include_raw_data: bool = True
    
    def __post_init__(self):
        """Validate main configuration."""
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive")
        # Validate channels
        if not self.channels:
            raise ValueError("At least one channel must be specified")
        channel_names = [ch.name for ch in self.channels]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("Channel names must be unique")
        # Validate control variables
        for control_var in self.control_variables:
            if not hasattr(control_var, 'name'):
                raise ValueError("Control variable must have a 'name' attribute")
            if control_var.base_spend < 0:
                raise ValueError(f"Control variable {control_var.name} base_spend must be non-negative")


# Default configuration preset
DEFAULT_CONFIG = MMMDataConfig(
    n_periods=129,
    channels=[
        ChannelConfig(
            name="x-seasonal",
            pattern="seasonal",
            base_spend=5000.0,
            seasonal_amplitude=0.4,
            base_effectiveness=0.8
        ),
        ChannelConfig(
            name="x-linear-trend",
            pattern="linear_trend",
            base_spend=3000.0,
            spend_trend=0.05,
            base_effectiveness=0.6
        ),
        ChannelConfig(
            name="x-on-off",
            pattern="on_off",
            base_spend=1500.0,
            activation_probability=0.7,
            base_effectiveness=0.4
        )
    ],
    regions=RegionConfig(
        n_regions=3,
        region_names=["geo_a", "geo_b", "geo_c"],
        baseline_variation=0.1,
        channel_param_variation=0.05,
        transform_variation=0.03
    ),
    transforms=TransformConfig(
        adstock_fun="geometric_adstock",
        adstock_kwargs=[{"alpha": 0.6}, {"alpha": 0.7}, {"alpha": 0.8}],
        saturation_fun="hill_function",
        saturation_kwargs=[{"slope": 1.0, "kappa": 2000.0}, {"slope": 1.0, "kappa": 2500.0}, {"slope": 1.0, "kappa": 3000.0}]
    ),
    control_variables=[
        ControlConfig(
            name="price",
            pattern="linear_trend",
            base_value=10.0,
            value_volatility=1.0,
            value_trend=0.1
        ),
        ControlConfig(
            name="promotion",
            base_value=15.0,
            value_volatility=1.0,
            value_trend=0.0,
            seasonal_amplitude=0.05
        )
    ],
    seed=42
) 
