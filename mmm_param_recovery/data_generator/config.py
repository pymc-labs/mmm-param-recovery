"""
Configuration classes for MMM Dataset Generator.

This module defines the configuration dataclasses used to specify parameters
for data generation, including channel patterns, regional settings, and
transformation functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Callable
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
    start_period: Optional[int] = None  # When channel starts (None = immediate)
    ramp_up_periods: int = 4  # Number of periods to ramp up to full spend
    
    # On/off parameters
    activation_probability: float = 0.8  # Probability of being active in any period
    min_active_periods: int = 2  # Minimum consecutive active periods
    max_active_periods: int = 8  # Maximum consecutive active periods
    
    # Custom pattern function
    custom_pattern_func: Optional[Callable] = None
    
    # Effectiveness parameters
    base_effectiveness: float = 0.5  # Base effectiveness multiplier
    effectiveness_trend: float = 0.0  # Trend in effectiveness over time
    
    # Regional variation
    regional_effectiveness_variation: float = 0.2  # Std dev of regional effectiveness variation
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.base_spend < 0:
            raise ValueError("base_spend must be non-negative")
        if self.spend_volatility < 0:
            raise ValueError("spend_volatility must be non-negative")
        if not 0 <= self.seasonal_amplitude <= 1:
            raise ValueError("seasonal_amplitude must be between 0 and 1")
        if self.activation_probability < 0 or self.activation_probability > 1:
            raise ValueError("activation_probability must be between 0 and 1")
        if self.base_effectiveness < 0:
            raise ValueError("base_effectiveness must be non-negative")
        if self.regional_effectiveness_variation < 0:
            raise ValueError("regional_effectiveness_variation must be non-negative")
        if "_" in self.name:
            raise ValueError("channel name must not contain underscores (use hyphens instead)")


@dataclass
class TransformConfig:
    """Configuration for adstock and saturation transformations."""
    
    # Adstock parameters
    adstock_type: Literal["geometric", "weibull", "custom"] = "geometric"
    adstock_alpha: float = 0.5  # Retention rate for geometric adstock
    adstock_lam: float = 0.5  # Shape parameter for Weibull adstock
    adstock_k: float = 2.0  # Scale parameter for Weibull adstock
    
    # Saturation parameters
    saturation_type: Literal["hill", "logistic", "custom"] = "hill"
    saturation_ec50: float = 1000.0  # Half-saturation point
    saturation_slope: float = 1.0  # Slope parameter for Hill function
    saturation_max: float = 1.0  # Maximum saturation level
    
    # Custom transformation functions
    custom_adstock_func: Optional[Callable] = None
    custom_saturation_func: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate transformation parameters."""
        if not 0 <= self.adstock_alpha <= 1:
            raise ValueError("adstock_alpha must be between 0 and 1")
        if self.adstock_lam <= 0:
            raise ValueError("adstock_lam must be positive")
        if self.adstock_k <= 0:
            raise ValueError("adstock_k must be positive")
        if self.saturation_ec50 <= 0:
            raise ValueError("saturation_ec50 must be positive")
        if self.saturation_slope <= 0:
            raise ValueError("saturation_slope must be positive")
        if self.saturation_max <= 0:
            raise ValueError("saturation_max must be positive")


@dataclass
class RegionConfig:
    """Configuration for geographic regions."""
    
    n_regions: int = 1
    region_names: Optional[List[str]] = None
    
    # Baseline sales parameters
    base_sales_rate: float = 10000.0  # Base sales per period
    sales_trend: float = 0.02  # Linear trend in sales over time
    sales_volatility: float = 0.15  # Coefficient of variation for sales noise
    
    # Seasonal parameters
    seasonal_amplitude: float = 0.2  # Amplitude of seasonal variation in sales
    seasonal_phase: float = 0.0  # Phase shift in radians
    
    # Regional variation parameters
    regional_sales_variation: float = 0.3  # Std dev of regional sales rate variation
    regional_trend_variation: float = 0.01  # Std dev of regional trend variation
    
    # Region similarity controls
    region_correlation: float = 0.7  # Correlation between regions for similar behavior
    channel_effectiveness_correlation: float = 0.8  # Correlation of channel effectiveness across regions
    
    def __post_init__(self):
        """Validate region configuration."""
        if self.n_regions < 1 or self.n_regions > 50:
            raise ValueError("n_regions must be between 1 and 50")
        if self.base_sales_rate <= 0:
            raise ValueError("base_sales_rate must be positive")
        if self.sales_volatility < 0:
            raise ValueError("sales_volatility must be non-negative")
        if not 0 <= self.seasonal_amplitude <= 1:
            raise ValueError("seasonal_amplitude must be between 0 and 1")
        if self.regional_sales_variation < 0:
            raise ValueError("regional_sales_variation must be non-negative")
        if not -1 <= self.region_correlation <= 1:
            raise ValueError("region_correlation must be between -1 and 1")
        if not -1 <= self.channel_effectiveness_correlation <= 1:
            raise ValueError("channel_effectiveness_correlation must be between -1 and 1")
        
        # Generate region names if not provided
        if self.region_names is None:
            if self.n_regions == 1:
                self.region_names = ["All"]
            else:
                self.region_names = [f"Region_{i+1}" for i in range(self.n_regions)]
        elif len(self.region_names) != self.n_regions:
            raise ValueError("region_names length must match n_regions")


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
    control_variables: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
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
        for var_name, var_config in self.control_variables.items():
            if "base_value" not in var_config:
                raise ValueError(f"Control variable {var_name} must have 'base_value'")
            if "volatility" not in var_config:
                raise ValueError(f"Control variable {var_name} must have 'volatility'")


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
        region_names=["geo_a", "geo_b", "geo_c"]
    ),
    transforms=TransformConfig(
        adstock_alpha=0.6,
        saturation_ec50=2000.0
    ),
    control_variables={
        "price": {
            "base_value": 10.0,
            "volatility": 0.1,
            "trend": 0.02,
            "seasonal_amplitude": 0.05
        },
        "promotion": {
            "base_value": 0.2,
            "volatility": 0.05,
            "trend": 0.0,
            "seasonal_amplitude": 0.1
        }
    },
    seed=42
) 