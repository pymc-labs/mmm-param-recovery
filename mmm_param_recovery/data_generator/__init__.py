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
MMM Dataset Generator

A lightweight Python module for generating reproducible datasets for benchmarking 
Marketing Mix Models (MMM) with known ground truth parameters.

This module provides tools to create synthetic marketing data with configurable
channel patterns, geographic regions, adstock/saturation transformations, and
comprehensive ground truth information for model validation.
"""

__version__ = "0.1.0"
__author__ = "PyMC Labs"

# Core API exports
from .core import generate_mmm_dataset
from .config import (
    MMMDataConfig,
    ChannelConfig,
    RegionConfig,
    TransformConfig,
    DEFAULT_CONFIG
)
from .presets import get_preset_config, list_available_presets, customize_preset

# Utility functions
from .validation import validate_config, check_data_quality, validate_output_schema
from .utils import SeedManager, set_random_state, validate_seed

# Visualization functions
from .visualization import (
    plot_channel_spend,
    plot_channel_contributions,
    plot_roas_comparison,
    plot_regional_comparison,
    plot_data_quality
)

# Ground truth utilities
from .ground_truth import calculate_roas_values, calculate_attribution_percentages

__all__ = [
    # Main function
    "generate_mmm_dataset",
    
    # Configuration
    "MMMDataConfig",
    "ChannelConfig", 
    "RegionConfig",
    "TransformConfig",
    "DEFAULT_CONFIG",
    
    # Presets
    "get_preset_config",
    "list_available_presets",
    "customize_preset",
    
    # Validation
    "validate_config",
    "check_data_quality",
    "validate_output_schema",
    
    # Utilities
    "SeedManager",
    "set_random_state", 
    "validate_seed",
    
    # Visualization
    "plot_channel_spend",
    "plot_channel_contributions", 
    "plot_roas_comparison",
    "plot_regional_comparison",
    "plot_data_quality",
    
    # Ground truth
    "calculate_roas_values",
    "calculate_attribution_percentages",
] 