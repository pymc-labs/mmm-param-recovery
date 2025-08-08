"""
MMM Parameter Recovery Package

A package for generating synthetic MMM (Media Mix Modeling) datasets
for parameter recovery studies.
"""

from .data_generator import generate_mmm_dataset, DEFAULT_CONFIG
from .data_generator.config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig

__version__ = "0.1.0"
__all__ = [
    "generate_mmm_dataset",
    "DEFAULT_CONFIG", 
    "MMMDataConfig",
    "ChannelConfig",
    "RegionConfig",
    "TransformConfig"
] 