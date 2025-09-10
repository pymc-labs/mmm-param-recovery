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