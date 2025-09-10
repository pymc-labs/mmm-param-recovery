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
Adstock and saturation transformations for MMM Dataset Generator.

This module contains functions for applying adstock and saturation transformations
using PyMC Marketing's transformers module.
"""

import numpy as np
from typing import Optional, Callable, Any, Tuple, Dict
from pytensor.tensor.variable import TensorVariable
from pymc_marketing.mmm import transformers
from .config import TransformConfig, ChannelConfig


def apply_transform(
    spend_data: np.ndarray,
    transform_func: Callable[..., TensorVariable],
    **kwargs: Any
) -> np.ndarray:
    """
    Apply a PyMC Marketing transformation function to spend data.
    
    Parameters
    ----------
    spend_data : np.ndarray
        Raw spend data
    transform_func : Callable
        PyMC Marketing transformation function (e.g., geometric_adstock, hill_function)
    **kwargs
        Keyword arguments specific to the transformation function
        
    Returns
    -------
    np.ndarray
        Transformed spend data.
    """
    return transform_func(spend_data, **kwargs).eval().flatten()


def apply_transformations(
    spend_data: np.ndarray,
    transforms: TransformConfig,
    channel_idx: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply adstock and saturation transformations to spend data.
    
    Parameters
    ----------
    spend_data : np.ndarray
        Raw spend data
    transforms : TransformConfig
        Transformation configuration
    channel_idx : int, optional
        Channel index for per-channel parameter selection
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Transformed spend data and transformation parameters used
    """
    # Start with original data
    transformed_data = spend_data.copy()
    transform_params = {
        'adstock_function': None,
        'adstock_params': {},
        'saturation_function': None,
        'saturation_params': {}
    }
    
    # Apply adstock transformation
    if isinstance(transforms.adstock_fun, list):
        adstock_fun = transforms.adstock_fun[channel_idx % len(transforms.adstock_fun)]
    else:
        adstock_fun = transforms.adstock_fun

    if adstock_fun and adstock_fun != "none":
        adstock_kwargs = transforms.get_adstock_kwargs(channel_idx)
        transformed_data = apply_transform(
            transformed_data,
            getattr(transformers, adstock_fun),
            **adstock_kwargs
        )
        transform_params['adstock_function'] = adstock_fun
        transform_params['adstock_params'] = adstock_kwargs

    # Apply saturation transformation
    if isinstance(transforms.saturation_fun, list):
        saturation_fun = transforms.saturation_fun[channel_idx % len(transforms.saturation_fun)]
    else:
        saturation_fun = transforms.saturation_fun

    if saturation_fun and saturation_fun != "none":
        saturation_kwargs = transforms.get_saturation_kwargs(channel_idx)
        transformed_data = apply_transform(
            transformed_data,
            getattr(transformers, saturation_fun),
            **saturation_kwargs
        )
        transform_params['saturation_function'] = saturation_fun
        transform_params['saturation_params'] = saturation_kwargs

    return transformed_data, transform_params 