"""
Adstock and saturation transformations for MMM Dataset Generator.

This module contains functions for applying adstock and saturation transformations
using PyMC Marketing's transformers module.
"""

import numpy as np
from typing import Optional, Callable, Any
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
    channel_config: Optional[ChannelConfig] = None,
    channel_idx: int = 0
) -> np.ndarray:
    """
    Apply adstock and saturation transformations to spend data.
    
    Parameters
    ----------
    spend_data : np.ndarray
        Raw spend data
    transforms : TransformConfig
        Transformation configuration
    channel_config : ChannelConfig, optional
        Channel configuration for channel-specific transformations
    channel_idx : int, optional
        Channel index for per-channel parameter selection
        
    Returns
    -------
    np.ndarray
        Transformed spend data
    """
    # Start with original data
    transformed_data = spend_data.copy()
    
    # Apply adstock transformation
    if isinstance(transforms.adstock_fun, list):
        adstock_fun = transforms.adstock_fun[channel_idx % len(transforms.adstock_fun)]
    else:
        adstock_fun = transforms.adstock_fun

    if adstock_fun and adstock_fun != "none":
        transformed_data = apply_transform(
            transformed_data,
            getattr(transformers, adstock_fun),
            **transforms.get_adstock_kwargs(channel_idx) # type: ignore
        )

    
    # Apply saturation transformation
    if isinstance(transforms.saturation_fun, list):
        saturation_fun = transforms.saturation_fun[channel_idx % len(transforms.saturation_fun)]
    else:
        saturation_fun = transforms.saturation_fun

    if saturation_fun and saturation_fun != "none":
        transformed_data = apply_transform(
            transformed_data,
            getattr(transformers, saturation_fun),
            **transforms.get_saturation_kwargs(channel_idx) # type: ignore
        )

    
    return transformed_data 