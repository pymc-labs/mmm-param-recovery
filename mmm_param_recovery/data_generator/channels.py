"""
Channel pattern generation for MMM Dataset Generator.

This module contains functions for generating different channel spend patterns.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from .config import ChannelConfig


def _generate_linear_trend_pattern(
    n_periods: int,
    base_spend: float,
    spend_trend: float,
    spend_volatility: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate linear trend channel pattern with noise.
    
    Parameters
    ----------
    n_periods : int
        Number of time periods
    base_spend : float
        Base spend level
    spend_trend : float
        Slope of the trend line (positive = increasing, negative = decreasing)
    spend_volatility : float
        Standard deviation of the noise relative to the base spend.
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Spend values for each time period
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate linear trend from base_spend to base_spend + trend
    trend_range = spend_trend * n_periods
    linear_trend = np.linspace(base_spend, base_spend + trend_range, n_periods)
    
    # Add noise with specified volatility
    noise_std = base_spend * spend_volatility
    noise = rng.normal(0, noise_std, n_periods)
    
    # Combine trend and noise, ensure non-negative values
    spend = np.clip(linear_trend + noise, 0, None)
    
    return spend.astype(float)


def _generate_seasonal_pattern(
    time_index: pd.DatetimeIndex,
    base_spend: float,
    seasonal_amplitude: float,
    seasonal_phase: float,
    spend_volatility: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate seasonal channel pattern with annual seasonality.
    
    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index for the data
    base_spend : float
        Base spend level
    seasonal_amplitude : float
        Amplitude of seasonal variation in relative to the base spend (0-1)
    seasonal_phase : float
        Phase shift in radians. With seasonal_phase = 0, peaks in summer, and drops in winter.
    spend_volatility : float
        Standard deviation of the noise relative to the base spend.
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Spend values for each time period
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    n_periods = len(time_index)
    
    # Generate seasonal pattern based on day of year
    days_in_year = 365.25
    day_of_year = time_index.dayofyear
    
    # Create seasonal pattern:
    #   - with seasonal_phase = 0, peaks in summer, and drops in winter
    seasonality = 1 - np.cos(seasonal_phase + 2 * np.pi * day_of_year / days_in_year)
    
    # Scale seasonality to base_spend range
    seasonal_spend = base_spend + base_spend * seasonal_amplitude * seasonality
    
    # Add noise with specified volatility
    noise_std = base_spend * spend_volatility
    noise = rng.normal(0, noise_std, n_periods)
    
    # Combine seasonal pattern and noise, ensure non-negative values
    spend = np.clip(seasonal_spend + noise, 0, None)
    
    return spend.astype(float)


def _generate_delayed_start_pattern(
    n_periods: int,
    base_spend: float,
    start_period: int,
    ramp_up_periods: int,
    spend_volatility: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate delayed start channel pattern with configurable start time.
    
    Parameters
    ----------
    n_periods : int
        Number of time periods
    base_spend : float
        Base spend level
    start_period : int
        Period when channel starts (0-indexed)
    ramp_up_periods : int
        Number of periods to ramp up to full spend
    spend_volatility : float
        Standard deviation of the noise relative to the base spend.
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Spend values for each time period
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    spend = np.zeros(n_periods, dtype=float)
    
    # Ramp up period
    if ramp_up_periods > 0:
        ramp_end = min(start_period + ramp_up_periods, n_periods, end_period)
        ramp_periods = ramp_end - start_period
        
        for i in range(ramp_periods):
            period = start_period + i
            # Linear ramp from 0 to base_spend
            ramp_factor = (i + 1) / ramp_periods
            spend[period] = base_spend * ramp_factor
    
    # Full spend after ramp up
    full_spend_start = start_period + ramp_up_periods
    if full_spend_start < min(n_periods, end_period):
        spend[full_spend_start:] = base_spend
    
    # Add noise with specified volatility (only to non-zero periods)
    noise_std = base_spend * spend_volatility
    noise = rng.normal(0, noise_std, n_periods)
    
    # Apply noise only to periods with spend > 0
    spend = np.where(spend > 0, spend + noise, spend)
    spend = np.clip(spend, 0, None)
    
    return spend.astype(float)


def _generate_on_off_pattern(
    n_periods: int,
    base_spend: float,
    activation_probability: float,
    min_active_periods: int,
    max_active_periods: int,
    spend_volatility: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate on/off channel pattern with random activation.
    
    Parameters
    ----------
    n_periods : int
        Number of time periods
    base_spend : float
        Base spend level when active
    activation_probability : float
        Probability of being active in any period
    min_active_periods : int
        Minimum consecutive active periods
    max_active_periods : int
        Maximum consecutive active periods
    spend_volatility : float
        Coefficient of variation for spend noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Spend values for each time period
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    spend = np.zeros(n_periods, dtype=float)
    current_period = 0
    
    while current_period < n_periods:
        # Decide if this should be an active period
        if rng.random() < activation_probability:
            # Determine how long this active period should last
            active_duration = int(rng.integers(min_active_periods, max_active_periods + 1))
            active_end = min(current_period + active_duration, n_periods)
            
            # Set spend for active periods
            spend[current_period:active_end] = base_spend
            # Ensure at least one period of inactivity between active periods
            current_period = active_end + 1
        else:
            # Inactive period
            current_period += 1
    
    # Add noise with specified volatility (only to active periods)
    noise_std = base_spend * spend_volatility
    noise = rng.normal(0, noise_std, n_periods)
    
    # Apply noise only to periods with spend > 0
    spend = np.where(spend > 0, spend + noise, spend)
    spend = np.clip(spend, 0, None)
    
    return spend.astype(float)


def generate_channel_spend(
    channel: ChannelConfig,
    time_index: pd.DatetimeIndex,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate spend pattern for a single channel.
    
    Parameters
    ----------
    channel : ChannelConfig
        Channel configuration
    time_index : pd.DatetimeIndex
        Time index for the data
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Spend values for each time period
    """
    
    n_periods = len(time_index)
    
    if channel.pattern == "linear_trend":
        return _generate_linear_trend_pattern(
            n_periods=n_periods,
            base_spend=channel.base_spend,
            spend_trend=channel.spend_trend,
            spend_volatility=channel.spend_volatility,
            seed=seed
        )
    
    elif channel.pattern == "seasonal":
        return _generate_seasonal_pattern(
            time_index=time_index,
            base_spend=channel.base_spend,
            seasonal_amplitude=channel.seasonal_amplitude,
            seasonal_phase=channel.seasonal_phase,
            spend_volatility=channel.spend_volatility,
            seed=seed
        )
    
    elif channel.pattern == "delayed_start":
        return _generate_delayed_start_pattern(
            n_periods=n_periods,
            base_spend=channel.base_spend,
            start_period=channel.start_period,
            ramp_up_periods=channel.ramp_up_periods,
            spend_volatility=channel.spend_volatility,
            seed=seed
        )
    
    elif channel.pattern == "on_off":
        return _generate_on_off_pattern(
            n_periods=n_periods,
            base_spend=channel.base_spend,
            activation_probability=channel.activation_probability,
            min_active_periods=channel.min_active_periods,
            max_active_periods=channel.max_active_periods,
            spend_volatility=channel.spend_volatility,
            seed=seed
        )
    
    elif channel.pattern == "custom":
        if channel.custom_pattern_func is None:
            raise ValueError("custom_pattern_func must be provided for custom pattern")
        return channel.custom_pattern_func(channel, time_index, seed)
    
    else:
        raise ValueError(f"Unknown channel pattern: {channel.pattern}") 