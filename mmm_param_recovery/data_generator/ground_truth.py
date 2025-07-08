"""
Ground truth calculation for MMM Dataset Generator.

This module contains functions for calculating ROAS and attribution metrics
for MMM datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from .config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


def calculate_roas_values(
    spend_data: pd.DataFrame,
    transformed_data: pd.DataFrame,
    config: MMMDataConfig
) -> Dict[str, Dict[str, float]]:
    """
    Calculate true ROAS values for each channel and region combination.
    
    Parameters
    ----------
    spend_data : pd.DataFrame
        Raw spend data with MultiIndex (date, geo)
    transformed_data : pd.DataFrame
        Transformed spend data with MultiIndex (date, geo)
    config : MMMDataConfig
        Configuration used for generation
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        ROAS values per region and channel
    """
    roas_values = {}
    
    for region_name in config.regions.region_names: # type: ignore
        region_roas = {}
        
        for channel_idx, channel in enumerate(config.channels):
            # Find the corresponding columns
            spend_col = None
            transformed_col = None
            
            # Look for spend column
            for col in spend_data.columns:
                if col.startswith(f'x{channel_idx + 1}'):
                    spend_col = col
                    break
            
            # Look for transformed column
            for col in transformed_data.columns:
                if col.startswith(f'contribution_x{channel_idx + 1}'):
                    transformed_col = col
                    break
            
            if spend_col and transformed_col:
                # Get spend and transformed data for this channel and region
                region_spend = spend_data[spend_col].xs(region_name, level='geo')
                region_transformed = transformed_data[transformed_col].xs(region_name, level='geo')
                
                # Calculate total spend and total contribution
                total_spend = region_spend.sum()
                total_contribution = region_transformed.sum()
                
                # Calculate ROAS (Return on Ad Spend)
                if total_spend > 0:
                    roas = total_contribution / total_spend
                else:
                    roas = 0.0
                
                region_roas[channel.name] = roas
        
        roas_values[region_name] = region_roas
    
    return roas_values


def calculate_attribution_percentages(
    transformed_data: pd.DataFrame,
    config: MMMDataConfig
) -> Dict[str, Dict[str, float]]:
    """
    Calculate attribution percentages for each channel and region.
    
    Parameters
    ----------
    transformed_data : pd.DataFrame
        Transformed spend data with MultiIndex (date, geo)
    config : MMMDataConfig
        Configuration used for generation
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Attribution percentages per region and channel
    """
    attribution_percentages = {}
    
    for region_name in config.regions.region_names: # type: ignore
        # Get data for this region
        region_data = transformed_data.xs(region_name, level='geo')
        
        # Find contribution columns for this region
        contribution_cols = [col for col in region_data.columns if col.startswith('contribution_')]
        
        if contribution_cols:
            # Calculate total contribution across all channels
            total_contribution = region_data[contribution_cols].sum().sum()
            
            region_attribution = {}
            for col in contribution_cols:
                # Extract channel name from column name
                # Format: contribution_x{i+1}_{channel_name} or contribution_x{i+1}
                channel_name = col.replace('contribution_', '')
                
                # Map to actual channel name if possible
                for channel_idx, channel in enumerate(config.channels):
                    expected_col = f'x{channel_idx + 1}_{channel.name}' if channel.name else f'x{channel_idx + 1}'
                    if channel_name == expected_col:
                        channel_name = channel.name
                        break
                
                channel_total_contribution = region_data[col].sum()
                
                if total_contribution > 0:
                    attribution_pct = (channel_total_contribution / total_contribution) * 100
                else:
                    attribution_pct = 0.0
                
                region_attribution[channel_name] = attribution_pct
            
            attribution_percentages[region_name] = region_attribution
    
    return attribution_percentages 