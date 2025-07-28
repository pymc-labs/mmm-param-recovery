"""
Visualization functions for MMM Dataset Generator.

This module contains plotting functions for data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import warnings
from matplotlib.figure import Figure

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def plot_channel_spend(
    data: pd.DataFrame, 
    channels: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    **kwargs
) -> Figure:
    """
    Plot channel spend patterns over time.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing spend data with columns: date, geo, x1-*, x2-*, etc.
    channels : List[str], optional
        List of channel names to plot. If None, plots all channels.
    regions : List[str], optional
        List of regions to plot. If None, plots all regions.
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    required_cols = ['date', 'geo']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get channel columns
    channel_cols = [col for col in data.columns if col.startswith('x') and col[1:].split('_')[0].isdigit()]
    if not channel_cols:
        raise ValueError("No channel columns (x1, x2, etc.) found in data")
    
    # Filter channels if specified
    if channels is not None:
        channel_cols = [col for col in channel_cols if any(ch in col for ch in channels)]
        if not channel_cols:
            raise ValueError("No matching channel columns found")
    
    # Filter regions if specified
    plot_data = data.copy()
    if regions is not None:
        plot_data = plot_data[plot_data['geo'].isin(regions)]
        if len(plot_data) == 0:
            raise ValueError("No data found for specified regions")
    
    # Ensure date column is datetime
    plot_data['date'] = pd.to_datetime(plot_data['date'])
    plot_data = plot_data.sort_values(['date', 'geo'])
    
        # Create figure
    fig, axes = plt.subplots(
        nrows=len(channel_cols), 
        ncols=len(plot_data['geo'].unique()), 
        figsize=figsize,
        sharex=True,
        sharey='row'  # Share y-axis within each row
    )
    
    # Handle single channel or single region case
    if len(channel_cols) == 1:
        axes = axes.reshape(1, -1)
    if len(plot_data['geo'].unique()) == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each channel
    for i, channel_col in enumerate(channel_cols):
        
        # Get channel name for title
        channel_name = channel_col.split('_', 1)[1] if '_' in channel_col else channel_col
        
        # Plot each region separately
        for j, region in enumerate(plot_data['geo'].unique()):
            ax = axes[i, j]
            region_data = plot_data[plot_data['geo'] == region]
            region_data = region_data.sort_values('date')
                
            ax.plot(region_data['date'], region_data[channel_col], 
                   linewidth=2, alpha=0.8, **kwargs)
            
            # Set column title (region name) only for first row
            if i == 0:
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')
            # Set y-axis label only for first column (leftmost)
            if j == 0:
                ax.set_ylabel(f'{channel_name}', fontsize=12)
            else:
                ax.set_ylabel('')  # Remove y-axis label for other columns
            
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            if i == len(channel_cols) - 1:  # Last row
                ax.set_xlabel('Date', fontsize=10)
                # Rotate x-axis labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_channel_contributions(
    data: pd.DataFrame, 
    channels: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    **kwargs
) -> Figure:
    """
    Plot channel contributions over time.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing contribution data with columns: date, geo, contribution_x1-*, etc.
    channels : List[str], optional
        List of channel names to plot. If None, plots all channels.
    regions : List[str], optional
        List of regions to plot. If None, plots all regions.
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    required_cols = ['date', 'geo']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get contribution columns
    contribution_cols = [col for col in data.columns if col.startswith('contribution_')]
    if not contribution_cols:
        raise ValueError("No contribution columns (contribution_x1, contribution_x2, etc.) found in data")
    
    # Filter channels if specified
    if channels is not None:
        contribution_cols = [col for col in contribution_cols if any(ch in col for ch in channels)]
        if not contribution_cols:
            raise ValueError("No matching contribution columns found")
    
    # Filter regions if specified
    plot_data = data.copy()
    if regions is not None:
        plot_data = plot_data[plot_data['geo'].isin(regions)]
        if len(plot_data) == 0:
            raise ValueError("No data found for specified regions")
    
    # Ensure date column is datetime
    plot_data['date'] = pd.to_datetime(plot_data['date'])
    plot_data = plot_data.sort_values(['date', 'geo'])
    
    # Create figure
    n_rows = len(contribution_cols)
    n_cols = len(plot_data['geo'].unique())
    
    fig, axes = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=figsize,
        sharex=True,
        sharey='row'  # Share y-axis within each row
    )
    
    # Handle single row or single column case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each channel contribution
    for i, contribution_col in enumerate(contribution_cols):
        # Get channel name for title
        channel_name = contribution_col.replace('contribution_', '')
        if '_' in channel_name:
            channel_name = channel_name.split('_', 1)[1]
        
        # Plot each region separately
        for j, region in enumerate(plot_data['geo'].unique()):
            ax = axes[i, j]
            region_data = plot_data[plot_data['geo'] == region]
            region_data = region_data.sort_values('date')
            
            ax.plot(region_data['date'], region_data[contribution_col], 
                   linewidth=2, alpha=0.8, label=f'{channel_name}', **kwargs)
            
            # Set column title (region name) only for first row
            if i == 0:
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')
            
            # Set y-axis label only for first column (leftmost)
            if j == 0:
                ax.set_ylabel(f'{channel_name} Contribution', fontsize=12)
            else:
                ax.set_ylabel('')
            
            ax.grid(True, alpha=0.3)
            
            # Format x-axis for last row
            if i == len(contribution_cols) - 1:
                ax.set_xlabel('Date', fontsize=10)
                # Rotate x-axis labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_roas_comparison(ground_truth: Dict[str, Any], figsize: tuple = (12, 8), **kwargs) -> Figure:
    """
    Plot ROAS comparison across channels and regions.
    
    Parameters
    ----------
    ground_truth : Dict[str, Any]
        Ground truth data containing ROAS information in the format:
        {'roas_values': {region: {channel: roas_value}}}
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Validate input data
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must be a dictionary")
    
    if 'roas_values' not in ground_truth:
        raise ValueError("ground_truth must contain 'roas_values' key")
    
    roas_values = ground_truth['roas_values']
    if not isinstance(roas_values, dict):
        raise ValueError("roas_values must be a dictionary")
    
    if not roas_values:
        raise ValueError("roas_values dictionary is empty")
    
    # Get all regions and channels
    regions = list(roas_values.keys())
    channels = set()
    for region_data in roas_values.values():
        if isinstance(region_data, dict):
            channels.update(region_data.keys())
    channels = sorted(list(channels))
    
    if not channels:
        raise ValueError("No channels found in ROAS data")
    
    # Create figure with subplots
    n_regions = len(regions)
    n_channels = len(channels)
    
    # Determine subplot layout
    if n_regions == 1:
        # Single region: horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        x_pos = np.arange(len(channels))
        roas_data = [roas_values[regions[0]].get(channel, 0) for channel in channels]
        
        bars = ax.bar(x_pos, roas_data, alpha=0.8, color='steelblue')
        
        # Add value labels on bars
        for bar, value in zip(bars, roas_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('ROAS', fontsize=12)
        ax.set_title(f'ROAS Comparison - {regions[0]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(channels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Multiple regions: subplot grid
        fig, axes = plt.subplots(
            nrows=1, ncols=n_regions, 
            figsize=figsize,
            sharey=True
        )
        
        # Handle single subplot case
        if n_regions == 1:
            axes = [axes]
        
        # Plot each region
        for i, region in enumerate(regions):
            ax = axes[i]
            region_roas = roas_values[region]
            
            if isinstance(region_roas, dict):
                # Create bar chart for this region
                x_pos = np.arange(len(channels))
                roas_data = [region_roas.get(channel, 0) for channel in channels]
                
                bars = ax.bar(x_pos, roas_data, alpha=0.8, color=f'C{i}')
                
                # Add value labels on bars
                for bar, value in zip(bars, roas_data):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_xlabel('Channel', fontsize=10)
                ax.set_ylabel('ROAS' if i == 0 else '', fontsize=10)
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(channels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle('ROAS Comparison Across Regions', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_regional_comparison(
    data: pd.DataFrame, 
    metric: str = 'y',
    channels: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    **kwargs
) -> Figure:
    """
    Plot regional comparison visualizations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing regional data with columns: date, geo, y, x1-*, x2-*, etc.
    metric : str, default 'y'
        Metric to compare across regions ('y' for sales, or channel name)
    channels : List[str], optional
        List of channels to include in comparison. If None, uses all channels.
    figsize : tuple, default (15, 10)
        Figure size (width, height)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    required_cols = ['date', 'geo']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure date column is datetime
    plot_data = data.copy()
    plot_data['date'] = pd.to_datetime(plot_data['date'])
    plot_data = plot_data.sort_values(['date', 'geo'])
    
    # Get regions
    regions = plot_data['geo'].unique()
    if len(regions) < 2:
        raise ValueError("Need at least 2 regions for regional comparison")
    
    # Determine what to plot
    if metric == 'y':
        # Plot sales comparison
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Sales over time by region
        ax1 = axes[0]
        for region in regions:
            region_data = plot_data[plot_data['geo'] == region]
            ax1.plot(region_data['date'], region_data['y'], 
                    linewidth=2, alpha=0.8, label=region)
        ax1.set_title('Sales Over Time by Region', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sales distribution by region
        ax2 = axes[1]
        sales_by_region = [plot_data[plot_data['geo'] == region]['y'].values for region in regions]
        ax2.boxplot(sales_by_region, tick_labels=regions)
        ax2.set_title('Sales Distribution by Region', fontweight='bold')
        ax2.set_ylabel('Sales')
        ax2.grid(True, alpha=0.3)
        
        # 3. Average sales by region
        ax3 = axes[2]
        avg_sales = [plot_data[plot_data['geo'] == region]['y'].mean() for region in regions]
        bars = ax3.bar(regions, avg_sales, alpha=0.8, color='steelblue')
        ax3.set_title('Average Sales by Region', fontweight='bold')
        ax3.set_ylabel('Average Sales')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_sales):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Sales volatility by region
        ax4 = axes[3]
        sales_volatility = [plot_data[plot_data['geo'] == region]['y'].std() for region in regions]
        bars = ax4.bar(regions, sales_volatility, alpha=0.8, color='lightcoral')
        ax4.set_title('Sales Volatility by Region', fontweight='bold')
        ax4.set_ylabel('Sales Standard Deviation')
        
        # Add value labels on bars
        for bar, value in zip(bars, sales_volatility):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
    else:
        # Plot channel comparison
        # Get channel columns
        channel_cols = [col for col in plot_data.columns if col.startswith('x') and col[1:].split('_')[0].isdigit()]
        
        if not channel_cols:
            raise ValueError("No channel columns found in data")
        
        # Filter channels if specified
        if channels is not None:
            channel_cols = [col for col in channel_cols if any(ch in col for ch in channels)]
            if not channel_cols:
                raise ValueError("No matching channel columns found")
        
        # Create subplots for each channel
        n_channels = len(channel_cols)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_channels == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        # Plot each channel
        for i, channel_col in enumerate(channel_cols):
            ax = axes[i]
            channel_name = channel_col.split('_', 1)[1] if '_' in channel_col else channel_col
            
            # Plot spend over time by region
            for region in regions:
                region_data = plot_data[plot_data['geo'] == region]
                ax.plot(region_data['date'], region_data[channel_col], 
                       linewidth=2, alpha=0.8, label=region)
            
            ax.set_title(f'{channel_name} Spend by Region', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Spend')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_data_quality(data: pd.DataFrame, figsize: tuple = (15, 12), **kwargs) -> Figure:
    """
    Plot data quality validation diagnostics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to validate with columns: date, geo, y, x1-*, x2-*, etc.
    figsize : tuple, default (15, 12)
        Figure size (width, height)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    required_cols = ['date', 'geo', 'y']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure date column is datetime
    plot_data = data.copy()
    plot_data['date'] = pd.to_datetime(plot_data['date'])
    
    # Get channel columns
    channel_cols = [col for col in plot_data.columns if col.startswith('x') and col[1:].split('_')[0].isdigit()]
    
    # Create comprehensive data quality dashboard
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sales distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(plot_data['y'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_title('Sales Distribution', fontweight='bold')
    ax1.set_xlabel('Sales')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Sales over time
    ax2 = fig.add_subplot(gs[0, 1])
    for region in plot_data['geo'].unique():
        region_data = plot_data[plot_data['geo'] == region]
        ax2.plot(region_data['date'], region_data['y'], 
                linewidth=1, alpha=0.7, label=region)
    ax2.set_title('Sales Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Missing values heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    missing_data = plot_data.isnull().sum()
    if missing_data.sum() > 0:
        # Create heatmap of missing values
        missing_matrix = plot_data.isnull().astype(int)
        im = ax3.imshow(missing_matrix.T, cmap='Reds', aspect='auto')
        ax3.set_title('Missing Values Heatmap', fontweight='bold')
        ax3.set_xlabel('Row Index')
        ax3.set_ylabel('Column')
        plt.colorbar(im, ax=ax3, label='Missing (1) / Present (0)')
    else:
        ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14, fontweight='bold')
        ax3.set_title('Missing Values Check', fontweight='bold')
    
    # 4. Channel spend distributions
    ax4 = fig.add_subplot(gs[1, 0])
    if channel_cols:
        # Plot boxplot of channel spend distributions
        spend_data = [plot_data[col] for col in channel_cols]
        channel_names = [col.split('_', 1)[1] if '_' in col else col for col in channel_cols]
        bp = ax4.boxplot(spend_data)
        ax4.set_title('Channel Spend Distributions', fontweight='bold')
        ax4.set_ylabel('Spend')
        ax4.set_xticklabels(channel_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Channel Data', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.set_title('Channel Spend Distributions', fontweight='bold')
    
    # 5. Correlation heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    # Select numeric columns for correlation
    numeric_cols = plot_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = plot_data[numeric_cols].corr()
        im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax5.set_title('Correlation Matrix', fontweight='bold')
        ax5.set_xticks(range(len(numeric_cols)))
        ax5.set_yticks(range(len(numeric_cols)))
        ax5.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax5.set_yticklabels(numeric_cols)
        plt.colorbar(im, ax=ax5, label='Correlation')
    else:
        ax5.text(0.5, 0.5, 'Insufficient Data\nfor Correlation', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=14, fontweight='bold')
        ax5.set_title('Correlation Matrix', fontweight='bold')
    
    # 6. Data summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    Dataset Summary:
    
    Shape: {plot_data.shape}
    Regions: {len(plot_data['geo'].unique())}
    Time Period: {plot_data['date'].min().strftime('%Y-%m-%d')} to {plot_data['date'].max().strftime('%Y-%m-%d')}
    Channels: {len(channel_cols)}
    
    Sales Statistics:
    Mean: {plot_data['y'].mean():.2f}
    Std: {plot_data['y'].std():.2f}
    Min: {plot_data['y'].min():.2f}
    Max: {plot_data['y'].max():.2f}
    
    Missing Values: {plot_data.isnull().sum().sum()}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 7. Regional comparison
    ax7 = fig.add_subplot(gs[2, 0])
    regions = plot_data['geo'].unique()
    if len(regions) > 1:
        region_stats = []
        region_names = []
        for region in regions:
            region_data = plot_data[plot_data['geo'] == region]
            region_stats.append([
                region_data['y'].mean(),
                region_data['y'].std(),
                region_data['y'].min(),
                region_data['y'].max()
            ])
            region_names.append(region)
        
        # Create grouped bar chart
        x = np.arange(len(region_names))
        width = 0.2
        
        ax7.bar(x - width*1.5, [stats[0] for stats in region_stats], width, label='Mean', alpha=0.8)
        ax7.bar(x - width*0.5, [stats[1] for stats in region_stats], width, label='Std', alpha=0.8)
        ax7.bar(x + width*0.5, [stats[2] for stats in region_stats], width, label='Min', alpha=0.8)
        ax7.bar(x + width*1.5, [stats[3] for stats in region_stats], width, label='Max', alpha=0.8)
        
        ax7.set_title('Regional Sales Statistics', fontweight='bold')
        ax7.set_xlabel('Region')
        ax7.set_ylabel('Sales')
        ax7.set_xticks(x)
        ax7.set_xticklabels(region_names)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Single Region\nData', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=14, fontweight='bold')
        ax7.set_title('Regional Sales Statistics', fontweight='bold')
    
    # 8. Time series decomposition (if enough data)
    ax8 = fig.add_subplot(gs[2, 1])
    if len(plot_data) > 12:  # Need enough data for decomposition
        # Simple trend analysis
        plot_data_sorted = plot_data.sort_values('date')
        x_trend = np.arange(len(plot_data_sorted))
        z = np.polyfit(x_trend, plot_data_sorted['y'], 1)
        p = np.poly1d(z)
        
        ax8.plot(plot_data_sorted['date'], plot_data_sorted['y'], 
                linewidth=1, alpha=0.7, label='Actual')
        ax8.plot(plot_data_sorted['date'], p(x_trend), 
                linewidth=2, color='red', label='Trend')
        ax8.set_title('Sales Trend Analysis', fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Sales')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'Insufficient Data\nfor Trend Analysis', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=14, fontweight='bold')
        ax8.set_title('Sales Trend Analysis', fontweight='bold')
    
    # 9. Data quality score
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate quality metrics
    missing_pct = (plot_data.isnull().sum().sum() / (plot_data.shape[0] * plot_data.shape[1])) * 100
    negative_sales = (plot_data['y'] < 0).sum()
    zero_sales = (plot_data['y'] == 0).sum()
    
    quality_score = 100 - missing_pct - (negative_sales / len(plot_data)) * 50
    quality_score = max(0, min(100, quality_score))
    
    # Create quality report
    quality_text = f"""
    Data Quality Report:
    
    Overall Score: {quality_score:.1f}/100
    
    Issues Found:
    • Missing Values: {missing_pct:.1f}%
    • Negative Sales: {negative_sales}
    • Zero Sales: {zero_sales}
    
    Recommendations:
    {'✓ Data looks good!' if quality_score > 80 else '⚠ Check for data issues'}
    """
    
    ax9.text(0.05, 0.95, quality_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    return fig 