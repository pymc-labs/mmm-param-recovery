"""Visualization functions for model results and comparisons."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import arviz as az
from meridian.analysis import visualizer
from . import storage


def setup_plot_style() -> None:
    """Set up consistent plot styling."""
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 8


def plot_meridian_posterior_predictive(
    meridian_model: Any,
    data_df: pd.DataFrame,
    dataset_name: str,
    save: bool = True
) -> None:
    """Plot Meridian posterior predictive.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    data_df : pd.DataFrame
        Original dataset
    dataset_name : str
        Name of the dataset
    save : bool
        Whether to save the plot
    """
    from meridian.analysis import analyzer
    
    # Extract geo and time information
    geos = data_df["geo"].unique()
    dates = data_df["time"].unique()
    
    # Get Meridian Analyzer
    meridian_analyzer = analyzer.Analyzer(meridian_model)
    
    # Get expected outcomes (predictions) using the Analyzer
    # This returns shape: (n_chains, n_draws, n_geos, n_times)
    expected_outcomes = meridian_analyzer.expected_outcome(
        use_posterior=True,
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=True,
        batch_size=100
    )
    
    # Convert to numpy array and reshape for easier handling
    # Shape: (n_chains * n_draws, n_geos, n_times)
    predictions = expected_outcomes.numpy()
    n_chains, n_draws, n_geos_pred, n_times_pred = predictions.shape
    predictions = predictions.reshape(-1, n_geos_pred, n_times_pred)
    
    # Create matplotlib figure with improved aspect ratio
    n_geos = len(geos)
    fig, axes = plt.subplots(
        ncols=n_geos,
        figsize=(8 * n_geos, 5),  # Adjusted width for better aspect ratio
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    
    if n_geos == 1:
        axes = [axes]
    
    for j, geo in enumerate(geos):
        ax = axes[j]
        geo_data = data_df[data_df["geo"] == geo].copy()
        
        # Sort by time to ensure proper ordering
        geo_data = geo_data.sort_values("time")
        
        # Get predictions for this geo - ensure we have right number of time points
        geo_predictions = predictions[:, j, :]  # Shape: (samples, n_times_pred)
        
        # Match the number of time points between predictions and data
        n_times_data = len(geo_data)
        if n_times_pred != n_times_data:
            # If mismatch, take the minimum and align
            n_times_use = min(n_times_pred, n_times_data)
            geo_predictions = geo_predictions[:, :n_times_use]
            geo_data = geo_data.iloc[:n_times_use]
            print(f"  Warning: Time dimension mismatch. Using {n_times_use} time points.")
        
        # Calculate HDI intervals - arviz needs shape (time, samples)
        # geo_predictions is (samples, time), so we transpose to (time, samples)
        # But we need to ensure arviz interprets it correctly
        hdi_94 = np.zeros((len(geo_data), 2))
        hdi_50 = np.zeros((len(geo_data), 2))
        
        for t in range(len(geo_data)):
            time_samples = geo_predictions[:, t]  # Get all samples for this time point
            hdi_94[t] = az.hdi(time_samples, hdi_prob=0.94)
            hdi_50[t] = az.hdi(time_samples, hdi_prob=0.50)
        
        # Plot HDI intervals
        time_points = range(len(geo_data))
        ax.fill_between(
            time_points,
            hdi_94[:, 0],
            hdi_94[:, 1],
            alpha=0.2,
            color="C0",
            label="94% HDI"
        )
        ax.fill_between(
            time_points,
            hdi_50[:, 0],
            hdi_50[:, 1],
            alpha=0.4,
            color="C0",
            label="50% HDI"
        )
        
        # Plot median prediction
        median_pred = np.median(geo_predictions, axis=0)
        ax.plot(time_points, median_pred, color="C0", linewidth=2, alpha=0.8)
        
        # Plot actual data
        ax.plot(
            time_points,
            geo_data["y"].values,
            color="black",
            linewidth=1.5,
            label="Actual"
        )
        
        ax.set_title(f"{geo}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if j == 0:
            ax.set_ylabel("KPI Value")
        ax.set_xlabel("Time Period")
        
        # Set x-axis labels (show every nth date)
        n_ticks = 6
        tick_indices = np.linspace(0, len(geo_data) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        time_values = geo_data["time"].values
        ax.set_xticklabels([str(time_values[int(i)])[:10] for i in tick_indices], rotation=45, ha='right')
    
    fig.suptitle(
        f"{dataset_name.replace('_', ' ').title()} – Meridian",
        fontsize=16,
        fontweight="bold",
        y=1.03
    )
    
    if save:
        plot_path = storage.get_plot_path(dataset_name, "posterior_predictive_meridian.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved Meridian posterior predictive plot to {plot_path}")
    
    plt.close('all')


def plot_pymc_posterior_predictive(
    pymc_model: Any,
    data_df: pd.DataFrame,
    dataset_name: str,
    sampler: str,
    save: bool = True
) -> None:
    """Plot PyMC-Marketing posterior predictive.
    
    Parameters
    ----------
    pymc_model : MMM
        Fitted PyMC-Marketing model
    data_df : pd.DataFrame
        Original dataset
    dataset_name : str
        Name of the dataset
    sampler : str
        Sampler name
    save : bool
        Whether to save the plot
    """
    geos = pymc_model.model.coords["geo"]
    dates = pymc_model.model.coords["date"]
    
    fig, axes = plt.subplots(
        ncols=len(geos),
        figsize=(8 * len(geos), 5),  # Adjusted width for better aspect ratio
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    
    if len(geos) == 1:
        axes = [axes]
    
    for j, geo in enumerate(geos):
        ax = axes[j]
        
        az.plot_hdi(
            x=dates,
            y=pymc_model.idata["posterior_predictive"].y_original_scale.sel(geo=geo),
            color="C0",
            smooth=False,
            hdi_prob=0.94,
            fill_kwargs={"alpha": 0.2, "label": "94% HDI"},
            ax=ax,
        )
        
        az.plot_hdi(
            x=dates,
            y=pymc_model.idata["posterior_predictive"].y_original_scale.sel(geo=geo),
            color="C0",
            smooth=False,
            hdi_prob=0.5,
            fill_kwargs={"alpha": 0.4, "label": "50% HDI"},
            ax=ax,
        )
        
        sns.lineplot(
            data=data_df.query("geo == @geo"),
            x="time",
            y="y",
            color="black",
            ax=ax,
        )
        
        ax.legend(loc="upper left")
        ax.set(title=f"{geo}")
    
    fig.suptitle(
        f"{dataset_name.replace('_', ' ').title()} – PyMC-Marketing ({sampler})",
        fontsize=16,
        fontweight="bold",
        y=1.03
    )
    
    if save:
        plot_path = storage.get_plot_path(dataset_name, f"posterior_predictive_pymc_{sampler}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved PyMC-Marketing ({sampler}) posterior predictive plot to {plot_path}")
    
    plt.close('all')










def plot_model_comparison(
    meridian_model: Any,
    pymc_model: Any,
    data_df: pd.DataFrame,
    dataset_name: str,
    save: bool = True
) -> None:
    """Plot combined comparison of Meridian and PyMC-Marketing predictions.
    
    Shows actual data, Meridian predictions with confidence intervals,
    and PyMC-Marketing (nutpie) predictions with confidence intervals
    all on the same plot for easy comparison.
    
    Parameters
    ----------
    meridian_model : meridian.model.model.Meridian
        Fitted Meridian model
    pymc_model : MMM
        Fitted PyMC-Marketing model (preferably nutpie sampler)
    data_df : pd.DataFrame
        Original dataset
    dataset_name : str
        Name of the dataset
    save : bool
        Whether to save the plot
    """
    from meridian.analysis import analyzer
    
    # Extract geo and time information
    geos = data_df["geo"].unique()
    dates = data_df["time"].unique()
    
    # Get Meridian predictions
    meridian_analyzer = analyzer.Analyzer(meridian_model)
    meridian_outcomes = meridian_analyzer.expected_outcome(
        use_posterior=True,
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=True,
        batch_size=100
    )
    
    # Process Meridian predictions
    meridian_preds = meridian_outcomes.numpy()
    n_chains, n_draws, n_geos_pred, n_times_pred = meridian_preds.shape
    meridian_preds = meridian_preds.reshape(-1, n_geos_pred, n_times_pred)
    
    # Get PyMC predictions from posterior predictive
    pymc_posterior_pred = pymc_model.idata["posterior_predictive"].y_original_scale
    
    # Create figure with improved layout
    n_geos = len(geos)
    fig, axes = plt.subplots(
        ncols=n_geos,
        figsize=(10 * n_geos, 6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    
    if n_geos == 1:
        axes = [axes]
    
    for j, geo in enumerate(geos):
        ax = axes[j]
        geo_data = data_df[data_df["geo"] == geo].copy()
        geo_data = geo_data.sort_values("time")
        
        # Ensure consistent time points
        n_times_data = len(geo_data)
        
        # --- Meridian Predictions ---
        meridian_geo_preds = meridian_preds[:, j, :]
        if n_times_pred != n_times_data:
            n_times_use = min(n_times_pred, n_times_data)
            meridian_geo_preds = meridian_geo_preds[:, :n_times_use]
            geo_data_aligned = geo_data.iloc[:n_times_use]
        else:
            geo_data_aligned = geo_data
        
        # Calculate Meridian HDI
        meridian_hdi_94 = np.zeros((len(geo_data_aligned), 2))
        meridian_hdi_50 = np.zeros((len(geo_data_aligned), 2))
        
        for t in range(len(geo_data_aligned)):
            time_samples = meridian_geo_preds[:, t]
            meridian_hdi_94[t] = az.hdi(time_samples, hdi_prob=0.94)
            meridian_hdi_50[t] = az.hdi(time_samples, hdi_prob=0.50)
        
        # --- PyMC Predictions ---
        pymc_geo_preds = pymc_posterior_pred.sel(geo=geo)
        
        # Calculate PyMC HDI - returns Dataset with 'y_original_scale' variable
        pymc_hdi_94_ds = az.hdi(pymc_geo_preds, hdi_prob=0.94)
        pymc_hdi_50_ds = az.hdi(pymc_geo_preds, hdi_prob=0.50)
        
        # Extract the HDI values as arrays
        pymc_hdi_94 = pymc_hdi_94_ds["y_original_scale"].values
        pymc_hdi_50 = pymc_hdi_50_ds["y_original_scale"].values
        
        # Time points for plotting
        time_points = range(len(geo_data_aligned))
        
        # Plot actual data (black line)
        ax.plot(
            time_points,
            geo_data_aligned["y"].values,
            color="black",
            linewidth=2,
            label="Actual",
            zorder=5
        )
        
        # Plot Meridian predictions (blue)
        ax.fill_between(
            time_points,
            meridian_hdi_94[:, 0],
            meridian_hdi_94[:, 1],
            alpha=0.15,
            color="C0",
            label="Meridian 94% HDI"
        )
        ax.fill_between(
            time_points,
            meridian_hdi_50[:, 0],
            meridian_hdi_50[:, 1],
            alpha=0.25,
            color="C0",
            label="Meridian 50% HDI"
        )
        meridian_median = np.median(meridian_geo_preds, axis=0)
        ax.plot(
            time_points,
            meridian_median,
            color="C0",
            linewidth=1.5,
            label="Meridian median",
            linestyle="--",
            alpha=0.8
        )
        
        # Plot PyMC predictions (orange)
        # Ensure PyMC predictions align with time points
        pymc_n_times = len(pymc_hdi_94)
        if pymc_n_times != len(geo_data_aligned):
            pymc_time_points = range(min(pymc_n_times, len(geo_data_aligned)))
            pymc_hdi_94 = pymc_hdi_94[:len(pymc_time_points)]
            pymc_hdi_50 = pymc_hdi_50[:len(pymc_time_points)]
        else:
            pymc_time_points = time_points
            
        ax.fill_between(
            pymc_time_points,
            pymc_hdi_94[:, 0],
            pymc_hdi_94[:, 1],
            alpha=0.15,
            color="C1",
            label="PyMC 94% HDI"
        )
        ax.fill_between(
            pymc_time_points,
            pymc_hdi_50[:, 0],
            pymc_hdi_50[:, 1],
            alpha=0.25,
            color="C1",
            label="PyMC 50% HDI"
        )
        pymc_median = pymc_geo_preds.median(dim=["chain", "draw"]).values
        ax.plot(
            pymc_time_points,
            pymc_median[:len(pymc_time_points)],
            color="C1",
            linewidth=1.5,
            label="PyMC median",
            linestyle=":",
            alpha=0.8
        )
        
        # Formatting
        ax.set_title(f"{geo}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("Time Period", fontsize=11)
        if j == 0:
            ax.set_ylabel("KPI Value", fontsize=11)
        
        # Legend - only on first subplot to avoid clutter
        if j == 0:
            ax.legend(
                loc="upper left",
                framealpha=0.9,
                fontsize=9,
                ncol=2
            )
        
        # Format x-axis with dates
        n_ticks = 6
        tick_indices = np.linspace(0, len(geo_data_aligned) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        time_values = geo_data_aligned["time"].values
        ax.set_xticklabels(
            [str(time_values[int(i)])[:10] for i in tick_indices],
            rotation=45,
            ha='right'
        )
    
    # Main title
    fig.suptitle(
        f"{dataset_name.replace('_', ' ').title()} – Model Comparison (Meridian vs PyMC-Marketing)",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    
    if save:
        plot_path = storage.get_plot_path(dataset_name, "model_comparison.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved model comparison plot to {plot_path}")
    
    plt.close('all')


def plot_channel_contribution_distributions(
    channel_metrics_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot distributions of bias and sRMSE for channel contributions.
    
    Parameters
    ----------
    channel_metrics_df : pd.DataFrame
        DataFrame with channel contribution metrics
    save : bool
        Whether to save the plot
    """
    # Set up the plot with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define color palette for models
    model_colors = {
        'Meridian': '#1f77b4',
        'PyMC-Marketing - pymc': '#ff7f0e', 
        'PyMC-Marketing - blackjax': '#2ca02c',
        'PyMC-Marketing - numpyro': '#d62728',
        'PyMC-Marketing - nutpie': '#9467bd'
    }
    
    # Plot 1: Bias Distribution
    ax1 = axes[0]
    for model in channel_metrics_df['Model'].unique():
        model_data = channel_metrics_df[channel_metrics_df['Model'] == model]['Bias']
        color = model_colors.get(model, '#333333')
        
        # Plot violin plot
        parts = ax1.violinplot(
            [model_data.values],
            positions=[list(channel_metrics_df['Model'].unique()).index(model)],
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        
        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_color(color)
                parts[partname].set_linewidth(1.5)
    
    ax1.set_xticks(range(len(channel_metrics_df['Model'].unique())))
    ax1.set_xticklabels(channel_metrics_df['Model'].unique(), rotation=45, ha='right')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Bias', fontsize=12)
    ax1.set_title('Distribution of Channel Contribution Bias', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: sRMSE Distribution
    ax2 = axes[1]
    for model in channel_metrics_df['Model'].unique():
        model_data = channel_metrics_df[channel_metrics_df['Model'] == model]['SRMSE']
        color = model_colors.get(model, '#333333')
        
        # Plot violin plot
        parts = ax2.violinplot(
            [model_data.values],
            positions=[list(channel_metrics_df['Model'].unique()).index(model)],
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        
        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_color(color)
                parts[partname].set_linewidth(1.5)
    
    ax2.set_xticks(range(len(channel_metrics_df['Model'].unique())))
    ax2.set_xticklabels(channel_metrics_df['Model'].unique(), rotation=45, ha='right')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('sRMSE', fontsize=12)
    ax2.set_title('Distribution of Channel Contribution sRMSE', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='sRMSE = 1.0')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Overall title
    fig.suptitle('Channel Contribution Recovery Metrics Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots")
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_file = plot_path / "channel_metrics_distribution.png"
        plt.savefig(plot_file, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved channel metrics distribution plot to {plot_file}")
    
    plt.close('all')


def plot_channel_metrics_comparison(
    channel_metrics_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot detailed comparison of channel metrics across models.
    
    Parameters
    ----------
    channel_metrics_df : pd.DataFrame
        DataFrame with channel contribution metrics (including Region column if multi-region)
    save : bool
        Whether to save the plot
    """
    # Set up the plot with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define color palette for models
    model_colors = {
        'Meridian': '#1f77b4',
        'PyMC-Marketing - pymc': '#ff7f0e', 
        'PyMC-Marketing - blackjax': '#2ca02c',
        'PyMC-Marketing - numpyro': '#d62728',
        'PyMC-Marketing - nutpie': '#9467bd'
    }
    
    metrics = ['Bias', 'SRMSE', 'R²', 'MAPE (%)']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Create box plot for each channel
        channels = channel_metrics_df['Channel'].unique()
        x_pos = np.arange(len(channels))
        width = 0.15
        
        for i, model in enumerate(channel_metrics_df['Model'].unique()):
            model_data = channel_metrics_df[channel_metrics_df['Model'] == model]
            
            # Average across regions for each channel
            values = []
            for ch in channels:
                ch_data = model_data[model_data['Channel'] == ch][metric]
                if len(ch_data) > 0:
                    values.append(ch_data.mean())  # Average across regions
                else:
                    values.append(np.nan)
            
            color = model_colors.get(model, '#333333')
            offset = (i - len(channel_metrics_df['Model'].unique())/2) * width
            
            ax.bar(x_pos + offset, values, width, 
                  label=model.replace('PyMC-Marketing - ', 'PyMC-'),
                  color=color, alpha=0.7)
        
        ax.set_xlabel('Channel', fontsize=11)
        ax.set_ylabel(f'{metric} (avg across regions)', fontsize=11)
        ax.set_title(f'{metric} by Channel and Model', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([ch.replace('_', '\n') for ch in channels], fontsize=9)
        
        # Add reference lines
        if metric == 'Bias':
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        elif metric == 'SRMSE':
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        elif metric == 'R²':
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Overall title
    fig.suptitle('Channel-Level Metric Comparison Across Models (Averaged Over Regions)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots")
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_file = plot_path / "channel_metrics_comparison.png"
        plt.savefig(plot_file, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved channel metrics comparison plot to {plot_file}")
    
    plt.close('all')


