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
    plt.rcParams["figure.figsize"] = [12, 7]
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
    n_geos = len(data_df["geo"].unique())
    model_fit = visualizer.ModelFit(meridian_model)
    
    fig = model_fit.plot_model_fit(
        n_top_largest_geos=n_geos,
        show_geo_level=True,
        include_baseline=False,
        include_ci=True
    )
    
    if save:
        plot_path = storage.get_plot_path(dataset_name, "posterior_predictive_meridian.html")
        # Meridian visualizer returns a plotly figure, save as HTML
        try:
            fig.write_html(str(plot_path))
            print(f"  ✓ Saved Meridian posterior predictive plot to {plot_path}")
        except AttributeError:
            # If it's an Altair chart, save differently
            fig.save(str(plot_path))
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
        figsize=(6 * len(geos), 5),
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


def plot_runtime_comparison(
    runtime_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot runtime comparison across models.
    
    Parameters
    ----------
    runtime_df : pd.DataFrame
        Runtime comparison dataframe
    save : bool
        Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    runtime_df.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Model Runtime Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots/runtime_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved runtime comparison plot to {plot_path}")
    
    plt.close('all')


def plot_ess_comparison(
    ess_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot ESS comparison across models.
    
    Parameters
    ----------
    ess_df : pd.DataFrame
        ESS comparison dataframe
    save : bool
        Whether to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ["min", "q10", "q50", "q90"]
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        metric_data = ess_df[ess_df["ESS"] == metric].pivot(
            index="Dataset",
            columns="Sampler",
            values="value"
        )
        
        metric_data.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'ESS {metric}')
        ax.set_ylabel('ESS')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Effective Sample Size Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots/ess_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved ESS comparison plot to {plot_path}")
    
    plt.close('all')


def plot_performance_metrics(
    performance_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot performance metrics comparison.
    
    Parameters
    ----------
    performance_df : pd.DataFrame
        Performance metrics dataframe
    save : bool
        Whether to save the plot
    """
    metrics = performance_df["Metric"].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        metric_data = performance_df[performance_df["Metric"] == metric]
        
        libraries = [col for col in metric_data.columns 
                    if col not in ["Dataset", "Geo", "Metric"]]
        
        x = np.arange(len(metric_data))
        width = 0.35
        
        for i, lib in enumerate(libraries):
            offset = width * (i - len(libraries) / 2 + 0.5)
            ax.bar(x + offset, metric_data[lib], width, label=lib)
        
        ax.set_xlabel('Dataset-Geo')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['Dataset']}-{row['Geo']}" 
             for _, row in metric_data.iterrows()],
            rotation=45,
            ha='right'
        )
        ax.legend()
    
    plt.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots/performance_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved performance comparison plot to {plot_path}")
    
    plt.close('all')


def plot_ess_per_second_comparison(
    ess_per_second_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot ESS per second (efficiency) comparison across models.
    
    Parameters
    ----------
    ess_per_second_df : pd.DataFrame
        ESS per second comparison dataframe
    save : bool
        Whether to save the plot
    """
    # Pivot data for plotting
    pivot_data = ess_per_second_df.pivot_table(
        index=['Dataset', 'Sampler'],
        columns='Metric',
        values='ESS_per_s'
    ).reset_index()
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['min_per_s', 'q10_per_s', 'q50_per_s', 'q90_per_s']
    metric_labels = ['Min ESS/s', 'Q10 ESS/s', 'Median ESS/s', 'Q90 ESS/s']
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        if metric in pivot_data.columns:
            # Group by dataset for grouped bar chart
            datasets = pivot_data['Dataset'].unique()
            samplers = pivot_data['Sampler'].unique()
            
            x = np.arange(len(datasets))
            width = 0.8 / len(samplers)
            
            for i, sampler in enumerate(samplers):
                sampler_data = pivot_data[pivot_data['Sampler'] == sampler]
                values = []
                for dataset in datasets:
                    dataset_value = sampler_data[sampler_data['Dataset'] == dataset][metric].values
                    values.append(dataset_value[0] if len(dataset_value) > 0 else 0)
                
                offset = width * (i - len(samplers) / 2 + 0.5)
                ax.bar(x + offset, values, width, label=sampler)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('ESS per second')
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('ESS/s (Sampling Efficiency) Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots/ess_per_second_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved ESS/s comparison plot to {plot_path}")
    
    plt.close('all')


def plot_diagnostics_summary(
    diagnostics_df: pd.DataFrame,
    save: bool = True
) -> None:
    """Plot diagnostics summary.
    
    Parameters
    ----------
    diagnostics_df : pd.DataFrame
        Diagnostics summary dataframe
    save : bool
        Whether to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Runtime vs ESS min
    ax = axes[0, 0]
    for lib in diagnostics_df['Library'].unique():
        lib_data = diagnostics_df[diagnostics_df['Library'] == lib]
        ax.scatter(lib_data['Runtime (s)'], lib_data['ESS min'], label=lib, s=100)
    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('ESS min')
    ax.set_title('Runtime vs ESS min')
    ax.legend()
    
    # Divergences
    ax = axes[0, 1]
    divergences = diagnostics_df.pivot(index='Dataset', columns='Library', values='Divergences')
    divergences.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Number of Divergences')
    ax.set_title('Divergent Transitions')
    
    # R-hat max
    ax = axes[1, 0]
    rhat = diagnostics_df.pivot(index='Dataset', columns='Library', values='R-hat max')
    rhat.plot(kind='bar', ax=ax, rot=45)
    ax.axhline(y=1.1, color='r', linestyle='--', label='R-hat = 1.1')
    ax.set_ylabel('R-hat max')
    ax.set_title('Maximum R-hat')
    
    # Model size
    ax = axes[1, 1]
    size = diagnostics_df.pivot(index='Dataset', columns='Library', values='Size (MB)')
    size.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Model Memory Footprint')
    
    plt.suptitle('Model Diagnostics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = Path("data/results/summary/combined_plots/diagnostics_summary.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        print(f"  ✓ Saved diagnostics summary plot to {plot_path}")
    
    plt.close('all')