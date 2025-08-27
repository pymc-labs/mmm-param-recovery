#!/usr/bin/env python
"""Create detailed comparison plots for channel contributions across models with confidence intervals."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mmm_param_recovery.benchmarking import storage, data_loader, evaluation
import tensorflow as tf

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot channel contribution comparisons across models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="small_business",
        choices=["small_business", "medium_business", "large_business", "growing_business"],
        help="Dataset name"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Channel name to plot (e.g., x3_Local-Ads). If not provided, plots best and worst R² channels"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region to plot (e.g., geo_a). If not provided, plots all regions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: channel_comparison_{dataset}_{channel}.png)"
    )
    return parser.parse_args()


def get_confidence_intervals(model, dataset_name, channel_columns, channel_idx, library="meridian"):
    """Extract confidence intervals for channel contributions."""
    
    if library == "meridian":
        # Get Meridian posterior samples
        from meridian.analysis import analyzer
        model_analysis = analyzer.Analyzer(model)
        
        # Get incremental outcomes with all samples
        incremental_outcomes = model_analysis.incremental_outcome(
            aggregate_times=False,
            aggregate_geos=False,
            use_kpi=False,
            use_posterior=True
        )
        
        # Shape: (n_chains, n_draws, n_geos, n_times, n_channels)
        # Get contributions for specific channel
        channel_contrib = incremental_outcomes[:, :, :, :, channel_idx]
        
        # Calculate percentiles across chains and draws
        lower = tf.reduce_mean(tf.math.reduce_percentile(channel_contrib, 5, axis=[0, 1]), axis=0).numpy()
        upper = tf.reduce_mean(tf.math.reduce_percentile(channel_contrib, 95, axis=[0, 1]), axis=0).numpy()
        median = tf.reduce_mean(tf.math.reduce_percentile(channel_contrib, 50, axis=[0, 1]), axis=0).numpy()
        
    else:  # PyMC
        # Get PyMC posterior samples
        # Shape: (chain, draw, date, geo, channel)
        contribution_da = model.idata["posterior"]["channel_contribution_original_scale"]
        
        # Get contributions for specific channel
        channel_contrib = contribution_da.isel(channel=channel_idx)
        
        # Calculate percentiles across chains and draws
        lower = channel_contrib.quantile(0.05, dim=['chain', 'draw']).values
        upper = channel_contrib.quantile(0.95, dim=['chain', 'draw']).values
        median = channel_contrib.quantile(0.50, dim=['chain', 'draw']).values
        
    return lower, median, upper


def find_best_worst_channels(dataset_name):
    """Find channels with best and worst R² values."""
    
    # Try to read the channel contribution metrics
    metrics_file = "data/results/summary/channel_contribution_metrics.csv"
    
    import os
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        
        # Filter for dataset and Meridian
        meridian_metrics = metrics_df[
            (metrics_df['Dataset'] == dataset_name) & 
            (metrics_df['Model'] == 'Meridian')
        ]
        
        if len(meridian_metrics) > 0:
            # Group by channel and calculate mean R² across regions
            channel_r2 = meridian_metrics.groupby('Channel')['R²'].mean().sort_values()
            
            worst_channel = channel_r2.index[0]
            best_channel = channel_r2.index[-1]
            
            print(f"\nChannel R² values for {dataset_name} (from CSV):")
            for ch in channel_r2.index:
                print(f"  {ch}: R²={channel_r2[ch]:.3f}")
            
            print(f"\nBest R²: {best_channel} ({channel_r2[best_channel]:.3f})")
            print(f"Worst R²: {worst_channel} ({channel_r2[worst_channel]:.3f})")
            
            return best_channel, worst_channel
    
    # Fallback: if no metrics CSV or no data, return None to trigger manual selection
    print(f"\nNo pre-computed R² metrics found for {dataset_name}.")
    print("Will use first and middle channels as default.")
    return None, None


def plot_channel_for_region(
    data_df, truth_df, channel, region, 
    meridian_model, pymc_model, channel_columns,
    ax_spend, ax_contrib, ax_error, show_legend=True
):
    """Plot channel contributions for a specific region."""
    
    channel_idx = channel_columns.index(channel)
    
    # Filter data for region
    region_data = data_df[data_df['geo'] == region]
    truth_region = truth_df[truth_df['geo'] == region].copy()
    
    # Extract contributions
    true_col = f"contribution_{channel}"
    true_contrib = truth_region[true_col].values if true_col in truth_region.columns else np.zeros(len(truth_region))
    
    # Get dates
    dates = pd.to_datetime(truth_region['time'].values)
    
    # Get spend data
    spend = region_data[channel].values if channel in region_data.columns else np.zeros(len(region_data))
    
    # Extract Meridian contributions with confidence intervals
    from meridian.analysis import analyzer
    model_analysis = analyzer.Analyzer(meridian_model)
    
    incremental_outcomes = model_analysis.incremental_outcome(
        aggregate_times=False,
        aggregate_geos=False,
        use_kpi=False,
        use_posterior=True
    )
    
    # Get region index
    geo_names = meridian_model.input_data.geo.coords["geo"].values
    if len(geo_names) == 1 and geo_names[0] == 'national_geo':
        geo_idx = 0
    else:
        geo_idx = list(geo_names).index(region) if region in geo_names else 0
    
    # Get Meridian contributions
    meridian_samples = incremental_outcomes[:, :, geo_idx, :, channel_idx]
    meridian_lower = np.percentile(meridian_samples, 5, axis=(0, 1))
    meridian_upper = np.percentile(meridian_samples, 95, axis=(0, 1))
    meridian_median = np.median(meridian_samples, axis=(0, 1))
    
    # Extract PyMC contributions with confidence intervals
    pymc_contrib_da = pymc_model.idata["posterior"]["channel_contribution_original_scale"]
    pymc_channel_contrib = pymc_contrib_da.sel(geo=region, channel=channel_columns[channel_idx])
    pymc_lower = pymc_channel_contrib.quantile(0.05, dim=['chain', 'draw']).values
    pymc_upper = pymc_channel_contrib.quantile(0.95, dim=['chain', 'draw']).values
    pymc_median = pymc_channel_contrib.quantile(0.50, dim=['chain', 'draw']).values
    
    # Define colors
    spend_color = '#2E7D32'
    true_color = '#1565C0'
    meridian_color = '#D32F2F'
    pymc_color = '#F57C00'
    
    # Plot spend
    bars = ax_spend.bar(dates, spend, alpha=0.6, color=spend_color, label='Spend', width=6)
    
    # Highlight zero-spend periods
    for i, s in enumerate(spend):
        if s == 0:
            ax_spend.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                            alpha=0.1, color='red', zorder=0)
            ax_contrib.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                              alpha=0.1, color='red', zorder=0)
            ax_error.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                            alpha=0.1, color='red', zorder=0)
    
    ax_spend.set_ylabel('Spend ($)', fontsize=10)
    ax_spend.set_title(f'{region}: {channel} (red shading = zero spend)', fontsize=11)
    if show_legend:
        ax_spend.legend(loc='upper right', fontsize=9)
    ax_spend.grid(True, alpha=0.3)
    
    # Plot contributions with confidence intervals
    ax_contrib.plot(dates, true_contrib, label='Ground Truth', color=true_color, linewidth=2, alpha=0.9)
    
    # Meridian with CI
    ax_contrib.plot(dates, meridian_median, label='Meridian', color=meridian_color, linewidth=1.5)
    ax_contrib.fill_between(dates, meridian_lower, meridian_upper, color=meridian_color, alpha=0.2)
    
    # PyMC with CI
    ax_contrib.plot(dates, pymc_median, label='PyMC-Marketing (nutpie)', color=pymc_color, linewidth=1.5)
    ax_contrib.fill_between(dates, pymc_lower, pymc_upper, color=pymc_color, alpha=0.2)
    
    ax_contrib.set_ylabel('Contribution ($)', fontsize=10)
    if show_legend:
        ax_contrib.legend(loc='upper right', fontsize=9)
    ax_contrib.grid(True, alpha=0.3)
    
    # Plot errors
    meridian_error = meridian_median - true_contrib
    pymc_error = pymc_median - true_contrib
    
    width = 3
    ax_error.bar(dates - pd.Timedelta(days=width/2), meridian_error, width=width, 
                alpha=0.6, color=meridian_color, label='Meridian Error')
    ax_error.bar(dates + pd.Timedelta(days=width/2), pymc_error, width=width, 
                alpha=0.6, color=pymc_color, label='PyMC Error')
    
    ax_error.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax_error.set_ylabel('Error ($)', fontsize=10)
    ax_error.set_xlabel('Date', fontsize=10)
    if show_legend:
        ax_error.legend(loc='upper right', fontsize=9)
    ax_error.grid(True, alpha=0.3)
    
    # Format x-axes
    for ax in [ax_spend, ax_contrib, ax_error]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Calculate and return metrics
    r2_meridian = calculate_r2(true_contrib, meridian_median)
    r2_pymc = calculate_r2(true_contrib, pymc_median)
    mape_meridian = calculate_mape(true_contrib, meridian_median)
    mape_pymc = calculate_mape(true_contrib, pymc_median)
    bias_meridian = calculate_bias(true_contrib, meridian_median)
    bias_pymc = calculate_bias(true_contrib, pymc_median)
    
    return {
        'region': region,
        'r2_meridian': r2_meridian,
        'r2_pymc': r2_pymc,
        'mape_meridian': mape_meridian,
        'mape_pymc': mape_pymc,
        'bias_meridian': bias_meridian,
        'bias_pymc': bias_pymc
    }


def create_channel_comparison_plot(dataset_name, channel=None, region=None):
    """Create comparison plot for specified channel and dataset."""
    
    seed = 2148
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    datasets = data_loader.load_multiple_datasets([dataset_name], seed)
    data_df, channel_columns, control_columns, truth_df = datasets[0]
    
    # Determine channel(s) to plot
    if channel is None:
        best_channel, worst_channel = find_best_worst_channels(dataset_name)
        if best_channel is None or worst_channel is None:
            # Fallback: use first and middle channel
            channels_to_plot = [channel_columns[0], channel_columns[len(channel_columns)//2]]
            plot_titles = ['First Channel', 'Middle Channel']
            print(f"Using channels: {channels_to_plot}")
        else:
            channels_to_plot = [best_channel, worst_channel]
            plot_titles = ['Best R² Channel', 'Worst R² Channel']
    else:
        if channel not in channel_columns:
            raise ValueError(f"Channel {channel} not found. Available: {channel_columns}")
        channels_to_plot = [channel]
        plot_titles = [channel]
    
    # Determine regions to plot
    all_regions = data_df['geo'].unique()
    if region is not None:
        if region not in all_regions:
            raise ValueError(f"Region {region} not found. Available: {all_regions}")
        regions_to_plot = [region]
    else:
        # Plot up to 3 regions for space
        regions_to_plot = all_regions[:min(3, len(all_regions))]
    
    # Load models
    print("Loading Meridian model...")
    meridian_model, _, _ = storage.load_meridian_model(dataset_name)
    
    print("Loading PyMC-Marketing (nutpie) model...")
    if not storage.model_exists(dataset_name, "pymc", "nutpie"):
        raise ValueError(f"PyMC nutpie model not found for {dataset_name}")
    pymc_model, _, _ = storage.load_pymc_model(dataset_name, "nutpie")
    
    # Create plots
    n_channels = len(channels_to_plot)
    n_regions = len(regions_to_plot)
    
    fig, axes = plt.subplots(3 * n_channels, n_regions, 
                            figsize=(6 * n_regions, 4 * n_channels * 3),
                            squeeze=False)
    
    all_metrics = []
    
    for ch_idx, channel_name in enumerate(channels_to_plot):
        print(f"\nPlotting {channel_name}...")
        
        for reg_idx, region_name in enumerate(regions_to_plot):
            # Get axes for this channel-region combination
            ax_spend = axes[ch_idx * 3, reg_idx]
            ax_contrib = axes[ch_idx * 3 + 1, reg_idx]
            ax_error = axes[ch_idx * 3 + 2, reg_idx]
            
            # Only show legend on first subplot
            show_legend = (ch_idx == 0 and reg_idx == 0)
            
            # Plot for this region
            metrics = plot_channel_for_region(
                data_df, truth_df, channel_name, region_name,
                meridian_model, pymc_model, channel_columns,
                ax_spend, ax_contrib, ax_error, show_legend
            )
            metrics['channel'] = channel_name
            all_metrics.append(metrics)
    
    # Add overall title
    if channel is None:
        fig.suptitle(f'{dataset_name}: Best vs Worst R² Channels', fontsize=14, fontweight='bold', y=0.995)
    else:
        fig.suptitle(f'{dataset_name}: {channel} Channel Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # Add summary statistics at bottom
    stats_text = []
    for m in all_metrics:
        stats_text.append(
            f"{m['channel']} - {m['region']}: "
            f"Meridian R²={m['r2_meridian']:.2f}, "
            f"PyMC R²={m['r2_pymc']:.2f}"
        )
    
    fig.text(0.5, 0.01, ' | '.join(stats_text), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.03, top=0.97)
    
    # Save plot
    if channel:
        output_path = f'channel_comparison_{dataset_name}_{channel.replace("-", "_")}.png'
    else:
        output_path = f'channel_comparison_{dataset_name}_best_worst.png'
    
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Print detailed metrics
    print("\n" + "=" * 60)
    print("DETAILED METRICS")
    print("=" * 60)
    
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False))
    
    return fig, metrics_df


def calculate_r2(actual, predicted):
    """Calculate R-squared."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() < 3:
        return np.nan
    actual = actual[mask]
    predicted = predicted[mask]
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


def calculate_mape(actual, predicted):
    """Calculate MAPE."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (np.abs(actual) > 1e-3)
    if mask.sum() == 0:
        return np.nan
    actual = actual[mask]
    predicted = predicted[mask]
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_bias(actual, predicted):
    """Calculate bias."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    return np.mean(predicted - actual)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        fig, metrics = create_channel_comparison_plot(
            args.dataset,
            args.channel,
            args.region
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()