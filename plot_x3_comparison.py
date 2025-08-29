#!/usr/bin/env python
"""Create detailed comparison plot for x3_Local-Ads channel contributions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mmm_param_recovery.benchmarking import storage, data_loader, evaluation

def create_x3_comparison_plot():
    """Create comparison plot for x3_Local-Ads contributions."""
    
    dataset_name = "small_business"
    seed = 2148
    channel = "x3_Local-Ads"
    channel_idx = 2
    
    # Load data
    print("Loading data...")
    datasets = data_loader.load_multiple_datasets([dataset_name], seed)
    data_df, channel_columns, control_columns, truth_df = datasets[0]
    
    # Extract ground truth contributions
    truth_local = truth_df[truth_df['geo'] == 'Local'].copy()
    true_col = f"contribution_{channel}"
    true_contrib = truth_local[true_col].values
    
    # Get dates for x-axis
    dates = pd.to_datetime(truth_local['time'].values)
    
    # Get spend data
    spend = data_df[data_df['geo'] == 'Local'][channel].values
    
    print("Loading Meridian model...")
    # Load and extract Meridian contributions
    meridian_model, _, _ = storage.load_meridian_model(dataset_name)
    meridian_contrib_df = evaluation.extract_meridian_channel_contributions(
        meridian_model, channel_columns
    )
    meridian_contrib = meridian_contrib_df[channel].values
    
    print("Loading PyMC-Marketing (nutpie) model...")
    # Load and extract PyMC contributions
    pymc_model, _, _ = storage.load_pymc_model(dataset_name, "nutpie")
    pymc_contrib_df = evaluation.extract_pymc_channel_contributions(
        pymc_model, channel_columns
    )
    pymc_contrib = pymc_contrib_df[channel].values
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Define colors
    spend_color = '#2E7D32'  # Dark green
    true_color = '#1565C0'  # Blue
    meridian_color = '#D32F2F'  # Red
    pymc_color = '#F57C00'  # Orange
    
    # ============= SUBPLOT 1: Spend Pattern =============
    ax1 = axes[0]
    bars = ax1.bar(dates, spend, alpha=0.6, color=spend_color, label='Spend', width=6)
    
    # Highlight zero-spend periods
    for i, s in enumerate(spend):
        if s == 0:
            ax1.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                       alpha=0.1, color='red', zorder=0)
    
    ax1.set_ylabel('Spend ($)', fontsize=11)
    ax1.set_title(f'x3_Local-Ads: Spend Pattern (red shading = zero spend periods)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(dates[0] - pd.Timedelta(days=5), dates[-1] + pd.Timedelta(days=5))
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ============= SUBPLOT 2: Contribution Comparison =============
    ax2 = axes[1]
    
    # Plot contributions
    ax2.plot(dates, true_contrib, label='Ground Truth', color=true_color, linewidth=2.5, alpha=0.8)
    ax2.plot(dates, meridian_contrib, label='Meridian', color=meridian_color, linewidth=2, alpha=0.8, linestyle='--')
    ax2.plot(dates, pymc_contrib, label='PyMC-Marketing (nutpie)', color=pymc_color, linewidth=2, alpha=0.8, linestyle='-.')
    
    # Highlight zero-spend periods
    for i, s in enumerate(spend):
        if s == 0:
            ax2.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                       alpha=0.1, color='red', zorder=0)
    
    ax2.set_ylabel('Contribution ($)', fontsize=11)
    ax2.set_title('Channel Contributions Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(dates[0] - pd.Timedelta(days=5), dates[-1] + pd.Timedelta(days=5))
    
    # Add annotations for problematic periods
    # Find periods where Meridian is way off
    error = meridian_contrib - true_contrib
    worst_idx = np.argmax(np.abs(error))
    ax2.annotate(f'Largest error\n({error[worst_idx]:+.0f})', 
                xy=(dates[worst_idx], meridian_contrib[worst_idx]),
                xytext=(dates[worst_idx] + pd.Timedelta(days=20), meridian_contrib[worst_idx] + 500),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                fontsize=9, color='red')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ============= SUBPLOT 3: Prediction Errors =============
    ax3 = axes[2]
    
    # Calculate errors
    meridian_error = meridian_contrib - true_contrib
    pymc_error = pymc_contrib - true_contrib
    
    # Plot errors as bars
    width = 3
    x_pos = np.arange(len(dates))
    
    # Create bar plot for errors
    ax3.bar(dates - pd.Timedelta(days=width/2), meridian_error, width=width, 
           alpha=0.6, color=meridian_color, label='Meridian Error')
    ax3.bar(dates + pd.Timedelta(days=width/2), pymc_error, width=width, 
           alpha=0.6, color=pymc_color, label='PyMC Error')
    
    # Add zero line
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Highlight zero-spend periods
    for i, s in enumerate(spend):
        if s == 0:
            ax3.axvspan(dates[i] - pd.Timedelta(days=3), dates[i] + pd.Timedelta(days=3), 
                       alpha=0.1, color='red', zorder=0)
    
    ax3.set_ylabel('Prediction Error ($)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Prediction Errors (Model - Truth)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(dates[0] - pd.Timedelta(days=5), dates[-1] + pd.Timedelta(days=5))
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add summary statistics
    fig.text(0.15, 0.02, f'Meridian: R²={calculate_r2(true_contrib, meridian_contrib):.3f}, '
                        f'MAPE={calculate_mape(true_contrib, meridian_contrib):.0f}%, '
                        f'Bias={calculate_bias(true_contrib, meridian_contrib):.0f}',
            fontsize=10, color=meridian_color)
    fig.text(0.55, 0.02, f'PyMC-Marketing: R²={calculate_r2(true_contrib, pymc_contrib):.3f}, '
                        f'MAPE={calculate_mape(true_contrib, pymc_contrib):.0f}%, '
                        f'Bias={calculate_bias(true_contrib, pymc_contrib):.0f}',
            fontsize=10, color=pymc_color)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for statistics
    
    # Save plot
    output_path = 'x3_local_ads_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\nGround Truth:")
    print(f"  Total: ${true_contrib.sum():,.0f}")
    print(f"  Mean: ${true_contrib.mean():.0f}")
    print(f"  When spend=0: ${true_contrib[spend == 0].mean():.0f}")
    
    print(f"\nMeridian:")
    print(f"  Total: ${meridian_contrib.sum():,.0f}")
    print(f"  Mean: ${meridian_contrib.mean():.0f}")
    print(f"  When spend=0: ${meridian_contrib[spend == 0].mean():.0f}")
    print(f"  R²: {calculate_r2(true_contrib, meridian_contrib):.3f}")
    print(f"  MAPE: {calculate_mape(true_contrib, meridian_contrib):.0f}%")
    print(f"  Bias: ${calculate_bias(true_contrib, meridian_contrib):.0f}")
    
    print(f"\nPyMC-Marketing (nutpie):")
    print(f"  Total: ${pymc_contrib.sum():,.0f}")
    print(f"  Mean: ${pymc_contrib.mean():.0f}")
    print(f"  When spend=0: ${pymc_contrib[spend == 0].mean():.0f}")
    print(f"  R²: {calculate_r2(true_contrib, pymc_contrib):.3f}")
    print(f"  MAPE: {calculate_mape(true_contrib, pymc_contrib):.0f}%")
    print(f"  Bias: ${calculate_bias(true_contrib, pymc_contrib):.0f}")
    
    return fig

def calculate_r2(actual, predicted):
    """Calculate R-squared."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
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

if __name__ == "__main__":
    fig = create_x3_comparison_plot()