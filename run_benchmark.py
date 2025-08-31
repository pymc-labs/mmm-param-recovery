#!/usr/bin/env python
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

"""Main benchmarking script for MMM parameter recovery comparison."""

import argparse
import gc
import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import track
from rich import box
from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmm_param_recovery.benchmarking import (
    data_loader,
    model_builder,
    model_fitter,
    diagnostics,
    evaluation,
    visualization,
    storage,
    bayesian_evaluation,
    parameter_counter
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run MMM benchmarking comparison between PyMC-Marketing and Meridian"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["small_business"],
        choices=["small_business", "medium_business", "large_business", "growing_business"],
        help="Datasets to benchmark"
    )
    
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["pymc", "blackjax", "numpyro", "nutpie"],
        choices=["pymc", "blackjax", "numpyro", "nutpie"],
        help="PyMC-Marketing samplers to use"
    )
    
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=["meridian", "pymc"],
        choices=["meridian", "pymc"],
        help="Libraries to benchmark"
    )
    
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of chains"
    )
    
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Number of draws per chain"
    )
    
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Number of tuning samples"
    )
    
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="Target acceptance probability"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=sum(map(ord, "mmm_multidimensional")),
        help="Random seed"
    )
    
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if cached results exist"
    )
    
    parser.add_argument(
        "--no-force-rerun",
        dest="force_rerun",
        action="store_false",
        help="Use cached results if available (default)"
    )
    
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate only plots from existing models"
    )
    
    parser.add_argument(
        "--bayesian-metrics",
        action="store_true",
        help="Use Bayesian metric calculation (compute metrics on posterior samples)"
    )
    
    parser.set_defaults(force_rerun=False)
    
    return parser.parse_args()


def run_benchmark_for_dataset(
    dataset_name: str,
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str],
    truth_df: pd.DataFrame,
    args: argparse.Namespace,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """Run benchmark for a single dataset with memory isolation.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_df : pd.DataFrame
        Prepared dataset
    channel_columns : List[str]
        Channel column names
    control_columns : List[str]
        Control column names
    truth_df : pd.DataFrame
        Ground truth dataframe
    args : argparse.Namespace
        Command line arguments
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with fitted models and metrics
    """
    if console is None:
        console = Console()
    all_performance_rows = []
    all_channel_contribution_rows = []
    channel_contribution_averages = []
    all_parameter_counts = []
    
    # PHASE 1: Fit and save models (if needed) with memory isolation
    if not args.plots_only:
        console.print()
        console.rule("[bold cyan]PHASE 1: FITTING MODELS (ISOLATED)[/bold cyan]")
        
        # Fit Meridian
        if "meridian" in args.libraries:
            if not storage.model_exists(dataset_name, "meridian") or args.force_rerun:
                console.print("\n[bold yellow]--- Meridian ---[/bold yellow]")
                meridian_result, runtime, ess = model_fitter.fit_meridian(
                    data_df,
                    channel_columns,
                    control_columns,
                    args.chains,
                    args.draws,
                    args.tune,
                    args.target_accept,
                    args.seed,
                    console
                )
                storage.save_meridian_model(meridian_result, dataset_name, runtime, ess)
                del meridian_result
                gc.collect()
                console.print("  [green]✓[/green] Model fitted, saved, and memory cleared")
        
        # Fit PyMC models
        if "pymc" in args.libraries:
            for sampler in args.samplers:
                if model_fitter.should_skip_sampler(sampler, dataset_name, console):
                    continue
                
                if not storage.model_exists(dataset_name, "pymc", sampler) or args.force_rerun:
                    console.print(f"\n[bold yellow]--- PyMC-Marketing - {sampler} ---[/bold yellow]")
                    pymc_result, runtime, ess = model_fitter.fit_pymc(
                        data_df,
                        channel_columns,
                        control_columns,
                        sampler,
                        args.chains,
                        args.draws,
                        args.tune,
                        args.target_accept,
                        args.seed,
                        console
                    )
                    storage.save_pymc_model(pymc_result, dataset_name, sampler, runtime, ess)
                    del pymc_result
                    gc.collect()
                    console.print("  [green]✓[/green] Model fitted, saved, and memory cleared")
    
    # PHASE 2: Load all models for evaluation
    console.print()
    console.rule("[bold cyan]PHASE 2: EVALUATING MODELS[/bold cyan]")
    results = {}
    
    # Load and evaluate Meridian
    if "meridian" in args.libraries and storage.model_exists(dataset_name, "meridian"):
        console.print("\n[bold yellow]--- Evaluating Meridian ---[/bold yellow]")
        meridian_result, runtime, ess = storage.load_meridian_model(dataset_name)
        results["Meridian"] = (meridian_result, runtime, ess)
        
        perf_rows = evaluation.evaluate_meridian_fit(meridian_result, data_df)
        for row in perf_rows:
            row["Dataset"] = dataset_name
        all_performance_rows.extend(perf_rows)
        
        # Evaluate channel contributions
        channel_df, avg_metrics = evaluation.evaluate_meridian_channel_contributions(
            meridian_result, truth_df, channel_columns, dataset_name
        )
        all_channel_contribution_rows.append(channel_df)
        channel_contribution_averages.append({
            "Dataset": dataset_name,
            "Model": "Meridian",
            **avg_metrics
        })
        
        # Count parameters
        param_counts = parameter_counter.categorize_parameters(meridian_result, "meridian")
        param_counts["Dataset"] = dataset_name
        param_counts["Model"] = "Meridian"
        all_parameter_counts.append(param_counts)
        
        visualization.plot_meridian_posterior_predictive(
            meridian_result, data_df, dataset_name
        )
    
    # Load and evaluate PyMC models  
    if "pymc" in args.libraries:
        for sampler in args.samplers:
            if model_fitter.should_skip_sampler(sampler, dataset_name):
                continue
                
            if storage.model_exists(dataset_name, "pymc", sampler):
                console.print(f"\n[bold yellow]--- Evaluating PyMC-Marketing - {sampler} ---[/bold yellow]")
                pymc_result, runtime, ess = storage.load_pymc_model(dataset_name, sampler)
                results[f"PyMC-Marketing - {sampler}"] = (pymc_result, runtime, ess)
                
                perf_rows = evaluation.evaluate_pymc_fit(pymc_result, data_df, sampler)
                for row in perf_rows:
                    row["Dataset"] = dataset_name
                all_performance_rows.extend(perf_rows)
                
                # Evaluate channel contributions
                channel_df, avg_metrics = evaluation.evaluate_pymc_channel_contributions(
                    pymc_result, truth_df, channel_columns, sampler, dataset_name
                )
                all_channel_contribution_rows.append(channel_df)
                channel_contribution_averages.append({
                    "Dataset": dataset_name,
                    "Model": f"PyMC-Marketing - {sampler}",
                    **avg_metrics
                })
                
                # Count parameters
                param_counts = parameter_counter.categorize_parameters(pymc_result, "pymc")
                param_counts["Dataset"] = dataset_name
                param_counts["Model"] = f"PyMC-Marketing - {sampler}"
                all_parameter_counts.append(param_counts)
                
                visualization.plot_pymc_posterior_predictive(
                    pymc_result, data_df, dataset_name, sampler
                )
    
    results["performance"] = all_performance_rows
    results["channel_contributions"] = all_channel_contribution_rows
    results["channel_averages"] = channel_contribution_averages
    results["parameter_counts"] = all_parameter_counts
    
    # PHASE 3: Bayesian Evaluation (if requested)
    if args.bayesian_metrics:
        console.print()
        console.rule("[bold cyan]PHASE 3: BAYESIAN EVALUATION[/bold cyan]")
        bayesian_results = {}
        
        # Evaluate Meridian with Bayesian metrics
        if "Meridian" in results:
            console.print("\n[bold yellow]--- Bayesian Evaluation: Meridian ---[/bold yellow]")
            meridian_model, _, _ = results["Meridian"]
            
            # Revenue metrics
            revenue_metrics = bayesian_evaluation.evaluate_revenue_bayesian(
                meridian_model, data_df, 'meridian'
            )
            
            # Channel contribution metrics
            contrib_detailed, contrib_aggregated = bayesian_evaluation.evaluate_contributions_bayesian(
                meridian_model, truth_df, channel_columns, 'meridian'
            )
            
            bayesian_results["Meridian"] = {
                'revenue': revenue_metrics,
                'contributions_detailed': contrib_detailed,
                'contributions_aggregated': contrib_aggregated
            }
            
            # Print summary
            for geo, metrics in revenue_metrics.items():
                console.print(f"  {geo} - R²: {bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(metrics['R²'], 3)}")
            console.print(f"  Channel Contributions - Avg R²: {bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(contrib_aggregated['R²'], 3)}")
        
        # Evaluate PyMC models with Bayesian metrics
        for sampler in args.samplers:
            model_key = f"PyMC-Marketing - {sampler}"
            if model_key in results:
                console.print(f"\n[bold yellow]--- Bayesian Evaluation: {model_key} ---[/bold yellow]")
                pymc_model, _, _ = results[model_key]
                
                # Revenue metrics
                revenue_metrics = bayesian_evaluation.evaluate_revenue_bayesian(
                    pymc_model, data_df, 'pymc'
                )
                
                # Channel contribution metrics
                contrib_detailed, contrib_aggregated = bayesian_evaluation.evaluate_contributions_bayesian(
                    pymc_model, truth_df, channel_columns, 'pymc'
                )
                
                bayesian_results[model_key] = {
                    'revenue': revenue_metrics,
                    'contributions_detailed': contrib_detailed,
                    'contributions_aggregated': contrib_aggregated
                }
                
                # Print summary
                for geo, metrics in revenue_metrics.items():
                    print(f"  {geo} - R²: {bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(metrics['R²'], 3)}")
                print(f"  Channel Contributions - Avg R²: {bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(contrib_aggregated['R²'], 3)}")
        
        results["bayesian_metrics"] = bayesian_results
    
    # Generate comparison plot if both models exist
    if "Meridian" in results and "PyMC-Marketing - nutpie" in results:
        console.print("\n[bold yellow]--- Generating Model Comparison Plot ---[/bold yellow]")
        meridian_result, _, _ = results["Meridian"]
        pymc_nutpie_result, _, _ = results["PyMC-Marketing - nutpie"]
        
        visualization.plot_model_comparison(
            meridian_result,
            pymc_nutpie_result,
            data_df,
            dataset_name
        )
    
    return results


def create_summary_tables(
    all_results: Dict[str, Dict[str, Any]],
    dataset_names: List[str],
    console: Optional[Console] = None
) -> None:
    """Create and save summary tables.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        All results keyed by dataset name
    dataset_names : List[str]
        List of dataset names
    console : Optional[Console]
        Rich console for output
    """
    if console is None:
        console = Console()
    # Runtime summary
    runtime_data = {"Dataset": dataset_names}
    ess_rows = []
    all_diagnostics_rows = []
    all_performance_rows = []
    all_channel_contribution_dfs = []
    all_channel_averages = []
    all_parameter_counts = []
    
    for dataset_name in dataset_names:
        dataset_results = all_results[dataset_name]
        
        # Collect runtime data
        for key, value in dataset_results.items():
            if key == "performance":
                all_performance_rows.extend(value)
                continue
            elif key == "channel_contributions":
                all_channel_contribution_dfs.extend(value)
                continue
            elif key == "channel_averages":
                all_channel_averages.extend(value)
                continue
            elif key == "parameter_counts":
                all_parameter_counts.extend(value)
                continue
            elif key == "bayesian_metrics":
                # Skip bayesian_metrics as it's handled separately
                continue
            
            model, runtime, ess = value
            
            if key not in runtime_data:
                runtime_data[key] = []
            runtime_data[key].append(runtime)
            
            # Collect ESS data
            for metric_name, metric_value in [
                ("min", ess.get("min")),
                ("q10", ess.get("q10")),
                ("q50", ess.get("q50")),
                ("q90", ess.get("q90"))
            ]:
                ess_rows.append({
                    "Dataset": dataset_name,
                    "Model": key,
                    "ESS": metric_name,
                    "value": metric_value
                })
        
        # Create diagnostics summary - filter out non-model entries
        model_results = {
            k: v for k, v in dataset_results.items() 
            if k not in ["performance", "channel_contributions", "channel_averages", "parameter_counts", "bayesian_metrics"]
        }
        diag_df = diagnostics.create_diagnostics_summary(
            model_results,
            dataset_name
        )
        all_diagnostics_rows.append(diag_df)
    
    # Fix for unequal column lengths in runtime_data
    # This happens when different datasets have different models fitted
    # (e.g., large datasets skip memory-intensive JAX samplers like blackjax/numpyro)
    # We need to ensure all columns have the same length by padding with NaN
    for key in runtime_data:
        if key != "Dataset":
            # Pad with NaN values if this model wasn't run for all datasets
            while len(runtime_data[key]) < len(dataset_names):
                runtime_data[key].append(np.nan)
    
    # SAMPLING PROCESS SUMMARY SECTION
    console.print()
    console.rule("[bold cyan]SAMPLING PROCESS SUMMARY[/bold cyan]")
    
    # Save runtime summary
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df.set_index("Dataset", inplace=True)
    storage.save_summary_dataframe(runtime_df, "runtime_comparison")
    
    # Create Rich table for runtime summary
    runtime_table = Table(title="Runtime Summary (seconds)", box=box.ROUNDED)
    runtime_table.add_column("Dataset", style="cyan", no_wrap=True)
    for col in runtime_df.columns:
        # Model/sampler columns get yellow styling
        runtime_table.add_column(col, justify="right", style="yellow")
    
    for idx, row in runtime_df.iterrows():
        runtime_table.add_row(
            str(idx),
            *[f"{val:.1f}" if pd.notna(val) else "N/A" for val in row]
        )
    
    console.print()
    console.print(runtime_table)
    
    # Save ESS summary
    ess_df = pd.DataFrame(ess_rows)
    storage.save_summary_dataframe(ess_df, "ess_comparison")
    
    ess_pivot = ess_df.pivot_table(
        index=["Dataset", "Model"],
        columns="ESS",
        values="value"
    ).round(2)
    # Sort by Dataset first, then by Model
    ess_pivot = ess_pivot.sort_index(level=['Dataset', 'Model'])
    # Create Rich table for ESS summary
    ess_table = Table(title="ESS Summary", box=box.ROUNDED)
    ess_table.add_column("Dataset", style="cyan")
    ess_table.add_column("Model", style="yellow")
    for col in ['min', 'q10', 'q50', 'q90']:
        ess_table.add_column(col, justify="right")
    
    for idx, row in ess_pivot.iterrows():
        dataset, model = idx
        ess_table.add_row(
            dataset, model,
            *[f"{row[col]:.0f}" if pd.notna(row[col]) else "N/A" for col in ['min', 'q10', 'q50', 'q90']]
        )
    
    console.print()
    console.print(ess_table)
    
    # Calculate and save ESS/s (Effective Sample Size per second)
    if ess_rows and runtime_data:
        # Reshape runtime data for merging
        runtime_melted = runtime_df.reset_index().melt(
            id_vars="Dataset",
            var_name="Model",
            value_name="Runtime"
        )
        
        # Reshape ESS data to wide format
        ess_wide = ess_df.pivot_table(
            index=["Dataset", "Model"],
            columns="ESS",
            values="value"
        ).reset_index()
        
        # Merge ESS and runtime data
        ess_with_runtime = ess_wide.merge(
            runtime_melted,
            on=["Dataset", "Model"],
            how="inner"
        )
        
        # Calculate ESS/s for each metric
        ess_per_second_rows = []
        for _, row in ess_with_runtime.iterrows():
            for metric in ["min", "q10", "q50", "q90"]:
                if metric in row and pd.notna(row[metric]) and row["Runtime"] > 0:
                    ess_per_second_rows.append({
                        "Dataset": row["Dataset"],
                        "Model": row["Model"],
                        "Metric": f"{metric}_per_s",
                        "ESS_per_s": row[metric] / row["Runtime"]
                    })
        
        # Create ESS/s DataFrame
        ess_per_second_df = pd.DataFrame(ess_per_second_rows)
        
        if not ess_per_second_df.empty:
            # Save ESS/s summary
            storage.save_summary_dataframe(ess_per_second_df, "ess_per_second_comparison")
            
            # Pivot for display
            ess_per_s_pivot = ess_per_second_df.pivot_table(
                index=["Dataset", "Model"],
                columns="Metric",
                values="ESS_per_s"
            )
            
            # Sort by Dataset first, then by median ESS/s (q50_per_s) within each dataset
            if "q50_per_s" in ess_per_s_pivot.columns:
                ess_per_s_pivot = ess_per_s_pivot.sort_index(level='Dataset').sort_values("q50_per_s", ascending=False)
            else:
                ess_per_s_pivot = ess_per_s_pivot.sort_index(level=['Dataset', 'Model'])
            
            # Create Rich table for ESS/s summary
            ess_per_s_table = Table(title="ESS/s (Efficiency) Summary", box=box.ROUNDED)
            ess_per_s_table.add_column("Dataset", style="cyan")
            ess_per_s_table.add_column("Model", style="yellow")
            for col in ess_per_s_pivot.columns:
                ess_per_s_table.add_column(col, justify="right")
            
            for idx, row in ess_per_s_pivot.iterrows():
                dataset, model = idx
                ess_per_s_table.add_row(
                    dataset, model,
                    *[f"{row[col]:.2f}" if pd.notna(row[col]) else "N/A" for col in ess_per_s_pivot.columns]
                )
            
            console.print()
            console.print(ess_per_s_table)
    
    # Save diagnostics summary
    if all_diagnostics_rows:
        diagnostics_df = pd.concat(all_diagnostics_rows, ignore_index=True)
        # Sort diagnostics by Dataset and Model
        if 'Library' in diagnostics_df.columns:
            diagnostics_df = diagnostics_df.rename(columns={'Library': 'Model'})
        diagnostics_df = diagnostics_df.sort_values(['Dataset', 'Model'])
        storage.save_summary_dataframe(diagnostics_df, "diagnostics_summary")
        # Create Rich table for diagnostics summary
        diag_table = Table(title="Diagnostics Summary", box=box.ROUNDED)
        for col in diagnostics_df.columns:
            justify = "right" if col not in ["Dataset", "Model"] else "left"
            diag_table.add_column(col, justify=justify, style="cyan" if col == "Dataset" else "yellow" if col == "Model" else None)
        
        for _, row in diagnostics_df.iterrows():
            values = []
            for col in diagnostics_df.columns:
                if pd.isna(row[col]):
                    values.append("N/A")
                elif isinstance(row[col], (int, float)):
                    if col in ["Runtime (s)", "ESS min", "ESS q50", "Size (MB)"]:
                        values.append(f"{row[col]:.1f}")
                    elif col == "R-hat max":
                        values.append(f"{row[col]:.2f}")
                    else:
                        values.append(str(int(row[col])))
                else:
                    values.append(str(row[col]))
            diag_table.add_row(*values)
        
        console.print()
        console.print(diag_table)
    
    # METRICS SUMMARY SECTION
    console.print()
    console.rule("[bold cyan]METRICS SUMMARY[/bold cyan]")
    
    # Save in-sample fit error summary
    if all_performance_rows:
        performance_df = evaluation.create_performance_summary(
            all_performance_rows,
            dataset_names
        )
        # Ensure consistent column naming - rename Library to Model if it exists
        if 'Library' in performance_df.columns:
            performance_df = performance_df.rename(columns={'Library': 'Model'})
        # Now sort by Dataset and Model
        if 'Dataset' in performance_df.columns and 'Model' in performance_df.columns:
            performance_df = performance_df.sort_values(['Dataset', 'Model'])
        storage.save_summary_dataframe(performance_df, "insample_fit_metrics")
        # Create Rich table for in-sample fit error metrics
        perf_table = Table(title="In-sample Fit Error Metrics", box=box.ROUNDED)
        for col in performance_df.columns:
            if col == "Dataset":
                perf_table.add_column(col, style="cyan")
            elif col == "Model":
                perf_table.add_column(col, style="yellow")
            elif col in ["Geo", "Metric"]:
                perf_table.add_column(col, justify="left")
            else:
                perf_table.add_column(col, justify="right")
        
        for _, row in performance_df.iterrows():
            values = []
            for col in performance_df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            perf_table.add_row(*values)
        
        console.print()
        console.print(perf_table)
    
    # Save channel contribution metrics
    if all_channel_contribution_dfs:
        # Combine all per-channel DataFrames
        channel_metrics_df = pd.concat(all_channel_contribution_dfs, ignore_index=True)
        # Sort by Dataset and Model
        channel_metrics_df = channel_metrics_df.sort_values(['Dataset', 'Model'])
        storage.save_summary_dataframe(channel_metrics_df, "channel_contribution_metrics")
        # Create Rich table for channel contributions (showing sample)
        channel_table = Table(
            title=f"Channel Contribution Metrics (showing 10 of {len(channel_metrics_df)} rows)",
            box=box.ROUNDED
        )
        for col in channel_metrics_df.columns:
            if col == "Dataset":
                channel_table.add_column(col, style="cyan")
            elif col == "Model":
                channel_table.add_column(col, style="yellow")
            elif col == "Channel":
                channel_table.add_column(col, style="green")
            elif col == "Region":
                channel_table.add_column(col, style="magenta")
            else:
                channel_table.add_column(col, justify="right")
        
        for _, row in channel_metrics_df.head(10).iterrows():
            values = []
            for col in channel_metrics_df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            channel_table.add_row(*values)
        
        console.print()
        console.print(channel_table)
        
        # Generate channel contribution plots
        visualization.plot_channel_contribution_distributions(channel_metrics_df)
        visualization.plot_channel_metrics_comparison(channel_metrics_df)
        
    # Save channel contribution recovery error metrics
    if all_channel_averages:
        channel_avg_df = pd.DataFrame(all_channel_averages)
        # Sort by Dataset and Model
        channel_avg_df = channel_avg_df.sort_values(['Dataset', 'Model'])
        storage.save_summary_dataframe(channel_avg_df, "channel_contribution_averages")
        # Create Rich table for channel contribution recovery
        avg_table = Table(title="Channel Contribution Recovery", box=box.ROUNDED)
        for col in channel_avg_df.columns:
            if col == "Dataset":
                avg_table.add_column(col, style="cyan")
            elif col == "Model":
                avg_table.add_column(col, style="yellow")
            else:
                avg_table.add_column(col, justify="right")
        
        for _, row in channel_avg_df.iterrows():
            values = []
            for col in channel_avg_df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            avg_table.add_row(*values)
        
        console.print()
        console.print(avg_table)
    
    # Save parameter counts summary
    if all_parameter_counts:
        param_df = parameter_counter.create_parameter_summary(all_parameter_counts)
        # Sort by Dataset and Model
        param_df = param_df.sort_values(['Dataset', 'Model'])
        storage.save_summary_dataframe(param_df, "parameter_counts")
        
        # Create Rich table for parameter counts
        param_table = Table(title="Model Parameter Counts", box=box.ROUNDED)
        
        for col in param_df.columns:
            if col == "Dataset":
                param_table.add_column(col, style="cyan")
            elif col == "Model":
                param_table.add_column(col, style="yellow")
            elif col == "total":
                param_table.add_column(col, style="bold green", justify="right")
            else:
                param_table.add_column(col, justify="right")
        
        for _, row in param_df.iterrows():
            values = []
            for col in param_df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif col == "total":
                    values.append(f"[bold green]{int(val)}[/bold green]")
                elif isinstance(val, (int, float)):
                    values.append(str(int(val)))
                else:
                    values.append(str(val))
            param_table.add_row(*values)
        
        console.print()
        console.print(param_table)
    
    # Save Bayesian metrics summary (if available)
    if any('bayesian_metrics' in result for result in all_results.values()):
        console.print()
        console.rule("[bold cyan]BAYESIAN METRICS SUMMARY[/bold cyan]")
        
        # Collect all Bayesian results
        bayesian_revenue_results = {}
        bayesian_contrib_results = {}
        
        for dataset_name, dataset_results in all_results.items():
            if 'bayesian_metrics' in dataset_results:
                bayesian_revenue_results[dataset_name] = {}
                bayesian_contrib_results[dataset_name] = {}
                
                for model_name, model_metrics in dataset_results['bayesian_metrics'].items():
                    # Store revenue metrics (averaged across geos)
                    revenue_metrics = model_metrics['revenue']
                    avg_revenue_metrics = {}
                    
                    for metric_name in ['R²', 'MAPE (%)', 'SRMSE', 'Durbin-Watson', 'Bias']:
                        # Collect metric values across geos
                        geo_values = []
                        for geo_metrics in revenue_metrics.values():
                            if metric_name in geo_metrics:
                                geo_values.append(geo_metrics[metric_name]['mean'])
                        
                        if geo_values:
                            # Use first geo's full stats as template
                            first_geo = list(revenue_metrics.keys())[0]
                            avg_revenue_metrics[metric_name] = dict(revenue_metrics[first_geo][metric_name])
                            avg_revenue_metrics[metric_name]['mean'] = np.nanmean(geo_values)
                    
                    # Also average the posterior mean MAPE
                    mape_pm_values = []
                    for geo_metrics in revenue_metrics.values():
                        if 'MAPE_posterior_mean (%)' in geo_metrics:
                            mape_pm_values.append(geo_metrics['MAPE_posterior_mean (%)'])
                    if mape_pm_values:
                        avg_revenue_metrics['MAPE_posterior_mean (%)'] = np.nanmean(mape_pm_values)
                    
                    bayesian_revenue_results[dataset_name][model_name] = avg_revenue_metrics
                    
                    # Store aggregated contribution metrics
                    bayesian_contrib_results[dataset_name][model_name] = model_metrics['contributions_aggregated']
        
        # Create Bayesian in-sample fit error table
        if bayesian_revenue_results:
            revenue_table = Table(title="Bayesian In-sample Fit Error Metrics", box=box.ROUNDED)
            revenue_table.add_column("Dataset", style="cyan")
            revenue_table.add_column("Model", style="yellow")
            revenue_table.add_column("R² (mean ± std) 90% HDI", justify="right")
            revenue_table.add_column("MAPE (%) Bayesian", justify="right")  # Proper Bayesian with uncertainty
            revenue_table.add_column("MAPE (%) Posterior Mean", justify="right")  # Traditional-style for comparison
            revenue_table.add_column("Durbin-Watson (mean ± std) 90% HDI", justify="right")
            
            # Sort by dataset and model for consistent display
            sorted_datasets = sorted(bayesian_revenue_results.keys())
            for dataset_name in sorted_datasets:
                models = bayesian_revenue_results[dataset_name]
                sorted_models = sorted(models.keys())
                for model_name in sorted_models:
                    metrics = models[model_name]
                    r2_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('R²', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 3
                    )
                    # Show both MAPE calculations
                    mape_bayesian_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('MAPE (%)', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 1
                    )
                    mape_posterior_mean = metrics.get('MAPE_posterior_mean (%)', np.nan)
                    mape_pm_str = f"{mape_posterior_mean:.1f}" if not np.isnan(mape_posterior_mean) else "N/A"
                    dw_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('Durbin-Watson', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 3
                    )
                    revenue_table.add_row(dataset_name, model_name, r2_str, mape_bayesian_str, mape_pm_str, dw_str)
            
            console.print()
            console.print(revenue_table)
            
            # Save to CSV
            revenue_df = pd.DataFrame([
                {
                    'Dataset': dataset,
                    'Model': model,
                    'R2_mean': metrics.get('R²', {}).get('mean', np.nan),
                    'R2_std': metrics.get('R²', {}).get('std', np.nan),
                    'R2_hdi_lower': metrics.get('R²', {}).get('hdi_lower', np.nan),
                    'R2_hdi_upper': metrics.get('R²', {}).get('hdi_upper', np.nan),
                    'MAPE_bayesian_mean': metrics.get('MAPE (%)', {}).get('mean', np.nan),
                    'MAPE_bayesian_std': metrics.get('MAPE (%)', {}).get('std', np.nan),
                    'MAPE_bayesian_hdi_lower': metrics.get('MAPE (%)', {}).get('hdi_lower', np.nan),
                    'MAPE_bayesian_hdi_upper': metrics.get('MAPE (%)', {}).get('hdi_upper', np.nan),
                    'MAPE_posterior_mean': metrics.get('MAPE_posterior_mean (%)', np.nan),
                    'DW_mean': metrics.get('Durbin-Watson', {}).get('mean', np.nan),
                    'DW_std': metrics.get('Durbin-Watson', {}).get('std', np.nan),
                    'DW_hdi_lower': metrics.get('Durbin-Watson', {}).get('hdi_lower', np.nan),
                    'DW_hdi_upper': metrics.get('Durbin-Watson', {}).get('hdi_upper', np.nan),
                    'Bias_mean': metrics.get('Bias', {}).get('mean', np.nan),
                    'Bias_std': metrics.get('Bias', {}).get('std', np.nan),
                    'SRMSE_mean': metrics.get('SRMSE', {}).get('mean', np.nan),
                    'SRMSE_std': metrics.get('SRMSE', {}).get('std', np.nan)
                }
                for dataset, models in bayesian_revenue_results.items()
                for model, metrics in models.items()
            ])
            storage.save_summary_dataframe(revenue_df, "bayesian_insample_fit_metrics")
        
        # Create channel contribution recovery table
        if bayesian_contrib_results:
            contrib_table = Table(title="Bayesian Channel Contribution Recovery", box=box.ROUNDED)
            contrib_table.add_column("Dataset", style="cyan")
            contrib_table.add_column("Model", style="yellow")
            contrib_table.add_column("Avg Bias (mean ± std) 90% HDI", justify="right")
            contrib_table.add_column("Avg SRMSE (mean ± std) 90% HDI", justify="right")
            contrib_table.add_column("Avg R² (mean ± std) 90% HDI", justify="right")
            contrib_table.add_column("Avg MAPE (%) (mean ± std) 90% HDI", justify="right")
            
            # Sort by dataset and model for consistent display
            sorted_datasets = sorted(bayesian_contrib_results.keys())
            for dataset_name in sorted_datasets:
                models = bayesian_contrib_results[dataset_name]
                sorted_models = sorted(models.keys())
                for model_name in sorted_models:
                    metrics = models[model_name]
                    bias_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('Bias', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 3
                    )
                    srmse_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('SRMSE', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 3
                    )
                    r2_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('R²', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 3
                    )
                    mape_str = bayesian_evaluation.bayesian_metrics.format_metric_with_hdi(
                        metrics.get('MAPE (%)', {'mean': np.nan, 'std': np.nan, 'hdi_lower': np.nan, 'hdi_upper': np.nan}), 1
                    )
                    contrib_table.add_row(dataset_name, model_name, bias_str, srmse_str, r2_str, mape_str)
            
            console.print()
            console.print(contrib_table)
            
            # Save to CSV
            contrib_df = pd.DataFrame([
                {
                    'Dataset': dataset,
                    'Model': model,
                    'Bias_mean': metrics.get('Bias', {}).get('mean', np.nan),
                    'Bias_std': metrics.get('Bias', {}).get('std', np.nan),
                    'Bias_hdi_lower': metrics.get('Bias', {}).get('hdi_lower', np.nan),
                    'Bias_hdi_upper': metrics.get('Bias', {}).get('hdi_upper', np.nan),
                    'SRMSE_mean': metrics.get('SRMSE', {}).get('mean', np.nan),
                    'SRMSE_std': metrics.get('SRMSE', {}).get('std', np.nan),
                    'SRMSE_hdi_lower': metrics.get('SRMSE', {}).get('hdi_lower', np.nan),
                    'SRMSE_hdi_upper': metrics.get('SRMSE', {}).get('hdi_upper', np.nan),
                    'R2_mean': metrics.get('R²', {}).get('mean', np.nan),
                    'R2_std': metrics.get('R²', {}).get('std', np.nan),
                    'R2_hdi_lower': metrics.get('R²', {}).get('hdi_lower', np.nan),
                    'R2_hdi_upper': metrics.get('R²', {}).get('hdi_upper', np.nan),
                    'MAPE_mean': metrics.get('MAPE (%)', {}).get('mean', np.nan),
                    'MAPE_std': metrics.get('MAPE (%)', {}).get('std', np.nan),
                    'MAPE_hdi_lower': metrics.get('MAPE (%)', {}).get('hdi_lower', np.nan),
                    'MAPE_hdi_upper': metrics.get('MAPE (%)', {}).get('hdi_upper', np.nan)
                }
                for dataset, models in bayesian_contrib_results.items()
                for model, metrics in models.items()
            ])
            storage.save_summary_dataframe(contrib_df, "bayesian_contribution_metrics")


def main() -> None:
    """Main entry point for benchmarking script."""
    args = parse_arguments()
    console = Console()
    
    # Create welcome panel
    welcome_text = Text("MMM BENCHMARKING COMPARISON", justify="center", style="bold cyan")
    config_text = f"""[bold]Configuration:[/bold]
• Datasets: {', '.join(args.datasets)}
• Libraries: {', '.join(args.libraries)}"""
    
    if "pymc" in args.libraries:
        config_text += f"\n• Samplers: {', '.join(args.samplers)}"
    
    config_text += f"""\n• Chains: {args.chains}, Draws: {args.draws}, Tune: {args.tune}
• Target Accept: {args.target_accept}, Seed: {args.seed}
• Force Rerun: {args.force_rerun}"""
    
    console.print(Panel.fit(welcome_text, border_style="cyan"))
    console.print(Panel(config_text, title="Settings", border_style="blue"))
    
    # Set up environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    warnings.filterwarnings("ignore", category=UserWarning)
    visualization.setup_plot_style()
    
    # Load datasets
    prepared_datasets = data_loader.load_multiple_datasets(args.datasets, args.seed, console)
    
    # Run benchmarks
    all_results = {}
    
    for i, (data_df, channel_columns, control_columns, truth_df) in enumerate(prepared_datasets):
        dataset_name = args.datasets[i]
        
        console.print()
        console.print(Panel(f"[bold cyan]BENCHMARKING: {dataset_name}[/bold cyan]", 
                           expand=False, border_style="yellow"))
        
        results = run_benchmark_for_dataset(
            dataset_name,
            data_df,
            channel_columns,
            control_columns,
            truth_df,
            args,
            console
        )
        
        all_results[dataset_name] = results
        
        # Clear memory between datasets
        gc.collect()
        console.print("  [green]✓[/green] Memory cleared between datasets")
    
    # Create summary tables and plots
    if not args.plots_only:
        console.print()
        console.print(Panel("[bold yellow]CREATING SUMMARIES[/bold yellow]", 
                           expand=False, border_style="yellow"))
        create_summary_tables(all_results, args.datasets, console)
    
    console.print()
    console.print(Panel("[bold green]BENCHMARKING COMPLETE[/bold green]", 
                       expand=False, border_style="green"))


if __name__ == "__main__":
    main()