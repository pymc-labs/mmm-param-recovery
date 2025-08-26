#!/usr/bin/env python
"""Main benchmarking script for MMM parameter recovery comparison."""

import argparse
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmm_param_recovery.benchmarking import (
    data_loader,
    model_builder,
    model_fitter,
    diagnostics,
    evaluation,
    visualization,
    storage
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
    
    parser.set_defaults(force_rerun=False)
    
    return parser.parse_args()


def run_benchmark_for_dataset(
    dataset_name: str,
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str],
    truth_df: pd.DataFrame,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Run benchmark for a single dataset.
    
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
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with fitted models and metrics
    """
    results = {}
    all_performance_rows = []
    
    # Meridian
    if "meridian" in args.libraries:
        print("\n--- Meridian ---")
        
        if not args.force_rerun and storage.model_exists(dataset_name, "meridian"):
            meridian_result, runtime, ess = storage.load_meridian_model(dataset_name)
        else:
            meridian_model = model_builder.build_meridian_model(
                data_df, channel_columns, control_columns
            )
            meridian_result, runtime, ess = model_fitter.fit_meridian(
                meridian_model,
                args.chains,
                args.draws,
                args.tune,
                args.target_accept,
                args.seed
            )
            storage.save_meridian_model(meridian_result, dataset_name, runtime, ess)
        
        results["Meridian"] = (meridian_result, runtime, ess)
        
        # Evaluate and plot
        perf_rows = evaluation.evaluate_meridian_fit(meridian_result, data_df)
        for row in perf_rows:
            row["Dataset"] = dataset_name
        all_performance_rows.extend(perf_rows)
        
        if not args.plots_only:
            visualization.plot_meridian_posterior_predictive(
                meridian_result, data_df, dataset_name
            )
    
    # PyMC-Marketing
    if "pymc" in args.libraries:
        pymc_model_template = model_builder.build_pymc_model(
            data_df, channel_columns, control_columns
        )
        
        for sampler in args.samplers:
            if model_fitter.should_skip_sampler(sampler, dataset_name):
                continue
            
            print(f"\n--- PyMC-Marketing - {sampler} ---")
            
            if not args.force_rerun and storage.model_exists(dataset_name, "pymc", sampler):
                pymc_result, runtime, ess = storage.load_pymc_model(dataset_name, sampler)
            else:
                pymc_result, runtime, ess = model_fitter.fit_pymc(
                    pymc_model_template,
                    data_df,
                    sampler,
                    args.chains,
                    args.draws,
                    args.tune,
                    args.target_accept,
                    args.seed
                )
                storage.save_pymc_model(pymc_result, dataset_name, sampler, runtime, ess)
            
            results[f"PyMC-Marketing - {sampler}"] = (pymc_result, runtime, ess)
            
            # Evaluate and plot
            perf_rows = evaluation.evaluate_pymc_fit(pymc_result, data_df, sampler)
            for row in perf_rows:
                row["Dataset"] = dataset_name
            all_performance_rows.extend(perf_rows)
            
            if not args.plots_only:
                visualization.plot_pymc_posterior_predictive(
                    pymc_result, data_df, dataset_name, sampler
                )
    
    results["performance"] = all_performance_rows
    return results


def create_summary_tables(
    all_results: Dict[str, Dict[str, Any]],
    dataset_names: List[str]
) -> None:
    """Create and save summary tables.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        All results keyed by dataset name
    dataset_names : List[str]
        List of dataset names
    """
    # Runtime summary
    runtime_data = {"Dataset": dataset_names}
    ess_rows = []
    all_diagnostics_rows = []
    all_performance_rows = []
    
    for dataset_name in dataset_names:
        dataset_results = all_results[dataset_name]
        
        # Collect runtime data
        for key, value in dataset_results.items():
            if key == "performance":
                all_performance_rows.extend(value)
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
                    "Sampler": key,
                    "ESS": metric_name,
                    "value": metric_value
                })
        
        # Create diagnostics summary
        diag_df = diagnostics.create_diagnostics_summary(
            {k: v for k, v in dataset_results.items() if k != "performance"},
            dataset_name
        )
        all_diagnostics_rows.append(diag_df)
    
    # Save runtime summary
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df.set_index("Dataset", inplace=True)
    storage.save_summary_dataframe(runtime_df, "runtime_comparison")
    print("\n=== Runtime Summary ===")
    print(runtime_df.round(1))
    
    # Save ESS summary
    ess_df = pd.DataFrame(ess_rows)
    storage.save_summary_dataframe(ess_df, "ess_comparison")
    
    ess_pivot = ess_df.pivot_table(
        index=["Dataset", "Sampler"],
        columns="ESS",
        values="value"
    ).round(2)
    print("\n=== ESS Summary ===")
    print(ess_pivot)
    
    # Save diagnostics summary
    if all_diagnostics_rows:
        diagnostics_df = pd.concat(all_diagnostics_rows, ignore_index=True)
        storage.save_summary_dataframe(diagnostics_df, "diagnostics_summary")
        print("\n=== Diagnostics Summary ===")
        print(diagnostics_df.round(2))
    
    # Save performance summary
    if all_performance_rows:
        performance_df = evaluation.create_performance_summary(
            all_performance_rows,
            dataset_names
        )
        storage.save_summary_dataframe(performance_df, "performance_metrics")
        print("\n=== Performance Metrics ===")
        print(performance_df)
    
    # Generate comparison plots
    visualization.plot_runtime_comparison(runtime_df)
    visualization.plot_ess_comparison(ess_df)
    
    if all_performance_rows:
        visualization.plot_performance_metrics(performance_df)
    
    if all_diagnostics_rows:
        visualization.plot_diagnostics_summary(diagnostics_df)


def main() -> None:
    """Main entry point for benchmarking script."""
    args = parse_arguments()
    
    print("=" * 60)
    print("MMM BENCHMARKING COMPARISON")
    print("=" * 60)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Libraries: {', '.join(args.libraries)}")
    if "pymc" in args.libraries:
        print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Chains: {args.chains}, Draws: {args.draws}, Tune: {args.tune}")
    print(f"Target Accept: {args.target_accept}, Seed: {args.seed}")
    print(f"Force Rerun: {args.force_rerun}")
    print("=" * 60)
    
    # Set up environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    warnings.filterwarnings("ignore", category=UserWarning)
    visualization.setup_plot_style()
    
    # Load datasets
    prepared_datasets = data_loader.load_multiple_datasets(args.datasets, args.seed)
    
    # Run benchmarks
    all_results = {}
    
    for i, (data_df, channel_columns, control_columns, truth_df) in enumerate(prepared_datasets):
        dataset_name = args.datasets[i]
        
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING: {dataset_name}")
        print(f"{'=' * 60}")
        
        results = run_benchmark_for_dataset(
            dataset_name,
            data_df,
            channel_columns,
            control_columns,
            truth_df,
            args
        )
        
        all_results[dataset_name] = results
    
    # Create summary tables and plots
    if not args.plots_only:
        print(f"\n{'=' * 60}")
        print("CREATING SUMMARIES")
        print(f"{'=' * 60}")
        create_summary_tables(all_results, args.datasets)
    
    print(f"\n{'=' * 60}")
    print("BENCHMARKING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()