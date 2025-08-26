#!/usr/bin/env python
"""Main benchmarking script for MMM parameter recovery comparison."""

import argparse
import gc
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
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with fitted models and metrics
    """
    all_performance_rows = []
    
    # PHASE 1: Fit and save models (if needed) with memory isolation
    if not args.plots_only:
        print("\n=== PHASE 1: FITTING MODELS (ISOLATED) ===")
        
        # Fit Meridian
        if "meridian" in args.libraries:
            if not storage.model_exists(dataset_name, "meridian") or args.force_rerun:
                print("\n--- Meridian ---")
                meridian_result, runtime, ess = model_fitter.fit_meridian(
                    data_df,
                    channel_columns,
                    control_columns,
                    args.chains,
                    args.draws,
                    args.tune,
                    args.target_accept,
                    args.seed
                )
                storage.save_meridian_model(meridian_result, dataset_name, runtime, ess)
                del meridian_result
                gc.collect()
                print("  ✓ Model fitted, saved, and memory cleared")
        
        # Fit PyMC models
        if "pymc" in args.libraries:
            for sampler in args.samplers:
                if model_fitter.should_skip_sampler(sampler, dataset_name):
                    continue
                
                if not storage.model_exists(dataset_name, "pymc", sampler) or args.force_rerun:
                    print(f"\n--- PyMC-Marketing - {sampler} ---")
                    pymc_result, runtime, ess = model_fitter.fit_pymc(
                        data_df,
                        channel_columns,
                        control_columns,
                        sampler,
                        args.chains,
                        args.draws,
                        args.tune,
                        args.target_accept,
                        args.seed
                    )
                    storage.save_pymc_model(pymc_result, dataset_name, sampler, runtime, ess)
                    del pymc_result
                    gc.collect()
                    print("  ✓ Model fitted, saved, and memory cleared")
    
    # PHASE 2: Load all models for evaluation
    print("\n=== PHASE 2: EVALUATING MODELS ===")
    results = {}
    
    # Load and evaluate Meridian
    if "meridian" in args.libraries and storage.model_exists(dataset_name, "meridian"):
        print("\n--- Evaluating Meridian ---")
        meridian_result, runtime, ess = storage.load_meridian_model(dataset_name)
        results["Meridian"] = (meridian_result, runtime, ess)
        
        perf_rows = evaluation.evaluate_meridian_fit(meridian_result, data_df)
        for row in perf_rows:
            row["Dataset"] = dataset_name
        all_performance_rows.extend(perf_rows)
        
        visualization.plot_meridian_posterior_predictive(
            meridian_result, data_df, dataset_name
        )
    
    # Load and evaluate PyMC models  
    if "pymc" in args.libraries:
        for sampler in args.samplers:
            if model_fitter.should_skip_sampler(sampler, dataset_name):
                continue
                
            if storage.model_exists(dataset_name, "pymc", sampler):
                print(f"\n--- Evaluating PyMC-Marketing - {sampler} ---")
                pymc_result, runtime, ess = storage.load_pymc_model(dataset_name, sampler)
                results[f"PyMC-Marketing - {sampler}"] = (pymc_result, runtime, ess)
                
                perf_rows = evaluation.evaluate_pymc_fit(pymc_result, data_df, sampler)
                for row in perf_rows:
                    row["Dataset"] = dataset_name
                all_performance_rows.extend(perf_rows)
                
                visualization.plot_pymc_posterior_predictive(
                    pymc_result, data_df, dataset_name, sampler
                )
    
    results["performance"] = all_performance_rows
    
    # Generate comparison plot if both models exist
    if "Meridian" in results and "PyMC-Marketing - nutpie" in results:
        print("\n--- Generating Model Comparison Plot ---")
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
    
    # Calculate and save ESS/s (Effective Sample Size per second)
    if ess_rows and runtime_data:
        # Reshape runtime data for merging
        runtime_melted = runtime_df.reset_index().melt(
            id_vars="Dataset",
            var_name="Sampler",
            value_name="Runtime"
        )
        
        # Reshape ESS data to wide format
        ess_wide = ess_df.pivot_table(
            index=["Dataset", "Sampler"],
            columns="ESS",
            values="value"
        ).reset_index()
        
        # Merge ESS and runtime data
        ess_with_runtime = ess_wide.merge(
            runtime_melted,
            on=["Dataset", "Sampler"],
            how="inner"
        )
        
        # Calculate ESS/s for each metric
        ess_per_second_rows = []
        for _, row in ess_with_runtime.iterrows():
            for metric in ["min", "q10", "q50", "q90"]:
                if metric in row and pd.notna(row[metric]) and row["Runtime"] > 0:
                    ess_per_second_rows.append({
                        "Dataset": row["Dataset"],
                        "Sampler": row["Sampler"],
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
                index=["Dataset", "Sampler"],
                columns="Metric",
                values="ESS_per_s"
            )
            
            # Sort by median ESS/s (q50_per_s)
            if "q50_per_s" in ess_per_s_pivot.columns:
                ess_per_s_pivot = ess_per_s_pivot.sort_values("q50_per_s", ascending=False)
            
            print("\n=== ESS/s (Efficiency) Summary ===")
            print(ess_per_s_pivot.round(2))
    
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
        
        # Clear memory between datasets
        gc.collect()
        print("  ✓ Memory cleared between datasets")
    
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