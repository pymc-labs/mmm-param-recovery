"""
Evaluation metrics for MMM model comparison.

This module provides functions to compute bias measures and reconstruction errors
for Media Mix Modeling (MMM) results using arviz inference data objects.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Literal, Optional, Tuple, Union
from meridian.model import model as meridian_model


def compute_error_measure(
    idata: xr.DataArray,
    true_values: xr.DataArray,
    time_dim: str = "date",
    error_measure: Literal["bias", "absolute_error"] = "bias",
) -> xr.DataArray:
    """
    Compute bias measure (mean absolute error) for channel contributions.

    This function computes the bias between predicted and true channel contributions,
    averaging over time but preserving chain and draw dimensions for credible intervals.

    Parameters
    ----------
    idata : xr.DataArray
        DataArray containing posterior samples
    true_values : xr.DataArray
        DataArray containing true values with same dimensions as predictions
    time_dim : str, default="date"
        Name of the time dimension. The bias is averaged over this dimension.

    Returns
    -------
    xr.Dataset
        Dataset containing bias measures with dimensions (chain, draw, ...), where ... are the dimensions of idata, except after averaging over time_dim.
    """
    # Extract predicted values
    error_measure_func = {"absolute_error": np.abs, "bias": lambda x: x}[error_measure]

    pred_values = idata

    # Assert true values include time_dim
    assert time_dim in true_values.dims, "True values must include time_dim"

    # Compute bias (mean absolute error)
    bias = error_measure_func(pred_values - true_values).mean(time_dim)

    # Create dataset with proper coordinates
    bias = xr.DataArray(
        bias.values, coords=bias.coords, attrs=bias.attrs, dims=bias.dims
    )

    return bias


def format_metrics_for_comparison(
    metrics_ds: xr.Dataset, library_name: str = "PyMC-Marketing"
) -> pd.DataFrame:
    """
    Format metrics dataset into a comparison-friendly DataFrame.

    Parameters
    ----------
    metrics_ds : xr.Dataset
        Dataset containing bias and error metrics
    library_name : str, default="PyMC-Marketing"
        Name of the library/model for comparison

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with metrics for comparison
    """
    # Stack chain and draw dimensions
    stacked = metrics_ds.stack(sample=("chain", "draw"))

    # Compute mean values across samples
    mean_metrics = stacked.mean(dim="sample")

    # Convert to DataFrame
    df = mean_metrics.to_dataframe().reset_index()

    # Add library column
    df["Library"] = library_name

    # Pivot to wide format if needed
    if "channel" in df.columns:
        df = df.pivot_table(
            index=["geo", "Library"],
            columns="channel",
            values=["contribution_bias", "contribution_absolute_error"],
            aggfunc="mean",
        ).reset_index()

    return df


def meridian_to_contribution_xr(
    mmm: meridian_model, true_values: xr.DataArray
) -> xr.DataArray:
    from meridian.analysis import analyzer

    analyzer = analyzer.Analyzer(mmm)

    contribution_tensor = analyzer.incremental_outcome(
        aggregate_times=False,
        aggregate_geos=False,
        use_kpi=True,
    )
    return xr.DataArray(
        contribution_tensor,
        dims=("chain", "draw", "geo", "date", "channel"),
        coords={
            "chain": mmm.inference_data.posterior.coords["chain"].values,
            "draw": mmm.inference_data.posterior.coords["draw"].values,
            "geo": mmm.inference_data.posterior.coords["geo"].values
            if len(mmm.inference_data.posterior.coords["geo"].values) > 1
            else true_values.coords["geo"].values,
            "date": pd.to_datetime(
                mmm.inference_data.posterior.coords["time"].values
            ).values,
            "channel": mmm.inference_data.posterior.coords["media_channel"].values,
        },
    )


def pymc_marketing_to_contribution_xr(idata: xr.Dataset) -> xr.DataArray:
    assert "channel_contribution_original_scale" in idata.data_vars, (
        "pymc marketing idata must contain channel_contribution_original_scale"
    )
    return idata["channel_contribution_original_scale"]


if __name__ == "__main__":
    import argparse
    import arviz as az
    from pymc_marketing.mmm.multidimensional import MMM

    parser = argparse.ArgumentParser(description="Evaluate MMM model performance")
    parser.add_argument(
        "--preset_name",
        type=str,
        required=True,
        help="Name of the preset/dataset to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed used for data generation (default: 20250723)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["both", "meridian", "pymc_marketing"],
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="data/results",
        help="Path to save evaluation results (default: data/results)",
    )

    args = parser.parse_args()

    true_values = xr.open_dataset(
        f"data/test_data/ground_truth_{args.preset_name}_{args.seed}.nc"
    )["channel_contribution_original_scale"]

    contribution_data: list[xr.DataArray] = []
    # Load the appropriate model based on model_type
    if args.model_type in ["meridian", "both"]:
        model_path = f"data/fits/meridian_{args.preset_name}_{args.seed}.pkl"
        mmm = meridian_model.load_mmm(model_path)
        # Convert to xarray format for evaluation
        contribution_data.append(meridian_to_contribution_xr(mmm, true_values))
    if args.model_type in ["pymc_marketing", "both"]:
        idata = MMM.load(
            f"data/fits/pymc_marketing_{args.preset_name}_{args.seed}.nc"
        ).idata.posterior
        contribution_data.append(pymc_marketing_to_contribution_xr(idata))

    models = (
        ["meridian", "pymc_marketing"]
        if args.model_type == "both"
        else [args.model_type]
    )
    # Concatenate contribution data along the "model" dimension
    contribution_xr = xr.concat(contribution_data, dim=xr.Variable("model", models))

    # Compute summary statistics using arviz
    evaluation_results = az.summary(
        contribution_xr.mean("date"), kind="stats", fmt="xarray"
    )
    output_file = f"{args.output_path}/results_{args.preset_name}_{args.seed}.nc"
    evaluation_results.to_netcdf(output_file)
    print(f"Saved evaluation results to {output_file}")
    print(
        f"Loaded {args.model_type if args.model_type != 'both' else 'both models'} for preset {args.preset_name} with seed {args.seed}"
    )
    print(
        f"Summary of results:\n{evaluation_results.mean(['channel', 'geo']).to_dataframe().reset_index().pivot(columns='metric', values='x', index='model')}"
    )
