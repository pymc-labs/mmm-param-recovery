import argparse
## Fit a meridian model with the specfied dataset.
from memory_profiler import profile
from meridian import constants
from meridian.analysis import analyzer, formatter, optimizer, summarizer, visualizer
from meridian.data import (
    data_frame_input_data_builder,
    input_data,
    load,
    test_utils,
)
from meridian.model import model, prior_distribution, spec
import numpy as np
import pandas as pd
import tensorflow_probability as tfp


@profile
def build_data(df: pd.DataFrame, channel_columns: list[str]) -> input_data.InputData:
    control_columns = [col for col in df.columns if len(col.split("-")) == 1 and col.startswith("c")]
    # Start building the data input
    builder = (
        data_frame_input_data_builder.DataFrameInputDataBuilder(kpi_type='revenue')
        .with_kpi(df, kpi_col="y")
        .with_population(df)
    )

    # Add controls only if any are found
    if control_columns:
        builder = builder.with_controls(df, control_cols=control_columns)

    # Add media columns
    builder = builder.with_media(
        df,
        media_cols=channel_columns,
        media_spend_cols=channel_columns,
        media_channels=channel_columns,
    )

    # Finalize the builder and store the result
    return builder.build()


@profile
def make_priors(df: pd.DataFrame, data: input_data.InputData, channel_columns = list[str]) -> prior_distribution.PriorDistribution:
    n_channels = len(channel_columns)

    # Group and sum media spend by geo
    sum_spend_geo_channel = df.groupby("geo")[channel_columns].sum()

    # Calculate spend share
    spend_share = (
        sum_spend_geo_channel.to_numpy() /
        sum_spend_geo_channel.sum(axis=1).to_numpy()[:, None]
    )

    # Calculate prior sigma
    prior_sigma = n_channels * spend_share

    n_time = len(data.time)
    knots = np.arange(0, n_time, 26).tolist()

    build_media_channel_args = data.get_paid_media_channels_argument_builder()

    beta_m = build_media_channel_args(
        **{
            col: (0, float(prior_sigma.mean(axis=0)[i]))
            for i, col in enumerate(channel_columns)
        }
    )

    beta_m_mu, beta_m_sigma = zip(*beta_m)

    return prior_distribution.PriorDistribution(
        beta_m=tfp.distributions.LogNormal(
            beta_m_mu, beta_m_sigma, name=constants.BETA_M
        )
    )

@profile
def make_model(data: input_data.InputData, prior: prior_distribution.PriorDistribution, ) -> model.Meridian:
    n_time = len(data.time)
    knots = np.arange(0, n_time, 26).tolist()

    # TODO: Control this from a config file.
    model_spec = spec.ModelSpec(
        prior=prior,
        media_effects_dist='log_normal',
        hill_before_adstock=False,
        max_lag=8,
        unique_sigma_for_each_geo=True,
        roi_calibration_period=None,
        rf_roi_calibration_period=None,
        knots=knots,
        baseline_geo=None,
        holdout_id=None,
        control_population_scaling_id=None,
        media_prior_type='coefficient',
        rf_prior_type='coefficient',
    )

    return model.Meridian(input_data=data, model_spec=model_spec)

@profile
def fit_model(mmm: model.Meridian) -> model.Meridian:
    # Todo, control sampling parameters from a config file.
    mmm.sample_posterior(
        n_chains=4, 
        n_adapt=1000, 
        n_burnin=500, 
        n_keep=1000
    )
    return mmm

def save_model(mmm: model.Meridian, path: str) -> None:
    model.save_mmm(mmm, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=20250723)
    args = parser.parse_args()

    df = pd.read_csv(f"data/test_data/test_data_{args.preset_name}_{args.seed}.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["population"] = 1
    channel_columns = [col for col in df.columns if col.startswith("x")]

    data = build_data(df, channel_columns)
    priors = make_priors(df, data, channel_columns)
    mmm = make_model(data, priors)

    mmm = fit_model(mmm)
    
    save_model(mmm, f"data/fits/meridian_{args.preset_name}_{args.seed}.pkl")
