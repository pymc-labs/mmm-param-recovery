import argparse
from memory_profiler import profile
import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, HillSaturationSigmoid
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)
from pymc_marketing.prior import Prior

def make_saturation(df: pd.DataFrame, channel_columns: list[str]) -> HillSaturationSigmoid:
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
    
    return HillSaturationSigmoid(
        priors = {
            "sigma":  Prior("InverseGamma", mu=1.5, sigma=prior_sigma.T, dims=("channel", "geo")),
        },
    )


def build_model(df: pd.DataFrame) -> MMM:
    channel_columns = [col for col in df.columns if col.startswith("x")]
    control_columns = [col for col in df.columns if len(col.split("-")) == 1 and col.startswith("c")]

    # TODO: Make configurable throught json.
    mmm = MMM(
        date_column="time",
        target_column="y",
        channel_columns=channel_columns,
        control_columns=control_columns,
        dims=("geo",),
        scaling={
            "channel": {"method": "max", "dims": ()},
            "target": {"method": "max", "dims": ()},
        },
        saturation=make_saturation(df, channel_columns),
        adstock=GeometricAdstock(l_max=8),
        yearly_seasonality=2,
    )


    x_train = df.drop(columns=["y"])
    y_train = df["y"]

    mmm.build_model(X=x_train, y=y_train)

    # Base contribution variables
    contribution_vars = [
        "channel_contribution",
        "intercept_contribution",
        "yearly_seasonality_contribution",
        "y",
    ]
    if control_columns:
        contribution_vars += ["control_contribution"]

    mmm.add_original_scale_contribution_variable(var=contribution_vars)

    return mmm


@profile
def fit_model(mmm: MMM, x_train: pd.DataFrame, y_train: pd.Series, rng: np.random.Generator) -> MMM:
    mmm_model.fit(
        X=x_train,
        y=y_train,
        chains=4,
        draws=1000,
        tune=500,
        target_accept=0.95,
        random_seed=rng,
        nuts_sampler = "numpyro",
    )

    mmm_model.sample_posterior_predictive(
        X=x_train,
        extend_idata=True,
        combined=True,
        random_seed=rng,
    )
    return mmm
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=20250723)
    args = parser.parse_args()
    rng: np.random.Generator = np.random.default_rng(seed=args.seed)
    
    df = pd.read_csv(f"data/test_data/test_data_{args.preset_name}_{args.seed}.csv")
    df["time"] = pd.to_datetime(df["time"])
    
    x_train = df.drop(columns=["y"])
    y_train = df["y"]
    # TODO: make configurable through config file.
    mmm_model = build_model(df)
    mmm_model = fit_model(mmm_model, x_train, y_train, rng)
    mmm_model.save(f"data/fits/pymc_marketing_{args.preset_name}_{args.seed}.nc")
