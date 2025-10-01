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

"""Model building functions for Meridian and PyMC-Marketing."""

from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import data_frame_input_data_builder, input_data
from meridian.model import model, prior_distribution, spec
from pymc_marketing.mmm import GeometricAdstock, HillSaturationSigmoid
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.prior import Prior


def calculate_prior_sigma(
    data_df: pd.DataFrame,
    channel_columns: List[str]
) -> np.ndarray:
    """Calculate prior sigma based on spend share.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset with geo and channel columns
    channel_columns : List[str]
        List of channel column names
        
    Returns
    -------
    np.ndarray
        Prior sigma values (shape: n_geos x n_channels)
    """
    n_channels = len(channel_columns)
    sum_spend_geo_channel = data_df.groupby("geo")[channel_columns].sum()
    spend_share = (
        sum_spend_geo_channel.to_numpy() /
        sum_spend_geo_channel.sum(axis=1).to_numpy()[:, None]
    )
    return n_channels * spend_share


def build_meridian_data(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str]
) -> input_data.InputData:
    """Build Meridian InputData object.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset with required columns
    channel_columns : List[str]
        List of channel column names
    control_columns : List[str]
        List of control column names
        
    Returns
    -------
    input_data.InputData
        Meridian input data object
    """
    builder = (
        data_frame_input_data_builder.DataFrameInputDataBuilder(kpi_type='revenue')
        .with_kpi(data_df, kpi_col="y")
        .with_population(data_df)
    )
    
    if control_columns:
        builder = builder.with_controls(data_df, control_cols=control_columns)
    
    builder = builder.with_media(
        data_df,
        media_cols=channel_columns,
        media_spend_cols=channel_columns,
        media_channels=channel_columns,
    )
    
    return builder.build()


def build_meridian_prior(
    built_data: input_data.InputData,
    channel_columns: List[str],
    prior_sigma: np.ndarray
) -> prior_distribution.PriorDistribution:
    """Build Meridian prior distribution.
    
    Parameters
    ----------
    built_data : input_data.InputData
        Meridian input data
    channel_columns : List[str]
        List of channel column names
    prior_sigma : np.ndarray
        Prior sigma values
        
    Returns
    -------
    prior_distribution.PriorDistribution
        Meridian prior distribution
    """
    build_media_channel_args = built_data.get_paid_media_channels_argument_builder()
    
    beta_m = build_media_channel_args(
        **{
            col: (0, float(prior_sigma.mean(axis=0)[i]))
            for i, col in enumerate(channel_columns)
        }
    )
    
    beta_m_mu, beta_m_sigma = zip(*beta_m)
    
    return prior_distribution.PriorDistribution(
        beta_m=tfp.distributions.Normal(
            beta_m_mu, beta_m_sigma, name=constants.BETA_M
        ),
        # Set alpha_m to Beta(1, 3) to match PyMC-Marketing's fast decay prior
        # This gives E[alpha] = 1/(1+3) = 0.25, favoring faster adstock decay
        alpha_m=tfp.distributions.Beta(
            1.0, 3.0, name=constants.ALPHA_M
        )
    )


def build_meridian_model_spec(
    prior: prior_distribution.PriorDistribution,
    n_time: int
) -> spec.ModelSpec:
    """Build Meridian model specification.
    
    Parameters
    ----------
    prior : prior_distribution.PriorDistribution
        Prior distribution
    n_time : int
        Number of time periods
        
    Returns
    -------
    spec.ModelSpec
        Meridian model specification
    """
    
    return spec.ModelSpec(
        prior=prior,
        media_effects_dist='normal',
        hill_before_adstock=False,
        max_lag=8,
        unique_sigma_for_each_geo=True,
        roi_calibration_period=None,
        rf_roi_calibration_period=None,
        baseline_geo=None,
        holdout_id=None,
        control_population_scaling_id=None,
        media_prior_type='coefficient',
        rf_prior_type='coefficient',
        enable_aks=True,
    )


def build_meridian_model(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str]
) -> model.Meridian:
    """Build complete Meridian model.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset with required columns
    channel_columns : List[str]
        List of channel column names
    control_columns : List[str]
        List of control column names
        
    Returns
    -------
    model.Meridian
        Meridian model ready for sampling
    """
    prior_sigma = calculate_prior_sigma(data_df, channel_columns)
    built_data = build_meridian_data(data_df, channel_columns, control_columns)
    prior = build_meridian_prior(built_data, channel_columns, prior_sigma)
    model_spec = build_meridian_model_spec(prior, len(built_data.time))
    
    return model.Meridian(input_data=built_data, model_spec=model_spec)


def build_pymc_saturation(prior_sigma: np.ndarray) -> HillSaturationSigmoid:
    """Build PyMC-Marketing saturation transformation.
    
    Parameters
    ----------
    prior_sigma : np.ndarray
        Prior sigma values
        
    Returns
    -------
    HillSaturationSigmoid
        Saturation transformation object
    """
    return HillSaturationSigmoid(
        priors={
            "sigma": Prior(
                "InverseGamma",
                mu=Prior("HalfNormal", sigma=prior_sigma.mean(axis=0), dims=("channel",)),
                sigma=Prior("HalfNormal", sigma=1.5),
                dims=("channel", "geo")
            ),
            "beta": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
            "lam": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
        },
    )


def build_pymc_model(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str]
) -> MMM:
    """Build PyMC-Marketing MMM model.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset with required columns
    channel_columns : List[str]
        List of channel column names
    control_columns : List[str]
        List of control column names
        
    Returns
    -------
    MMM
        PyMC-Marketing model ready for fitting
    """
    prior_sigma = calculate_prior_sigma(data_df, channel_columns)
    saturation = build_pymc_saturation(prior_sigma)
    
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
        saturation=saturation,
        adstock=GeometricAdstock(
            l_max=8,
            priors={"alpha": Prior("Beta", alpha=1, beta=3, dims=("channel",))},
        ),
        yearly_seasonality=2,
    )
    
    x_train = data_df.drop(columns=["y"])
    y_train = data_df["y"]
    
    mmm.build_model(X=x_train, y=y_train)
    
    contribution_vars = [
        "channel_contribution",
        "intercept_contribution",
        "yearly_seasonality_contribution",
        "y",
    ]
    
    if control_columns:
        contribution_vars.insert(1, "control_contribution")
    
    mmm.add_original_scale_contribution_variable(var=contribution_vars)
    
    return mmm


def sample_priors(
    meridian_model: model.Meridian,
    pymc_model: MMM,
    data_df: pd.DataFrame,
    n_samples: int = 1000
) -> Tuple[Any, Any, Any]:
    """Sample from prior distributions for both models.
    
    Parameters
    ----------
    meridian_model : model.Meridian
        Meridian model
    pymc_model : MMM
        PyMC-Marketing model
    data_df : pd.DataFrame
        Dataset
    n_samples : int
        Number of prior samples
        
    Returns
    -------
    Tuple[Any, Any, Any]
        Meridian prior samples, PyMC prior predictive, PyMC scalers
    """
    meridian_model.sample_prior(n_samples)
    
    x_train = data_df.drop(columns=["y"])
    y_train = data_df["y"]
    
    prior_predictive = pymc_model.sample_prior_predictive(
        X=x_train, y=y_train, samples=n_samples
    )
    scalers = pymc_model.get_scales_as_xarray()
    
    return meridian_model, prior_predictive, scalers
