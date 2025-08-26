"""Model fitting and sampling functions for benchmarking."""

import time
from copy import deepcopy
from typing import Dict, Optional, Any, Tuple
import pandas as pd
from meridian.model import model
from pymc_marketing.mmm.multidimensional import MMM
from . import diagnostics


def fit_meridian(
    meridian_model: model.Meridian,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int
) -> Tuple[model.Meridian, float, Dict[str, Optional[float]]]:
    """Fit Meridian model with specified sampling parameters.
    
    Parameters
    ----------
    meridian_model : model.Meridian
        Meridian model to fit
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    n_tune : int
        Number of tuning samples
    target_accept : float
        Target acceptance probability
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[model.Meridian, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    print(f"  Fitting Meridian with {n_chains} chains, {n_draws} draws, {n_tune} tune steps")
    
    model_copy = deepcopy(meridian_model)
    start = time.perf_counter()
    
    model_copy.sample_posterior(
        n_chains=n_chains,
        n_adapt=int(n_tune / 2),
        n_burnin=int(n_tune / 2),
        n_keep=n_draws,
        seed=(seed, seed),
        dual_averaging_kwargs={"target_accept_prob": target_accept}
    )
    
    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(model_copy.inference_data)
    
    print(f"  ✓ Meridian: {runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}")
    
    return model_copy, runtime, ess


def fit_pymc(
    pymc_model: MMM,
    data_df: pd.DataFrame,
    sampler: str,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int
) -> Tuple[MMM, float, Dict[str, Optional[float]]]:
    """Fit PyMC-Marketing model with specified sampler.
    
    Parameters
    ----------
    pymc_model : MMM
        PyMC-Marketing model to fit
    data_df : pd.DataFrame
        Dataset
    sampler : str
        Sampler name ('pymc', 'blackjax', 'numpyro', 'nutpie')
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    n_tune : int
        Number of tuning samples
    target_accept : float
        Target acceptance probability
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[MMM, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    print(f"  Fitting PyMC-Marketing with {sampler}, {n_chains} chains, {n_draws} draws, {n_tune} tune steps")
    
    model_copy = deepcopy(pymc_model)
    x = data_df.drop(columns=["y"])
    y = data_df["y"]
    
    kwargs = {}
    if sampler == "nutpie":
        kwargs = {"nuts_sampler_kwargs": {"backend": "jax", "gradient_backend": "jax"}}
    
    start = time.perf_counter()
    
    model_copy.fit(
        X=x,
        y=y,
        chains=n_chains,
        draws=n_draws,
        tune=n_tune,
        target_accept=target_accept,
        random_seed=seed,
        nuts_sampler=sampler,
        **kwargs
    )
    
    model_copy.sample_posterior_predictive(
        X=x,
        extend_idata=True,
        combined=True,
        random_seed=seed
    )
    
    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(model_copy.idata)
    
    print(f"  ✓ PyMC-Marketing - {sampler}: {runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}")
    
    return model_copy, runtime, ess


def should_skip_sampler(sampler: str, dataset_name: str) -> bool:
    """Check if a sampler should be skipped for a dataset.
    
    Parameters
    ----------
    sampler : str
        Sampler name
    dataset_name : str
        Dataset name
        
    Returns
    -------
    bool
        True if sampler should be skipped
    """
    if sampler in ["blackjax", "numpyro"] and "large" in dataset_name:
        print(f"  ⚠ Skipping {sampler} for large dataset (memory constraints)")
        return True
    return False