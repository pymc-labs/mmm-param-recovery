"""Model fitting and sampling functions for benchmarking."""

import time
from copy import deepcopy
from typing import Dict, Optional, Any, Tuple
import pandas as pd
from meridian.model import model
from pymc_marketing.mmm.multidimensional import MMM
from rich.console import Console
from . import diagnostics


def fit_meridian(
    data_df: pd.DataFrame,
    channel_columns: list,
    control_columns: list,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int,
    console: Optional[Console] = None
) -> Tuple[model.Meridian, float, Dict[str, Optional[float]]]:
    """Fit Meridian model with specified sampling parameters.
    
    Builds a fresh model from scratch and fits it, ensuring fair timing
    that includes model building and data preparation.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset
    channel_columns : list
        Channel column names
    control_columns : list
        Control column names
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
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    Tuple[model.Meridian, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    if console is None:
        console = Console()
    console.print(f"  Fitting Meridian with {n_chains} chains, {n_draws} draws, {n_tune} tune steps")
    
    # Import here to avoid circular dependency
    from . import model_builder

    # Start timing BEFORE building the model to include all preparation time
    start = time.perf_counter()
    
    # Build a fresh model from scratch (includes all data preparation)
    meridian_model = model_builder.build_meridian_model(
        data_df, channel_columns, control_columns
    )
    
    # Sample posterior
    meridian_model.sample_posterior(
        n_chains=n_chains,
        n_adapt=int(n_tune / 2),
        n_burnin=int(n_tune / 2),
        n_keep=n_draws,
        seed=(seed, seed),
        dual_averaging_kwargs={"target_accept_prob": target_accept}
    )
    
    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(meridian_model.inference_data)
    
    console.print(f"  [green]✓[/green] Meridian: {runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}")
    
    return meridian_model, runtime, ess


def fit_pymc(
    data_df: pd.DataFrame,
    channel_columns: list,
    control_columns: list,
    sampler: str,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int,
    console: Optional[Console] = None
) -> Tuple[MMM, float, Dict[str, Optional[float]]]:
    """Fit PyMC-Marketing model with specified sampler.
    
    Builds a fresh model from scratch and fits it, ensuring fair timing
    that includes model building and compilation.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset
    channel_columns : list
        Channel column names
    control_columns : list
        Control column names
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
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    Tuple[MMM, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    if console is None:
        console = Console()
    console.print(f"  Fitting PyMC-Marketing with {sampler}, {n_chains} chains, {n_draws} draws, {n_tune} tune steps")
    
    x = data_df.drop(columns=["y"])
    y = data_df["y"]
    
    kwargs = {}
    if sampler == "nutpie":
        kwargs = {"nuts_sampler_kwargs": {"backend": "jax", "gradient_backend": "jax"}}
    
    # Import here to avoid circular dependency
    from . import model_builder

    # Start timing BEFORE building the model to include compilation time
    start = time.perf_counter()
    
    
    # Build a fresh model from scratch (includes model compilation)
    pymc_model = model_builder.build_pymc_model(
        data_df, channel_columns, control_columns
    )
    
    # Fit the model
    pymc_model.fit(
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
    
    pymc_model.sample_posterior_predictive(
        X=x,
        extend_idata=True,
        combined=True,
        random_seed=seed
    )
    
    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(pymc_model.idata)
    
    console.print(f"  [green]✓[/green] PyMC-Marketing - {sampler}: {runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}")
    
    return pymc_model, runtime, ess


def should_skip_sampler(sampler: str, dataset_name: str, console: Optional[Console] = None) -> bool:
    """Check if a sampler should be skipped for a dataset.
    
    Parameters
    ----------
    sampler : str
        Sampler name
    dataset_name : str
        Dataset name
    console : Optional[Console]
        Rich console for output
        
    Returns
    -------
    bool
        True if sampler should be skipped
    """
    if console is None:
        console = Console()
    if sampler in ["blackjax", "numpyro"] and "large" in dataset_name:
        console.print(f"  [yellow]⚠[/yellow] Skipping {sampler} for large dataset (memory constraints)")
        return True
    return False