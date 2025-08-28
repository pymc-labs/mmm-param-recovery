"""Parameter counting functions for model comparison."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def count_pymc_parameters(model: Any) -> Dict[str, int]:
    """Count parameters in a PyMC-Marketing MMM model.
    
    Parameters
    ----------
    model : MMM
        PyMC-Marketing MMM model with fitted posterior
        
    Returns
    -------
    Dict[str, int]
        Parameter counts by category
    """
    counts = {
        "adstock": 0,
        "media_response": 0,
        "control": 0,
        "baseline": 0,
        "time_effects": 0,
        "noise": 0,
        "total": 0
    }
    
    if not hasattr(model, 'idata') or model.idata is None:
        return counts
    
    posterior = model.idata.posterior
    
    # Count adstock/carryover parameters
    if "adstock_alpha" in posterior:
        # Shape: (chain, draw, channel)
        counts["adstock"] = posterior["adstock_alpha"].shape[-1]
    
    # Count media response parameters (all saturation parameters)
    # These jointly determine the media response curves
    media_response_params = 0
    
    if "saturation_beta" in posterior:
        # Population-level beta: one per channel
        media_response_params += posterior["saturation_beta"].shape[-1]
    
    if "saturation_lam" in posterior:
        # Population-level lambda: one per channel
        media_response_params += posterior["saturation_lam"].shape[-1]
    
    if "saturation_sigma" in posterior:
        # Hierarchical sigma: (channel x geo)
        shape = posterior["saturation_sigma"].shape
        media_response_params += shape[-2] * shape[-1]  # channels * geos
    
    if "saturation_sigma_mu" in posterior:
        # Population mean for sigma: one per channel
        media_response_params += posterior["saturation_sigma_mu"].shape[-1]
    
    if "saturation_sigma_sigma" in posterior:
        # Population std for sigma: single parameter
        media_response_params += 1
    
    counts["media_response"] = media_response_params
    
    # Count control effect parameters
    if "gamma_control" in posterior:
        # Shape: (chain, draw, control)
        counts["control"] = posterior["gamma_control"].shape[-1]
    
    # Count baseline/intercept parameters
    if "intercept_contribution" in posterior:
        # Shape: (chain, draw, geo)
        counts["baseline"] = posterior["intercept_contribution"].shape[-1]
    
    # Count time-varying effects (seasonality)
    if "gamma_fourier" in posterior:
        # Shape: (chain, draw, geo, fourier_mode)
        shape = posterior["gamma_fourier"].shape
        counts["time_effects"] = shape[-2] * shape[-1]  # geo * fourier_modes
    
    # Count noise parameters
    if "y_sigma" in posterior:
        # Shape: (chain, draw, geo)
        counts["noise"] = posterior["y_sigma"].shape[-1]
    
    # Calculate total
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    
    return counts


def count_meridian_parameters(model: Any) -> Dict[str, int]:
    """Count parameters in a Meridian model.
    
    Parameters
    ----------
    model : Meridian
        Meridian model with fitted posterior
        
    Returns
    -------
    Dict[str, int]
        Parameter counts by category
    """
    counts = {
        "adstock": 0,
        "media_response": 0,
        "control": 0,
        "baseline": 0,
        "time_effects": 0,
        "noise": 0,
        "total": 0
    }
    
    if not hasattr(model, 'inference_data') or model.inference_data is None:
        return counts
    
    posterior = model.inference_data.posterior
    
    # Count adstock/carryover parameters
    if "alpha_m" in posterior:
        # Shape: (chain, draw, media_channel)
        counts["adstock"] = posterior["alpha_m"].shape[-1]
    
    # Count media response parameters (coefficients + saturation + noise)
    # All these parameters jointly determine the media response curves
    media_response_params = 0
    
    if "beta_m" in posterior:
        # Population-level media effects: one per channel
        media_response_params += posterior["beta_m"].shape[-1]
    
    if "beta_gm" in posterior:
        # Geo-specific media effects: (geo x media_channel)
        shape = posterior["beta_gm"].shape
        media_response_params += shape[-2] * shape[-1]  # geos * channels
    
    if "ec_m" in posterior:
        # Half-saturation points (Hill): one per channel
        media_response_params += posterior["ec_m"].shape[-1]
    
    if "slope_m" in posterior:
        # Hill slopes: one per channel
        media_response_params += posterior["slope_m"].shape[-1]
    
    if "eta_m" in posterior:
        # Media noise parameters: one per channel
        media_response_params += posterior["eta_m"].shape[-1]
    
    counts["media_response"] = media_response_params
    
    # Count control effect parameters
    control_params = 0
    
    if "gamma_c" in posterior:
        # Population-level control effects
        control_params += posterior["gamma_c"].shape[-1]
    
    if "gamma_gc" in posterior:
        # Geo-specific control effects
        shape = posterior["gamma_gc"].shape
        control_params += shape[-2] * shape[-1]  # geos * controls
    
    if "xi_c" in posterior:
        # Control noise parameters
        control_params += posterior["xi_c"].shape[-1]
    
    counts["control"] = control_params
    
    # Count baseline parameters
    if "tau_g" in posterior:
        # Geo-specific baselines
        counts["baseline"] = posterior["tau_g"].shape[-1]
    
    # Count time-varying effects (trend via splines)
    time_effects_params = 0
    
    if "knot_values" in posterior:
        # Spline knot values for trend
        time_effects_params += posterior["knot_values"].shape[-1]
    
    if "mu_t" in posterior:
        # Time-varying trend values
        # Note: This is derived from knots, so we don't count it separately
        pass  # Don't double-count
    
    counts["time_effects"] = time_effects_params
    
    # Count noise parameters
    if "sigma" in posterior:
        # Single observation noise parameter
        counts["noise"] = 1
    
    # Calculate total
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    
    return counts


def categorize_parameters(
    model: Any,
    library: str
) -> Dict[str, int]:
    """Count and categorize parameters for a model.
    
    Parameters
    ----------
    model : Any
        Fitted model (either MMM or Meridian)
    library : str
        Library name ("pymc" or "meridian")
        
    Returns
    -------
    Dict[str, int]
        Parameter counts by category
    """
    if library.lower() == "meridian":
        return count_meridian_parameters(model)
    else:
        return count_pymc_parameters(model)


def create_parameter_summary(
    all_parameter_counts: list
) -> pd.DataFrame:
    """Create a summary DataFrame of parameter counts.
    
    Parameters
    ----------
    all_parameter_counts : list
        List of dictionaries with parameter counts
        
    Returns
    -------
    pd.DataFrame
        Summary table of parameter counts
    """
    df = pd.DataFrame(all_parameter_counts)
    
    # Ensure columns are in a logical order
    column_order = [
        "Dataset", "Model", 
        "adstock", "media_response",
        "control", "baseline", 
        "time_effects", "noise", "total"
    ]
    
    # Only include columns that exist
    columns = [col for col in column_order if col in df.columns]
    df = df[columns]
    
    # Fill NaN with 0 for missing parameter types
    numeric_columns = [col for col in df.columns if col not in ["Dataset", "Model"]]
    df[numeric_columns] = df[numeric_columns].fillna(0).astype(int)
    
    return df