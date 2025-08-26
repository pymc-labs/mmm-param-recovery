"""Diagnostics calculation functions for model evaluation."""

from typing import Dict, Optional, Any, List
import numpy as np
import arviz as az
import pandas as pd
from pympler import asizeof


def compute_ess(idata: Any) -> Dict[str, Optional[float]]:
    """Compute ESS statistics from ArviZ InferenceData.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object from model sampling
        
    Returns
    -------
    Dict[str, Optional[float]]
        ESS statistics (min, q10, q50, q90)
    """
    try:
        es = az.ess(idata)
        arrays = []
        
        for v in es.data_vars:
            arr = np.asarray(es[v].values).astype(float)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            arrays.append(arr.ravel())
        
        if not arrays:
            return {'min': None, 'q10': None, 'q50': None, 'q90': None}
        
        values = np.concatenate(arrays)
        return {
            'min': float(np.min(values)),
            'q10': float(np.quantile(values, 0.1)),
            'q50': float(np.quantile(values, 0.5)),
            'q90': float(np.quantile(values, 0.9))
        }
    except Exception as e:
        print(f"  Warning: Could not compute ESS: {e}")
        return {'min': None, 'q10': None, 'q50': None, 'q90': None}


def compute_divergences(idata: Any) -> int:
    """Compute number of divergent transitions.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object from model sampling
        
    Returns
    -------
    int
        Number of divergent transitions
    """
    try:
        if hasattr(idata, 'sample_stats') and hasattr(idata.sample_stats, 'diverging'):
            return int(idata.sample_stats.diverging.sum().item())
    except (AttributeError, Exception):
        pass
    return -1


def compute_rhat(
    idata: Any,
    var_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute R-hat statistics.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object from model sampling
    var_names : Optional[List[str]]
        Variable names to compute R-hat for
        
    Returns
    -------
    Dict[str, float]
        R-hat maximum and count of variables with R-hat > 1.1
    """
    try:
        summary = az.summary(idata, var_names=var_names) if var_names else az.summary(idata)
        rhat_max = float(summary['r_hat'].max())
        rhat_bad_count = int((summary['r_hat'] > 1.1).sum())
        
        return {
            'max': rhat_max,
            'bad_count': rhat_bad_count
        }
    except Exception as e:
        print(f"  Warning: Could not compute R-hat: {e}")
        return {'max': np.nan, 'bad_count': -1}


def compute_model_size_mb(model: Any) -> float:
    """Compute model size in megabytes.
    
    Parameters
    ----------
    model : Any
        Model object
        
    Returns
    -------
    float
        Model size in MB
    """
    try:
        size_bytes = asizeof.asizeof(model)
        return size_bytes / (1024 ** 2)
    except Exception:
        return -1.0


def get_meridian_var_names() -> List[str]:
    """Get Meridian variable names for diagnostics.
    
    Returns
    -------
    List[str]
        List of Meridian variable names
    """
    return [
        "alpha_m", "beta_gm", "beta_m", "ec_m",
        "gamma_c", "gamma_gc", "sigma", "tau_g",
        "xi_c", "knot_values", "mu_t"
    ]


def get_pymc_var_names() -> List[str]:
    """Get PyMC-Marketing variable names for diagnostics.
    
    Returns
    -------
    List[str]
        List of PyMC-Marketing variable names
    """
    return [
        "adstock_alpha", "gamma_control", "gamma_fourier",
        "intercept_contribution", "saturation_beta",
        "saturation_lam", "saturation_sigma", "y_sigma"
    ]


def create_diagnostics_summary(
    models_dict: Dict[str, Any],
    dataset_name: str
) -> pd.DataFrame:
    """Create diagnostics summary dataframe.
    
    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary of fitted models keyed by library-sampler
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    pd.DataFrame
        Diagnostics summary
    """
    rows = []
    
    for key, model_data in models_dict.items():
        if model_data is None:
            continue
            
        model, runtime, ess = model_data
        
        if "meridian" in key.lower():
            var_names = get_meridian_var_names()
            idata = model.inference_data if hasattr(model, 'inference_data') else None
        else:
            var_names = get_pymc_var_names()
            idata = model.idata if hasattr(model, 'idata') else None
        
        if idata is None:
            continue
        
        divergences = compute_divergences(idata)
        rhat = compute_rhat(idata, var_names)
        size_mb = compute_model_size_mb(model)
        
        rows.append({
            'Dataset': dataset_name,
            'Library': key,
            'Runtime (s)': runtime,
            'ESS min': ess.get('min'),
            'ESS q50': ess.get('q50'),
            'Divergences': divergences,
            'R-hat max': rhat['max'],
            'R-hat > 1.1': rhat['bad_count'],
            'Size (MB)': size_mb
        })
    
    return pd.DataFrame(rows)