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


def compute_model_size_mb(dataset_name: str = None, library: str = None, sampler: str = None) -> float:
    """Compute model size in megabytes.
    
    Parameters
    ----------
    dataset_name : str, optional
        Dataset name for finding saved model file
    library : str, optional  
        Library name ('meridian' or 'pymc')
    sampler : str, optional
        Sampler name for PyMC models
        
    Returns
    -------
    float
        Model size in MB
    """
    from pathlib import Path
    
    # Convert bytes to megabytes
    BYTES_TO_MB = 1024 ** 2
    
    # Try to get file size for Meridian models
    if library and library.lower() == "meridian" and dataset_name:
        model_path = Path(f"data/results/{dataset_name}/meridian_model.pkl")
        if model_path.exists():
            size_bytes = model_path.stat().st_size
            return size_bytes / BYTES_TO_MB
    
    # Try to get file size for PyMC models
    if library and "pymc" in library.lower() and dataset_name and sampler:
        model_path = Path(f"data/results/{dataset_name}/pymc_{sampler}_model.nc")
        if model_path.exists():
            size_bytes = model_path.stat().st_size
            return size_bytes / BYTES_TO_MB
    
    # Return -1 if model file not found or invalid parameters
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
            library = "meridian"
            sampler = None
        else:
            var_names = get_pymc_var_names()
            idata = model.idata if hasattr(model, 'idata') else None
            library = "pymc"
            # Extract sampler from key like "PyMC-Marketing - nutpie"
            sampler = key.split(" - ")[-1] if " - " in key else None
        
        if idata is None:
            continue
        
        divergences = compute_divergences(idata)
        rhat = compute_rhat(idata, var_names)
        size_mb = compute_model_size_mb(dataset_name, library, sampler)
        
        rows.append({
            'Dataset': dataset_name,
            'Model': key,
            'Runtime (s)': runtime,
            'ESS min': ess.get('min'),
            'ESS q50': ess.get('q50'),
            'Divergences': divergences,
            'R-hat max': rhat['max'],
            'R-hat > 1.1': rhat['bad_count'],
            'Size (MB)': size_mb
        })
    
    return pd.DataFrame(rows)