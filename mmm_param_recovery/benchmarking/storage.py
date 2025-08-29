"""Storage management functions for benchmarking results."""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import pandas as pd
from meridian.model import model as meridian_model_module
from pymc_marketing.mmm.multidimensional import MMM


def get_paths(dataset_name: str, library: str, sampler: Optional[str] = None) -> Tuple[Path, Path]:
    """Get model and stats paths for a given configuration.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    library : str
        Either 'meridian' or 'pymc'
    sampler : Optional[str]
        Sampler name for PyMC models
        
    Returns
    -------
    Tuple[Path, Path]
        Model path and stats path
    """
    dir_path = Path(f"data/results/{dataset_name}")
    dir_path.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{sampler}" if sampler else ""
    model_ext = ".pkl" if library.lower() == "meridian" else ".nc"
    
    return (
        dir_path / f"{library}{suffix}_model{model_ext}",
        dir_path / f"{library}{suffix}_stats.pkl"
    )


def get_plot_path(dataset_name: str, plot_name: str) -> Path:
    """Get path for saving plots.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    plot_name : str
        Name of the plot file
        
    Returns
    -------
    Path
        Path to save the plot
    """
    plot_dir = Path(f"data/results/{dataset_name}/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / plot_name


def model_exists(dataset_name: str, library: str, sampler: Optional[str] = None) -> bool:
    """Check if model and stats files exist.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    library : str
        Either 'meridian' or 'pymc'
    sampler : Optional[str]
        Sampler name for PyMC models
        
    Returns
    -------
    bool
        True if both model and stats files exist
    """
    model_path, stats_path = get_paths(dataset_name, library, sampler)
    return model_path.exists() and stats_path.exists()


def save_meridian_model(
    model: Any,
    dataset_name: str,
    runtime: float,
    ess_stats: Dict[str, Optional[float]]
) -> None:
    """Save Meridian model and statistics.
    
    Parameters
    ----------
    model : meridian.model.model.Meridian
        Fitted Meridian model
    dataset_name : str
        Name of the dataset
    runtime : float
        Runtime in seconds
    ess_stats : Dict[str, Optional[float]]
        ESS statistics
    """
    model_path, stats_path = get_paths(dataset_name, "meridian")
    
    meridian_model_module.save_mmm(model, str(model_path))
    
    with open(stats_path, 'wb') as f:
        pickle.dump({'runtime': runtime, 'ess': ess_stats}, f)
    
    print(f"  ✓ Saved Meridian model and diagnostics at {model_path}")


def save_pymc_model(
    model: MMM,
    dataset_name: str,
    sampler: str,
    runtime: float,
    ess_stats: Dict[str, Optional[float]]
) -> None:
    """Save PyMC-Marketing model and statistics.
    
    Parameters
    ----------
    model : MMM
        Fitted PyMC-Marketing model
    dataset_name : str
        Name of the dataset
    sampler : str
        Sampler name
    runtime : float
        Runtime in seconds
    ess_stats : Dict[str, Optional[float]]
        ESS statistics
    """
    model_path, stats_path = get_paths(dataset_name, "pymc", sampler)
    
    model.save(str(model_path))
    
    with open(stats_path, 'wb') as f:
        pickle.dump({'runtime': runtime, 'ess': ess_stats}, f)
    
    print(f"  ✓ Saved PyMC-Marketing - {sampler} model and diagnostics at {model_path}")


def load_meridian_model(dataset_name: str) -> Tuple[Any, float, Dict[str, Optional[float]]]:
    """Load Meridian model and statistics.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Tuple[meridian.model.model.Meridian, float, Dict]
        Model, runtime, and ESS stats
    """
    model_path, stats_path = get_paths(dataset_name, "meridian")
    
    model = meridian_model_module.load_mmm(str(model_path))
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"  ✓ Loaded Meridian from cache ({stats['runtime']:.1f}s)")
    return model, stats['runtime'], stats['ess']


def load_pymc_model(dataset_name: str, sampler: str) -> Tuple[MMM, float, Dict[str, Optional[float]]]:
    """Load PyMC-Marketing model and statistics.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    sampler : str
        Sampler name
        
    Returns
    -------
    Tuple[MMM, float, Dict]
        Model, runtime, and ESS stats
    """
    model_path, stats_path = get_paths(dataset_name, "pymc", sampler)
    
    # Try to load the model
    try:
        model = MMM.load(str(model_path))
    except Exception:
        # If MMM.load fails, create a mock model with the InferenceData
        import xarray as xr
        
        class MockMMM:
            def __init__(self, idata):
                self.idata = idata
                
        # Load just the InferenceData
        import arviz as az
        idata = az.from_netcdf(str(model_path))
        model = MockMMM(idata)
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"  ✓ Loaded PyMC-Marketing - {sampler} from cache ({stats['runtime']:.1f}s)")
    return model, stats['runtime'], stats['ess']


def save_summary_dataframe(df: pd.DataFrame, name: str) -> None:
    """Save summary dataframe to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    name : str
        Name of the summary file (without extension)
    """
    summary_dir = Path("data/results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / f"{name}.csv", index=True)
    print(f"  ✓ Saved summary to {summary_dir / f'{name}.csv'}")


def save_dataset(dataset_result: Dict[str, Any], dataset_name: str) -> None:
    """Save generated dataset.
    
    Parameters
    ----------
    dataset_result : Dict
        Result from generate_mmm_dataset
    dataset_name : str
        Name of the dataset
    """
    data_path = Path(f"data/results/{dataset_name}")
    data_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / "data.pkl", 'wb') as f:
        pickle.dump(dataset_result, f)
    
    print(f"  ✓ Saved dataset for {dataset_name}")


def load_dataset(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Load cached dataset if it exists.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Optional[Dict]
        Dataset result or None if not found
    """
    data_file = Path(f"data/results/{dataset_name}/data.pkl")
    
    if data_file.exists():
        with open(data_file, 'rb') as f:
            return pickle.load(f)
    
    return None