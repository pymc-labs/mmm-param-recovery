#!/usr/bin/env python
"""Minimal GPU test for PyMC-Marketing and Meridian MMM models."""

import os
import sys
import time
import warnings
from typing import Dict, Any

# Enable GPU visibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force JAX to use CUDA

# Standard imports
import numpy as np
import pandas as pd

# GPU detection imports
try:
    import tensorflow as tf
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print(f"Error importing GPU libraries: {e}")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MMM libraries
from mmm_param_recovery.data_generator import generate_mmm_dataset, get_preset_config
from mmm_param_recovery.benchmarking import model_builder

# PyMC and sampling
import pymc as pm
import arviz as az

# Meridian
from meridian.model import model

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def check_gpu_availability() -> Dict[str, Any]:
    """Check and report GPU availability for different backends."""
    gpu_info = {
        "tensorflow": {"available": False, "devices": []},
        "jax": {"available": False, "devices": [], "backend": None},
        "cuda": {"available": False}
    }
    
    # Check TensorFlow GPU
    try:
        tf_gpus = tf.config.list_physical_devices('GPU')
        gpu_info["tensorflow"]["available"] = len(tf_gpus) > 0
        gpu_info["tensorflow"]["devices"] = [gpu.name for gpu in tf_gpus]
        print(f"TensorFlow GPU devices: {len(tf_gpus)}")
        for gpu in tf_gpus:
            print(f"  - {gpu.name}")
    except Exception as e:
        print(f"TensorFlow GPU check failed: {e}")
    
    # Check JAX GPU
    try:
        jax_devices = jax.devices()
        jax_backend = jax_devices[0].platform if jax_devices else 'cpu'
        gpu_info["jax"]["available"] = jax_backend in ['gpu', 'cuda']
        gpu_info["jax"]["backend"] = jax_backend
        gpu_info["jax"]["devices"] = [str(d) for d in jax_devices]
        print(f"JAX backend: {jax_backend}")
        print(f"JAX devices: {len(jax_devices)}")
        for device in jax_devices:
            print(f"  - {device}")
            
        # Test JAX GPU computation
        if gpu_info["jax"]["available"]:
            test_array = jnp.ones((100, 100))
            result = jnp.dot(test_array, test_array)
            result.block_until_ready()
            print("JAX GPU computation test: SUCCESS")
    except Exception as e:
        print(f"JAX GPU check failed: {e}")
    
    # Check CUDA availability via PyMC
    try:
        import torch
        gpu_info["cuda"]["available"] = torch.cuda.is_available()
        if gpu_info["cuda"]["available"]:
            print(f"CUDA available via PyTorch: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed, skipping CUDA check")
    
    return gpu_info


def generate_test_data(seed: int = 42) -> tuple:
    """Generate small test dataset for GPU testing."""
    print("\n" + "="*60)
    print("GENERATING TEST DATA")
    print("="*60)
    
    config = get_preset_config("small_business")
    config.n_periods = 52  # Reduce to 1 year for faster testing
    config.seed = seed
    
    result = generate_mmm_dataset(config)
    data_df = result['data']
    
    # Rename columns to match expected format
    data_df = data_df.rename(columns={'date': 'time'})
    
    # Add geo column if not present
    if 'geo' not in data_df.columns:
        data_df['geo'] = 'geo_a'  # Single geo for testing
    
    # Extract channel and control columns
    channel_columns = [col for col in data_df.columns if col.startswith('x')]
    control_columns = [col for col in data_df.columns if col.startswith('c') and not col.endswith('_effect')]
    
    print(f"Data shape: {data_df.shape}")
    print(f"Channels: {channel_columns}")
    print(f"Controls: {control_columns}")
    print(f"Date range: {data_df['time'].min()} to {data_df['time'].max()}")
    print(f"Geos: {data_df['geo'].unique()}")
    
    return data_df, channel_columns, control_columns


def test_pymc_gpu(data_df: pd.DataFrame, channel_columns: list, control_columns: list) -> Dict[str, Any]:
    """Test PyMC-Marketing with GPU-accelerated samplers."""
    print("\n" + "="*60)
    print("TESTING PYMC-MARKETING WITH GPU")
    print("="*60)
    
    results = {}
    
    # Build PyMC model
    pymc_model = model_builder.build_pymc_model(data_df, channel_columns, control_columns)
    
    # Prepare data
    x_train = data_df.drop(columns=["y"])
    y_train = data_df["y"]
    
    # Test different GPU-capable samplers
    gpu_samplers = ["numpyro", "blackjax", "nutpie"]
    
    for sampler in gpu_samplers:
        print(f"\n--- Testing {sampler} sampler ---")
        
        try:
            # Configure sampler-specific kwargs
            kwargs = {}
            if sampler == "nutpie":
                kwargs = {
                    "nuts_sampler_kwargs": {
                        "backend": "jax",
                        "gradient_backend": "jax"
                    }
                }
            
            # Time the sampling
            start_time = time.time()
            
            # Fit model with reduced samples for testing
            pymc_model.fit(
                X=x_train,
                y=y_train,
                chains=2,  # Reduced chains
                draws=500,  # Reduced draws
                tune=500,  # Reduced tuning
                target_accept=0.9,
                random_seed=42,
                nuts_sampler=sampler,
                **kwargs
            )
            
            runtime = time.time() - start_time
            
            # Check results
            idata = pymc_model.idata
            ess = az.ess(idata, var_names=["~log_likelihood"]).to_dataframe()
            min_ess = ess.min().min()
            
            results[sampler] = {
                "success": True,
                "runtime": runtime,
                "min_ess": min_ess,
                "chains": 2,
                "draws": 500
            }
            
            print(f"✓ {sampler}: Runtime={runtime:.2f}s, Min ESS={min_ess:.0f}")
            
        except Exception as e:
            results[sampler] = {
                "success": False,
                "error": str(e)
            }
            print(f"✗ {sampler}: {e}")
    
    return results


def test_meridian_gpu(data_df: pd.DataFrame, channel_columns: list, control_columns: list) -> Dict[str, Any]:
    """Test Meridian with GPU support via TensorFlow."""
    print("\n" + "="*60)
    print("TESTING MERIDIAN WITH GPU")
    print("="*60)
    
    try:
        # Build Meridian model
        meridian_model = model_builder.build_meridian_model(
            data_df, channel_columns, control_columns
        )
        
        # Time the sampling
        start_time = time.time()
        
        # Sample posterior with reduced samples for testing
        meridian_model.sample_posterior(
            n_chains=2,
            n_adapt=500,
            n_burnin=500,
            n_keep=500,
            seed=(42, 42),
            dual_averaging_kwargs={"target_accept_prob": 0.9}
        )
        
        runtime = time.time() - start_time
        
        # Check results
        inference_data = meridian_model.inference_data
        posterior_samples = inference_data.posterior
        
        result = {
            "success": True,
            "runtime": runtime,
            "chains": 2,
            "draws": 500,
            "posterior_vars": list(posterior_samples.data_vars.keys())[:5]  # Show first 5 vars
        }
        
        print(f"✓ Meridian: Runtime={runtime:.2f}s")
        print(f"  Posterior variables sampled: {len(posterior_samples.data_vars)}")
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }
        print(f"✗ Meridian: {e}")
    
    return result


def main():
    """Main GPU testing function."""
    print("="*60)
    print("GPU TEST FOR MMM LIBRARIES")
    print("="*60)
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    
    if not any([gpu_info["tensorflow"]["available"], 
                gpu_info["jax"]["available"], 
                gpu_info["cuda"]["available"]]):
        print("\n⚠️  WARNING: No GPU detected! Tests will run on CPU.")
        print("Continuing with CPU execution...")
    
    # Generate test data
    data_df, channel_columns, control_columns = generate_test_data()
    
    # Test PyMC-Marketing with GPU
    pymc_results = test_pymc_gpu(data_df, channel_columns, control_columns)
    
    # Test Meridian with GPU
    meridian_result = test_meridian_gpu(data_df, channel_columns, control_columns)
    
    # Summary
    print("\n" + "="*60)
    print("GPU TEST SUMMARY")
    print("="*60)
    
    print("\nGPU Availability:")
    print(f"  TensorFlow: {gpu_info['tensorflow']['available']}")
    print(f"  JAX: {gpu_info['jax']['available']} (backend: {gpu_info['jax']['backend']})")
    print(f"  CUDA: {gpu_info['cuda']['available']}")
    
    print("\nPyMC-Marketing Results:")
    for sampler, result in pymc_results.items():
        if result["success"]:
            print(f"  {sampler}: ✓ SUCCESS (Runtime: {result['runtime']:.2f}s)")
        else:
            print(f"  {sampler}: ✗ FAILED ({result['error'][:50]}...)")
    
    print("\nMeridian Results:")
    if meridian_result["success"]:
        print(f"  ✓ SUCCESS (Runtime: {meridian_result['runtime']:.2f}s)")
    else:
        print(f"  ✗ FAILED ({meridian_result['error'][:50]}...)")
    
    # Performance comparison (if both succeeded)
    successful_pymc = [s for s, r in pymc_results.items() if r["success"]]
    if meridian_result["success"] and successful_pymc:
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON (500 draws, 2 chains)")
        print("="*60)
        print(f"{'Library':<25} {'Runtime (s)':<15} {'Samples/sec':<15}")
        print("-"*55)
        
        if meridian_result["success"]:
            samples_per_sec = (500 * 2) / meridian_result["runtime"]
            print(f"{'Meridian':<25} {meridian_result['runtime']:<15.2f} {samples_per_sec:<15.2f}")
        
        for sampler in successful_pymc:
            if pymc_results[sampler]["success"]:
                runtime = pymc_results[sampler]["runtime"]
                samples_per_sec = (500 * 2) / runtime
                print(f"{f'PyMC-Marketing ({sampler})':<25} {runtime:<15.2f} {samples_per_sec:<15.2f}")
    
    print("\n" + "="*60)
    print("GPU TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()