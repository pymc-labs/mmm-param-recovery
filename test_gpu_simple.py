#!/usr/bin/env python
"""Simplified GPU test for PyMC-Marketing and Meridian MMM models."""

import os
import sys
import time
import warnings

# Configure environment for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Try CUDA first, fall back to CPU

# Standard imports
import numpy as np
import pandas as pd
import pickle

# GPU detection
try:
    import tensorflow as tf
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print(f"Warning: Some GPU libraries not available: {e}")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import benchmarking modules
from mmm_param_recovery.benchmarking import (
    data_loader,
    model_builder,
    model_fitter,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def check_gpu_status():
    """Quick GPU status check."""
    print("="*60)
    print("GPU STATUS CHECK")
    print("="*60)
    
    # TensorFlow GPUs
    try:
        tf_gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPUs: {len(tf_gpus)}")
    except:
        print("TensorFlow GPU check: N/A")
    
    # JAX backend
    try:
        devices = jax.devices()
        backend = devices[0].platform if devices else 'unknown'
        print(f"JAX backend: {backend}")
        print(f"JAX devices: {[str(d) for d in devices]}")
    except:
        print("JAX check: N/A")
    
    # Test JAX computation
    try:
        x = jnp.ones((100, 100))
        y = jnp.dot(x, x)
        y.block_until_ready()
        print("JAX computation: OK")
    except Exception as e:
        print(f"JAX computation: {e}")
    
    print()


def run_minimal_test():
    """Run minimal test with existing data loading infrastructure."""
    print("="*60)
    print("MINIMAL GPU TEST")
    print("="*60)
    
    # Load small dataset using existing infrastructure
    print("\nLoading dataset...")
    datasets = data_loader.load_multiple_datasets(["small_business"], seed=42)
    data_df, channel_columns, control_columns, truth_df = datasets[0]
    
    print(f"Data shape: {data_df.shape}")
    print(f"Channels: {channel_columns}")
    print(f"Controls: {control_columns}")
    print(f"Geos: {data_df['geo'].unique()}")
    
    # Test configurations
    test_configs = {
        "chains": 2,
        "draws": 100,  # Very small for testing
        "tune": 100,
        "target_accept": 0.9,
        "seed": 42
    }
    
    results = {}
    
    # Test Meridian
    print("\n" + "-"*60)
    print("Testing Meridian...")
    print("-"*60)
    try:
        meridian_model = model_builder.build_meridian_model(
            data_df, channel_columns, control_columns
        )
        
        start = time.time()
        fitted_model, runtime, ess = model_fitter.fit_meridian(
            meridian_model,
            test_configs["chains"],
            test_configs["draws"],
            test_configs["tune"],
            test_configs["target_accept"],
            test_configs["seed"]
        )
        
        results["meridian"] = {
            "success": True,
            "runtime": runtime,
            "min_ess": ess.get("min", "N/A")
        }
        print(f"✓ Meridian completed in {runtime:.2f}s")
        
    except Exception as e:
        results["meridian"] = {"success": False, "error": str(e)}
        print(f"✗ Meridian failed: {e}")
    
    # Test PyMC-Marketing with different samplers
    # Only test JAX-based samplers for GPU
    gpu_samplers = ["numpyro", "blackjax", "nutpie"]
    
    for sampler in gpu_samplers:
        print("\n" + "-"*60)
        print(f"Testing PyMC-Marketing ({sampler})...")
        print("-"*60)
        
        try:
            fitted_model, runtime, ess = model_fitter.fit_pymc(
                data_df,
                channel_columns,
                control_columns,
                sampler,
                test_configs["chains"],
                test_configs["draws"],
                test_configs["tune"],
                test_configs["target_accept"],
                test_configs["seed"]
            )
            
            results[f"pymc_{sampler}"] = {
                "success": True,
                "runtime": runtime,
                "min_ess": ess.get("min", "N/A")
            }
            print(f"✓ PyMC-{sampler} completed in {runtime:.2f}s")
            
        except Exception as e:
            results[f"pymc_{sampler}"] = {"success": False, "error": str(e)[:100]}
            print(f"✗ PyMC-{sampler} failed: {str(e)[:100]}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<20} {'Status':<10} {'Runtime (s)':<15} {'Min ESS':<10}")
    print("-"*60)
    
    for name, result in results.items():
        if result["success"]:
            status = "✓ OK"
            runtime = f"{result['runtime']:.2f}"
            min_ess = str(result["min_ess"])
        else:
            status = "✗ FAIL"
            runtime = "N/A"
            min_ess = "N/A"
        
        print(f"{name:<20} {status:<10} {runtime:<15} {min_ess:<10}")
    
    # Performance comparison for successful runs
    successful = {k: v for k, v in results.items() if v["success"]}
    if len(successful) > 1:
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"\n{'Model':<20} {'Samples/sec':<15}")
        print("-"*35)
        
        total_samples = test_configs["chains"] * test_configs["draws"]
        for name, result in successful.items():
            samples_per_sec = total_samples / result["runtime"]
            print(f"{name:<20} {samples_per_sec:.2f}")
    
    return results


def main():
    """Main test function."""
    print("="*60)
    print("GPU TEST FOR MMM LIBRARIES")
    print("="*60)
    print("This test runs minimal sampling to verify GPU acceleration")
    print()
    
    # Check GPU status
    check_gpu_status()
    
    # Run tests
    results = run_minimal_test()
    
    # Final status
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"\nTests passed: {successful}/{total}")
    
    if successful == 0:
        print("\n⚠️  No tests succeeded. This may indicate:")
        print("  - No GPU available (running on CPU)")
        print("  - Missing GPU libraries or drivers")
        print("  - Configuration issues")
    elif successful < total:
        print("\n⚠️  Some tests failed. Check error messages above.")
    else:
        print("\n✅ All tests passed successfully!")


if __name__ == "__main__":
    main()