#!/usr/bin/env python
"""Ultra-minimal GPU detection and configuration test."""

import os
import sys

# Configure for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

print("="*60)
print("GPU CONFIGURATION TEST")
print("="*60)

# TensorFlow GPU check
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nTensorFlow:")
    print(f"  GPU devices found: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"    - {gpu.name}")
    else:
        print("    Running on CPU")
except Exception as e:
    print(f"\nTensorFlow: Error - {e}")

# JAX GPU check
try:
    import jax
    import jax.numpy as jnp
    
    print(f"\nJAX:")
    devices = jax.devices()
    backend = devices[0].platform if devices else 'unknown'
    print(f"  Backend: {backend}")
    print(f"  Devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"    - Device {i}: {device}")
    
    # Test computation
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print(f"  Matrix multiplication test: SUCCESS")
    
    if backend == 'gpu' or backend == 'cuda':
        print("\n✅ GPU ACCELERATION AVAILABLE")
    else:
        print("\n⚠️  Running on CPU (GPU not detected)")
        
except Exception as e:
    print(f"\nJAX: Error - {e}")

# PyTorch CUDA check (if available)
try:
    import torch
    print(f"\nPyTorch:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError:
    print("\nPyTorch: Not installed")
except Exception as e:
    print(f"\nPyTorch: Error - {e}")

print("\n" + "="*60)
print("CONFIGURATION SUMMARY")
print("="*60)

print("\nTo enable GPU acceleration:")
print("1. Ensure CUDA drivers are installed")
print("2. Install GPU-enabled versions of libraries:")
print("   - pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
print("   - pip install tensorflow[and-cuda]")
print("3. Set environment variables:")
print("   - export CUDA_VISIBLE_DEVICES=0")
print("   - export JAX_PLATFORMS=cuda")

print("\nFor MMM models:")
print("- Meridian: Uses TensorFlow for GPU acceleration")
print("- PyMC-Marketing: Uses JAX (numpyro, blackjax) or custom (nutpie)")
print("\nNote: Both libraries will automatically fall back to CPU if GPU is unavailable.")