# MMM Parameter Recovery

A comprehensive benchmarking suite for Marketing Mix Modeling (MMM) parameter recovery studies. This package provides tools to evaluate and compare different Bayesian inference methods for MMM models.

## Features

- **Multiple Samplers**: Compare PyMC (NUTS, NUTpie), NumPyro, and Meridian samplers
- **Comprehensive Evaluation**: Bayesian metrics, convergence diagnostics, and performance benchmarks
- **Automated Benchmarking**: Run full parameter recovery studies with configurable datasets
- **Rich Visualizations**: Detailed plots for model comparison and diagnostics
- **Storage & Caching**: Efficient storage of results and model artifacts

## Installation

```bash
pip install mmm-param-recovery
```

This will automatically install the `mmm-data-generator` dependency.

## Quick Start

```python
from mmm_param_recovery.benchmarking import run_benchmark
from mmm_data_generator import get_preset_config

# Get a dataset configuration
config = get_preset_config('small_business')

# Run benchmark with multiple samplers
results = run_benchmark(
    dataset_config=config,
    samplers=['pymc_nuts', 'numpyro', 'meridian'],
    n_samples=1000
)

# View results
print(results.summary())
```

## Available Samplers

- **PyMC NUTS**: Standard Hamiltonian Monte Carlo
- **PyMC NUTpie**: Advanced HMC with improved efficiency
- **NumPyro**: JAX-based probabilistic programming
- **Meridian**: Google's specialized MMM inference engine

## Evaluation Metrics

- **Parameter Recovery**: Compare estimated vs. true parameters
- **Convergence Diagnostics**: ESS, R-hat, trace plots
- **Performance**: Runtime, memory usage, samples per second
- **Model Fit**: In-sample fit metrics, posterior predictive checks

## Dependencies

This package includes all dependencies needed for comprehensive MMM benchmarking:
- mmm-data-generator (lightweight data generation)
- meridian (Google's MMM inference)
- pymc, pymc-marketing (PyMC ecosystem)
- numpyro (JAX-based inference)
- tensorflow, tensorflow-probability
- arviz (Bayesian analysis)
- And many more...

## Documentation

For detailed documentation, see the [main repository](https://github.com/pymc-labs/mmm-param-recovery).

## License

Apache License 2.0
