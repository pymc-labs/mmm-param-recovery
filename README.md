# PyMC-Marketing vs. Meridian
We compare the two libraries in the following categories, from most important to least:
- Contribution recovery
- Predictive power
- Sampling efficiency (ESS / s)
- RAM footprint in sampling and of the final fitted model

## Quick Start

### Environment Setup

When working with GCP instances with preinstalled Jupyter Lab,
we have found Conda to be the easiest method.

```shell
conda env create -f cpu-environment.yaml
conda activate python312-cpu
```

### Running Benchmarks

The main benchmarking script provides a flexible CLI for comparing PyMC-Marketing and Meridian across different datasets and configurations.

#### Basic Usage

```bash
# Quick test with small dataset (recommended for laptops)
python run_benchmark.py --datasets small_business --samplers nutpie --chains 2 --draws 500 --tune 500

# Standard benchmark
python run_benchmark.py --datasets small_business medium_business \
                       --samplers pymc blackjax numpyro nutpie \
                       --chains 4 --draws 1000 --tune 1000

# Force re-run even if cached results exist
python run_benchmark.py --datasets small_business --force-rerun

# Use cached results (default behavior)
python run_benchmark.py --datasets small_business --no-force-rerun

# Generate only plots from existing models
python run_benchmark.py --datasets small_business --plots-only
```

#### Available Options

- `--datasets`: Choose from `small_business`, `medium_business`, `large_business`, `growing_business`
- `--samplers`: PyMC samplers - `pymc`, `blackjax`, `numpyro`, `nutpie` 
- `--libraries`: Run only specific libraries - `meridian`, `pymc`
- `--chains`: Number of MCMC chains (default: 4)
- `--draws`: Number of draws per chain (default: 1000)
- `--tune`: Number of tuning steps (default: 1000)
- `--target-accept`: Target acceptance probability (default: 0.9)
- `--seed`: Random seed for reproducibility
- `--force-rerun`: Force re-run even if cached results exist
- `--plots-only`: Generate only plots from existing models

#### Output Structure

```
data/results/
├── {dataset_name}/
│   ├── data.pkl                    # Cached generated dataset
│   ├── meridian_model.pkl          # Fitted Meridian model
│   ├── meridian_stats.pkl          # Meridian runtime & ESS
│   ├── pymc_{sampler}_model.nc     # PyMC models per sampler
│   ├── pymc_{sampler}_stats.pkl    # PyMC stats per sampler
│   └── plots/                      
│       ├── posterior_predictive_meridian.html
│       └── posterior_predictive_pymc_{sampler}.png
└── summary/
    ├── runtime_comparison.csv       # Runtime comparison table
    ├── ess_comparison.csv          # ESS metrics table
    ├── performance_metrics.csv     # R², MAPE, Durbin-Watson
    ├── diagnostics_summary.csv     # Convergence diagnostics
    └── combined_plots/
        ├── runtime_comparison.png
        ├── ess_comparison.png
        ├── performance_comparison.png
        └── diagnostics_summary.png
```

### Jupyter Notebooks

For interactive exploration, use the provided notebooks:

- `comparison_all_cpu_adstock.ipynb` - Main comparison notebook
- `comparison_all_cpu_adstock_large_dataset.ipynb` - Large dataset benchmark

## GPU Tests

```shell
conda create -n python311 python=3.11 -y && conda activate python311
conda install pip
pip install --user -r gpu-requirements-compiled.txt
ipython kernel install --name python311 --display-name "Python 3.11"  --user
```

## Notes

- **Memory constraints**: `blackjax` and `numpyro` samplers are automatically skipped for large datasets
- **Sequential execution**: Models are fitted one at a time to conserve CPU resources
- **Incremental saving**: Each model is saved immediately after fitting to prevent data loss
- **Caching**: Dataset generation and model fitting results are cached by default
