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
    └── diagnostics_summary.csv     # Convergence diagnostics
```


## Model Parameterization and Priors

This section details the exact model specifications used in the benchmark to ensure reproducibility.

### PyMC-Marketing MMM

#### Model Structure
- **Adstock**: Geometric adstock with maximum lag of 8 weeks
- **Saturation**: Hill saturation with sigmoid transformation (hierarchical)
- **Seasonality**: Yearly seasonality with 2 Fourier modes
- **Scaling**: Max scaling for both channels and target (dims=())

#### Prior Specifications
```python
# Adstock decay
adstock_alpha ~ Beta(alpha=1, beta=3)  # Per channel

# Saturation parameters (hierarchical)
saturation_sigma ~ InverseGamma(
    mu=HalfNormal(sigma=prior_sigma_per_channel),  # Population mean
    sigma=HalfNormal(sigma=1.5),                    # Population std
    dims=("channel", "geo")
)
saturation_beta ~ HalfNormal(sigma=1.5)  # Per channel
saturation_lam ~ HalfNormal(sigma=1.5)   # Per channel

# Control effects
gamma_control ~ Normal(0, 2)  # Implicit default, per control

# Intercept
intercept ~ Normal(mu=y_mean, sigma=y_std*2)  # Per geo

# Seasonality
gamma_fourier ~ Normal(0, 1)  # Per geo × fourier_mode

# Observation noise
y_sigma ~ HalfNormal(sigma=y_std*2)  # Per geo
```

Where `prior_sigma_per_channel = n_channels * spend_share` (spend share normalized per geo).

### Google Meridian

#### Model Structure
- **Adstock**: Geometric adstock with maximum lag of 8 weeks
- **Saturation**: Hill transformation applied after adstock
- **Trend**: Spline-based with knots every 26 weeks
- **Geo effects**: Unique sigma for each geo

#### Prior Specifications
```python
# Adstock decay  
alpha_m ~ Beta(alpha=1.0, beta=3.0)  # Per channel (matching PyMC)

# Media coefficients
beta_m ~ LogNormal(mu=0, sigma=prior_sigma_per_channel)  # Population level
beta_gm ~ Normal(0, 1)  # Geo-specific deviations
eta_m ~ HalfNormal(1)  # Media noise

# Hill saturation
ec_m ~ Beta(2, 2)  # Half-saturation point
slope_m ~ Gamma(2, 1)  # Hill slope

# Control effects
gamma_c ~ Normal(0, 1)  # Population level
gamma_gc ~ Normal(0, 1)  # Geo-specific
xi_c ~ HalfNormal(1)  # Control noise

# Baseline
tau_g ~ Normal(0, 5)  # Per geo

# Trend splines
knot_values ~ Normal(0, 5)  # Per knot

# Observation noise
sigma ~ HalfNormal(5)  # Single global parameter
```

### Key Differences

1. **Media Response Parameterization**:
   - PyMC: Uses hierarchical Hill saturation parameters that directly encode media effects
   - Meridian: Separates linear coefficients (beta) from saturation transformation (ec, slope)

2. **Geo-level Modeling**:
   - PyMC: Hierarchical priors with partial pooling across geos
   - Meridian: Explicit geo-specific parameters with population baselines

3. **Time-varying Effects**:
   - PyMC: Fourier-based seasonality (2 modes)
   - Meridian: Spline-based trend (knots every 26 weeks)

4. **Prior Informativeness**:
   - Both use weakly informative priors
   - Prior sigma for media effects based on spend shares in both models
   - Adstock priors favor faster decay (E[alpha] = 0.25)

### Parameter Count Comparison

For a dataset with G geos, C channels, and K controls:

| Component | PyMC-Marketing | Meridian |
|-----------|---------------|----------|
| Adstock | C | C |
| Media Response | C×(3 + G) + hierarchical | C×(5 + G) |
| Controls | K | K×(2 + G) + K |
| Baseline/Intercept | G | G |
| Time Effects | G×4 (seasonality) | 4-6 (trend knots) |
| Noise | G | 1 |

## Notes

- **Memory constraints**: `blackjax` and `numpyro` samplers are automatically skipped for large datasets
- **Sequential execution**: Models are fitted one at a time to conserve CPU resources
- **Incremental saving**: Each model is saved immediately after fitting to prevent data loss
- **Caching**: Dataset generation and model fitting results are cached by default

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 PyMC Labs
