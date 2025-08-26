# MMM Parameter Recovery Benchmarking Framework

## Project Overview
This repository benchmarks PyMC-Marketing and Google Meridian for Media Mix Modeling (MMM) parameter recovery using synthetic datasets with known ground truth.

## Architecture

### Core Modules

#### 1. Data Generation (`mmm_param_recovery.data_generator`)
**DO NOT MODIFY** - Use these functions as-is:
- `generate_mmm_dataset(config)` - Main entry point for synthetic data generation
- `get_preset_config(name)` - Returns preset configurations (small_business, medium_business, large_business, growing_business)
- Generates multi-region data with channels, controls, adstock, and saturation transformations

#### 2. Benchmarking Framework (`mmm_param_recovery.benchmarking`)
Modular components for model comparison:

- **data_loader.py**: Load and prepare datasets
  - `load_multiple_datasets(dataset_names, seed)`
  - Returns tuples of (data_df, channel_columns, control_columns, truth_df)

- **model_builder.py**: Construct models for both libraries
  - `build_meridian_model(data_df, channel_columns, control_columns)` 
  - `build_pymc_model(data_df, channel_columns, control_columns)`
  - Handles prior calculation based on spend shares

- **model_fitter.py**: Fit models with different samplers
  - `fit_meridian(model, n_chains, n_draws, n_tune, target_accept, seed)`
  - `fit_pymc(model, data_df, sampler, n_chains, n_draws, n_tune, target_accept, seed)`
  - Returns (fitted_model, runtime, ess_stats)

- **diagnostics.py**: Compute convergence metrics
  - `compute_ess(idata)` - Returns min, q10, q50, q90 ESS values
  - `create_diagnostics_summary(results, dataset_name)`

- **evaluation.py**: Performance metrics
  - `evaluate_meridian_fit(model, data_df)` 
  - `evaluate_pymc_fit(model, data_df, sampler)`
  - Returns R², MAPE, Durbin-Watson statistics

- **visualization.py**: Generate plots
  - Posterior predictive checks
  - Runtime/ESS comparisons
  - Performance metrics visualization

- **storage.py**: Model persistence
  - `save_meridian_model()`, `save_pymc_model()`
  - `load_meridian_model()`, `load_pymc_model()`
  - `model_exists(dataset_name, library, sampler=None)`

## Code Organization

### Data Generator Module Structure

The `data_generator` module follows a functional, layered architecture:

```
mmm_param_recovery/data_generator/
├── __init__.py          # Public API exports
├── core.py              # Main orchestration logic
├── config.py            # Configuration dataclasses
├── channels.py          # Channel/control generation functions
├── regions.py           # Regional variation functions
├── transforms.py        # Adstock/saturation transformations
├── ground_truth.py      # ROAS/attribution calculations
├── presets.py           # Business preset configurations
├── validation.py        # Input/output validation
├── visualization.py     # Plotting utilities
└── utils.py            # Helper functions (seed management)
```

**Key Design Patterns:**
- **Configuration-driven**: All parameters defined in dataclasses (`ChannelConfig`, `RegionConfig`, `TransformConfig`)
- **Pure functions**: Each module contains stateless functions that transform inputs to outputs
- **Layered generation**: Core orchestrates channel → region → transform pipeline
- **Separation of concerns**: Each file handles one aspect (channels, regions, transforms)

### Benchmarking Module Structure

The `benchmarking` module implements a pipeline pattern:

```
mmm_param_recovery/benchmarking/
├── __init__.py          # Module exports
├── data_loader.py       # Dataset loading and preparation
├── model_builder.py     # Model construction for both libraries
├── model_fitter.py      # Sampling/fitting logic
├── diagnostics.py       # Convergence metrics
├── evaluation.py        # Performance metrics
├── visualization.py     # Plotting functions
└── storage.py          # Model persistence
```

**Key Design Patterns:**
- **Pipeline architecture**: Each module represents a stage in the benchmarking pipeline
- **Library abstraction**: Separate functions for Meridian and PyMC-Marketing with consistent interfaces
- **Result aggregation**: Functions return standardized tuples (model, runtime, metrics)
- **Caching strategy**: Storage module handles all persistence logic

### Data Flow

1. **Generation Phase** (`data_generator`):
   ```
   config → channels.generate_channel_spend()
          → regions.generate_regional_variations() 
          → transforms.apply_transformations()
          → ground_truth.calculate_metrics()
          → validated output
   ```

2. **Benchmarking Phase** (`benchmarking`):
   ```
   dataset → data_loader.load_dataset()
           → model_builder.build_[library]_model()
           → model_fitter.fit_[library]()
           → diagnostics.compute_metrics()
           → evaluation.evaluate_fit()
           → visualization.create_plots()
           → storage.save_results()
   ```

### Function Naming Conventions

- **Data Generator**:
  - `generate_*` - Create new data
  - `calculate_*` - Compute derived metrics
  - `validate_*` - Check constraints
  - `plot_*` - Visualization functions

- **Benchmarking**:
  - `build_*` - Construct models
  - `fit_*` - Run sampling
  - `compute_*` - Calculate metrics
  - `evaluate_*` - Assess performance
  - `save_*/load_*` - Persistence operations
  - `plot_*` - Create visualizations

### Type Hints and Return Patterns

Both modules use consistent type patterns:

```python
# Data generator returns
Dict[str, Union[pd.DataFrame, Dict]]

# Model builder returns
Union[model.Meridian, MMM]

# Model fitter returns
Tuple[Model, float, Dict[str, Optional[float]]]

# Evaluation returns
List[Dict[str, Any]]
```

### Error Handling Philosophy

- **Fail fast**: Validate inputs early (e.g., `validate_config()`)
- **No silent failures**: Raise explicit exceptions with descriptive messages
- **No generic catches**: Let specific errors propagate for debugging
- **Validation layers**: Check data structure at module boundaries

## Key API Patterns

### PyMC-Marketing MMM
```python
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm import GeometricAdstock, HillSaturationSigmoid
from pymc_marketing.prior import Prior

# Build hierarchical saturation with geo-varying parameters
saturation = HillSaturationSigmoid(
    priors={
        "sigma": Prior(
            "InverseGamma",
            mu=Prior("HalfNormal", sigma=prior_sigma.mean(axis=0), dims=("channel",)),
            sigma=Prior("HalfNormal", sigma=1.5),
            dims=("channel", "geo")
        ),
        "beta": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
        "lam": Prior("HalfNormal", sigma=1.5, dims=("channel",))
    }
)

# Initialize model with geo dimensions
mmm = MMM(
    date_column="time",
    target_column="y",
    channel_columns=channel_columns,
    control_columns=control_columns,
    dims=("geo",),  # Critical for multi-region modeling
    scaling={
        "channel": {"method": "max", "dims": ()},
        "target": {"method": "max", "dims": ()}
    },
    saturation=saturation,
    adstock=GeometricAdstock(
        l_max=8,
        priors={"alpha": Prior("Beta", alpha=1, beta=3, dims=("channel",))}
    ),
    yearly_seasonality=2
)

# Build and fit
mmm.build_model(X=x_train, y=y_train)
mmm.fit(
    X=x_train, y=y_train,
    chains=4, draws=1000, tune=1000,
    nuts_sampler="nutpie",  # Options: "pymc", "blackjax", "numpyro", "nutpie"
    nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"}  # For nutpie
)

# Add contribution variables for analysis
mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "control_contribution", 
         "intercept_contribution", "yearly_seasonality_contribution", "y"]
)
```

### Google Meridian
```python
from meridian.data import data_frame_input_data_builder
from meridian.model import model, prior_distribution, spec
import tensorflow_probability as tfp

# Build input data with geo-level structure
builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
    kpi_type='revenue'
)
builder = (
    builder.with_kpi(data_df, kpi_col="y")
           .with_population(data_df)
           .with_controls(data_df, control_cols=control_columns)
           .with_media(
               data_df,
               media_cols=channel_columns,
               media_spend_cols=channel_columns,
               media_channels=channel_columns
           )
)
built_data = builder.build()

# Configure priors based on spend shares
beta_m = build_media_channel_args(
    **{col: (0, float(prior_sigma.mean(axis=0)[i])) 
       for i, col in enumerate(channel_columns)}
)
prior = prior_distribution.PriorDistribution(
    beta_m=tfp.distributions.LogNormal(beta_m_mu, beta_m_sigma)
)

# Model specification
model_spec = spec.ModelSpec(
    prior=prior,
    media_effects_dist='log_normal',
    hill_before_adstock=False,
    max_lag=8,
    unique_sigma_for_each_geo=True,
    knots=knots,  # Changepoints for trend
    media_prior_type='coefficient'
)

# Fit model
meridian_model = model.Meridian(input_data=built_data, model_spec=model_spec)
meridian_model.sample_posterior(
    n_chains=4,
    n_adapt=500,
    n_burnin=500,
    n_keep=1000,
    seed=(seed, seed),
    dual_averaging_kwargs={"target_accept_prob": 0.9}
)
```

## Data Format Requirements

### Input DataFrame Structure
```python
# Multi-index: (date, geo)
# Columns: y, x1_channel1, x2_channel2, ..., c1_control1, ...
data_df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [date_range, geo_regions], 
        names=['date', 'geo']
    )
)
```

### Channel Naming Convention
- Channels: `x{i}_{name}` (e.g., x1_tv, x2_digital)
- Controls: `c{i}` (e.g., c1, c2)
- Contributions: `contribution_x{i}_{name}`

## Performance Considerations

### Memory Management
- BlackJAX/NumPyro automatically skipped for large datasets
- Models saved immediately after fitting to prevent data loss
- Use `--force-rerun` flag sparingly

### Sampler Selection
- **nutpie**: Fastest, JAX-based, good for all dataset sizes
- **pymc**: Default PyMC sampler, stable but slower
- **blackjax**: JAX-based, fast but memory intensive
- **numpyro**: JAX-based, fast but memory intensive

### Optimal Settings by Dataset Size
- **small_business**: All samplers, 4 chains, 1000 draws
- **medium_business**: All samplers, 4 chains, 1000 draws  
- **large_business**: nutpie/pymc only, 4 chains, 1000 draws
- **growing_business**: All samplers, 4 chains, 1000 draws

## Common Tasks

### Run Full Benchmark
```bash
python run_benchmark.py --datasets small_business medium_business \
                       --samplers pymc nutpie \
                       --chains 4 --draws 1000 --tune 1000
```

### Quick Test
```bash
python run_benchmark.py --datasets small_business \
                       --samplers nutpie \
                       --chains 2 --draws 500 --tune 500
```

### Generate Plots Only
```bash
python run_benchmark.py --datasets small_business --plots-only
```

## Output Structure
```
data/results/
├── {dataset}/
│   ├── data.pkl                    # Cached dataset
│   ├── meridian_model.pkl          # Fitted model
│   ├── pymc_{sampler}_model.nc     # PyMC models
│   └── plots/                      # Visualizations
│       ├── posterior_predictive_meridian.png
│       └── posterior_predictive_pymc_{sampler}.png
└── summary/
    ├── runtime_comparison.csv      # Performance tables
    ├── ess_comparison.csv
    ├── performance_metrics.csv
    └── combined_plots/            # Comparison visualizations
```

## Code Style Guidelines
- Functional programming approach with pure functions
- No generic exception catching
- Type hints for all function signatures
- Immutable data structures where possible
- Modular organization by feature/responsibility

## Debugging Tips

### Check Model Convergence
```python
# For PyMC
import arviz as az
az.plot_trace(mmm.idata)
print(az.summary(mmm.idata))

# For Meridian  
meridian_model.inference_data.posterior
```

### Verify Data Structure
```python
# Ensure MultiIndex is correct
assert data_df.index.names == ['date', 'geo']
assert all(col.startswith('x') or col.startswith('c') or col == 'y' 
           for col in data_df.columns)
```

### Memory Profiling
```bash
mprof run --include-children python run_benchmark.py
mprof plot
```

## Common Issues

1. **OOM with JAX samplers**: Reduce batch size or use nutpie/pymc
2. **Slow sampling**: Check target_accept (0.9 recommended)
3. **Poor convergence**: Increase tune steps or adjust priors
4. **Data mismatch**: Verify geo column exists and matches region names

## Important Notes
- Never modify `data_generator` module - use as-is
- Always use absolute paths in file operations
- Models are cached by default - use `--force-rerun` to regenerate
- Geo-level modeling requires `dims=("geo",)` in PyMC-Marketing