# Task List: MMM Dataset Generator

Based on the Product Requirements Document: MMM Dataset Generator

## Relevant Files

- `mmm_param_recovery/data_generator/__init__.py` - Module initialization and main API exports
- `mmm_param_recovery/data_generator/config.py` - Configuration dataclasses for all parameters
- `mmm_param_recovery/data_generator/core.py` - Main data generation function and core logic
- `mmm_param_recovery/data_generator/utils.py` - Seed management and random number generation utilities
- `mmm_param_recovery/data_generator/validation.py` - Data validation and quality checks
- `mmm_param_recovery/data_generator/presets.py` - Default configuration presets for common use cases
- `mmm_param_recovery/data_generator/channels.py` - Channel pattern generation and configuration (placeholder)
- `mmm_param_recovery/data_generator/regions.py` - Geographic region management and hierarchical structure (placeholder)
- `mmm_param_recovery/data_generator/transforms.py` - Adstock and saturation function implementations (placeholder)
- `mmm_param_recovery/data_generator/ground_truth.py` - Ground truth parameter calculation and ROAS computation (placeholder)
- `mmm_param_recovery/data_generator/visualization.py` - Plotting functions for data visualization (placeholder)
- `examples/basic_usage.py` - Basic example script demonstrating the module
- `examples/notebooks/data_generator_demo.py` - Marimo notebook for interactive demonstration
- `tests/test_channels.py` - Unit tests for channel generation
- `tests/test_regions.py` - Unit tests for region management
- `tests/test_transforms.py` - Unit tests for adstock and saturation functions
- `tests/test_ground_truth.py` - Unit tests for ground truth calculations
- `tests/test_full_pipeline.py` - Integration tests for complete data generation
- `pyproject.toml` - Pyproject environment configuration with required dependencies
- `README.md` - Documentation updates to reflect new features

### Notes

- **Environment Management:** Use Pixi for reproducible environments. Run `pixi install` to set up the environment.
- **Notebooks:** Create Marimo notebooks (`.py` files) in `examples/notebooks/` for interactive analysis. Convert to `.ipynb` when ready for distribution using `marimo export notebook.py`.
- **Testing:** Use `pytest tests/` to run all tests or `pytest tests/unit/test_specific_module.py` for specific tests.
- **Code Quality:** Follow PEP 8 style guidelines and use type hints where appropriate.
- **Documentation:** Update docstrings and README files to reflect new functionality.

## Tasks

- [x] 1.0 Core Module Structure and Configuration
  - [x] 1.1 Create module directory structure and `__init__.py` with main API exports
  - [x] 1.2 Implement configuration dataclasses for channel, region, and transform parameters
  - [x] 1.3 Create main `generate_mmm_dataset()` function with comprehensive parameter validation
  - [x] 1.4 Implement seed management and random number generation utilities
  - [x] 1.5 Add error handling and warning systems for invalid parameter combinations
  - [x] 1.6 Create default configuration presets for common use cases
  - [x] 1.7 Implement output schema compliance with required column structure (`date`, `geo`, `y`, `x1-*`, `x2-*`, ..., `c1-*`, `c2-*`, ...)
  - [x] 1.8 Add data integrity validation ensuring unique date-geo combinations and no missing values

- [x] 2.0 Channel Pattern Generation System
  - [x] 2.1 Implement linear trend channel pattern with configurable parameters look for example in [data.ipynb]
  - [x] 2.2 Implement seasonal channel pattern with annual seasonality look for example in [data.ipynb]
  - [x] 2.3 Implement delayed start channel pattern with configurable start time look for example in [data.ipynb]
  - [x] 2.4 Implement on/off channel pattern with random activation look for example in [data.ipynb]
  - [x] 2.6 Add channel parameter validation and reasonable range checks

- [ ] 3.0 Adstock and Saturation Transformations
  - [ ] 3.1 Implement geometric adstock transformation with PyMC Marketing integration
  - [ ] 3.2 Implement Hill saturation function with configurable parameters
  - [ ] 3.3 Implement Logistic saturation function with configurable parameters
  - [ ] 3.4 Create transformation function registry for string-based selection
  - [ ] 3.5 Add support for allowing different adstock/saturation combinations per channel
  - [ ] 3.6 Implement transformation parameter validation
  - [ ] 3.7 Use PyMC MArketing MediaTransforms for applying multiple transforms sequentially

- [ ] 4.0 Geographic Region Management
  - [ ] 4.1 Create region configuration system supporting one or more regions
  - [ ] 4.2 Implement generation of region data with channel parameters drawn from a shared underlying distribution.
  - [ ] 4.3 Add regional baseline sales rate variation system
  - [ ] 4.4 Implement channel effectiveness variation per region
  - [ ] 4.5 Create region similarity controls
  - [ ] 4.7 Implement region-specific parameter validation

- [ ] 5.0 Ground Truth Calculation and ROAS
  - [ ] 5.1 Implement true parameter value tracking (betas, alphas, kappas, etc.)
  - [ ] 5.2 Create channel contribution calculation over time
  - [ ] 5.3 Implement ROAS computation for each channel and region
  - [ ] 5.4 Add attribution percentage calculations
  - [ ] 5.5 Implement baseline component tracking (intercept, trend, seasonality, controls)
  - [ ] 5.6 Create ground truth data structure and output formatting
  - [ ] 5.7 Add ground truth validation and consistency checks
  - [ ] 5.8 Ensure output dataframe follows required schema with proper column naming and data types

- [ ] 6.0 Visualization and Validation Functions
  - [ ] 6.1 Implement channel spend pattern visualization functions
  - [ ] 6.2 Create channel contribution over time plotting utilities
  - [ ] 6.3 Add ROAS visualization and comparison charts
  - [ ] 6.4 Implement regional comparison visualization functions
  - [ ] 6.5 Create data quality validation plots and diagnostics
  - [ ] 6.7 Implement comprehensive data validation with warning systems

- [ ] 7.0 Testing and Documentation
  - [ ] 7.1 Create unit tests for all channel generation functions
  - [ ] 7.2 Implement unit tests for region management and hierarchical structure
  - [ ] 7.3 Add unit tests for adstock and saturation transformations
  - [ ] 7.4 Create unit tests for ground truth calculations and ROAS
  - [ ] 7.5 Implement integration tests for complete data generation pipeline
  - [ ] 7.6 Add performance benchmarks for scalability testing
  - [ ] 7.7 Create comprehensive docstrings and API documentation
  - [ ] 7.8 Develop example scripts and Marimo notebook demonstrations
  - [ ] 7.9 Update README with usage examples and installation instructions
  - [ ] 7.10 Add tests for output schema compliance and data integrity validation 