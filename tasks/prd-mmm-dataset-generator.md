# Product Requirements Document: MMM Dataset Generator

## Introduction/Overview

The MMM Dataset Generator is a lightweight Python module designed to create reproducible datasets for benchmarking Marketing Mix Models (MMM). The module addresses the need for standardized, controlled datasets that allow researchers and practitioners to test parameter recovery, channel contribution analysis, and Return on Ad Spend (ROAS) calculations in a controlled environment.

The module will generate synthetic marketing data with known ground truth parameters, enabling users to validate their MMM implementations and compare different modeling approaches. This is particularly valuable given the privacy constraints and limitations of real-world marketing data.

## Goals

1. **Reproducible Data Generation**: Create datasets with known ground truth parameters that can be consistently reproduced using seed values
2. **Flexible Channel Configuration**: Support multiple advertising channel types with configurable patterns and behaviors
3. **Multi-Regional Support**: Generate data for multiple geographic regions with controllable similarity levels
4. **Comprehensive Ground Truth**: Provide complete parameter values, contributions, and ROAS calculations for validation
5. **PyMC Marketing Integration**: Ensure compatibility with the PyMC Marketing library for seamless workflow integration
6. **Educational Value**: Serve as a learning tool for understanding MMM concepts and parameter recovery

## User Stories

1. **As a researcher**, I want to generate synthetic MMM datasets with known parameters so that I can validate my model's ability to recover true parameter values.

2. **As a data scientist**, I want to test different MMM configurations on controlled data so that I can compare the performance of various adstock and saturation functions.

3. **As a marketing analyst**, I want to understand how channel contributions vary across different geographic regions so that I can optimize budget allocation strategies.

4. **As a student**, I want to experiment with MMM concepts using synthetic data so that I can learn without needing access to real marketing data.

5. **As a consultant**, I want to demonstrate MMM capabilities to clients using realistic but controlled data so that I can showcase the methodology without privacy concerns.

## Functional Requirements

### 1. Core Data Generation Function
- The system must provide a single function `generate_mmm_dataset()` as the main entry point
- The function must accept configuration parameters for all data generation aspects
- The function must return a pandas DataFrame with the generated data

### 2. Channel Configuration System
- The system must support all predefined channel types: linear trend, seasonal, delayed start, and on/off channels
- The system must allow users to configure existing channel patterns with custom parameters
- The system must support custom user-defined channel patterns through function inputs
- Each channel must support independent configuration of spend patterns, adstock parameters, and saturation parameters

### 3. Geographic Region Management
- The system must support 1-50 geographic regions
- The system must create individual baseline sales, trend, and seasonal patterns for each region
- The system must vary channel parameters by applying small scaling factors to base spends and other parameters
- The system must vary transformation parameters slightly to ensure similar behavior with regional variations
- The system must maintain reproducibility through proper seed management for each region

### 4. Adstock and Saturation Functions
- The system must default to geometric adstock transformation
- The system must support both Hill and Logistic saturation functions by default
- The system must allow mixing different adstock/saturation combinations per channel
- The system must accept string identifiers of adstock/saturation functions corresponding to PyMC Marketing functions
- The system must support slight regional variations in transformation parameters while maintaining similar behavior patterns

### 5. Ground Truth Information
- The system must provide true parameter values (betas, alphas, kappas, etc.) for all channels
- The system must calculate true channel contributions over time
- The system must compute true ROAS values for each channel
- The system must provide true attribution percentages
- The system must include all baseline components (intercept, trend, seasonality, control variables)

### 6. Visualization Capabilities
- The system must include basic plotting functions similar to those in the reference notebook [data.ipynb]
- The system must provide visualizations for channel spend patterns, contributions, and ROAS
- The system must support regional comparison visualizations
- The system must include validation plots showing data quality and parameter relationships

### 7. Reproducibility and Randomization
- The system must use a single seed for complete reproducibility
- The system must support generating multiple datasets with different seeds
- The system must maintain consistent random number generation across all components

### 8. Data Validation
- The system must perform basic validation (non-negative values, reasonable ranges)
- The system must check for data quality issues and provide warnings

### 9. Output Format and Integration
- The system must output pandas DataFrames as the primary format for the generated data and the true contributions. The ground truth parameters should be provided in a dictionary.
- The system must follow the specified output schema with columns: `date`, `geo`, `y`, `x1`, `x2`, ..., `c1`, `c2`, ...
- The system must ensure each row has a unique `date` and `geo` combination
- The system must maintain data integrity with no missing values and non-negative numeric values

### 10. Performance and Scalability
- The system must be scalable to handle any dataset size (50-200 time periods, 1-100 channels, 1-100 regions)
- The system must prioritize a simple and extensible implementation
- The system must provide reasonable performance for typical use cases
- The system must support both small-scale testing and large-scale simulation scenarios

## Non-Goals (Out of Scope)

1. **Real Data Integration**: The module will not import or process real marketing data
2. **Advanced Statistical Modeling**: The module will not include MMM fitting or inference capabilities
3. **Web Interface**: The module will not provide a web-based interface for data generation
4. **Database Integration**: The module will not include database storage or retrieval functionality
5. **Real-time Data Streaming**: The module will not support real-time data generation or streaming
6. **Advanced Visualization**: The module will not include interactive or advanced plotting capabilities beyond basic charts
7. **Machine Learning Features**: The module will not include automated parameter optimization or ML-based pattern generation

## Design Considerations

### API Design
- Follow Python conventions for function naming and parameter organization
- Use dataclasses or configuration objects for complex parameter sets
- Provide sensible defaults for all parameters
- Include comprehensive docstrings with examples

### Data Structure
- Use pandas DataFrames as the primary data structure
- Include clear column naming conventions 
- Separate input data from ground truth information
- Provide both raw and transformed data where applicable

### Output Schema Specification
The resulting dataframe must follow a specific schema structure:

**Required Columns:**
- `date`: datetime values representing the time period for each observation
- `geo`: string values representing the geographic region for each observation
- `y`: numeric values representing total sales for the specific date and region combination

**Channel Columns:**
- `x1-*`, `x2-*`, `x3`, ...: numeric values representing spend for each advertising channel. - The name might be followed with a hyphenated suffix.
- Channel columns must be named sequentially starting with "x1"

**Control Variable Columns:**
- `c1`, `c2-*`, `c3`, ...: numeric values representing control variables (e.g., price, promotions, external factors)
- The name might be followed with a hyphenated suffix.
- Control variable columns must be named sequentially starting with "c1"

**Data Integrity Requirements:**
- Each row must have a unique combination of `date` and `geo` values
- All numeric columns must contain non-negative values
- No missing values are allowed in any column
- The dataframe must be sorted by `date` and then by `geo` for consistent ordering

### Visualization Design
- Use matplotlib and seaborn for consistent plotting
- Follow the visualization patterns established in the reference notebook [data.ipynb]
- Provide both individual plots and combined summary visualizations
- Include proper labeling and legends for all charts

## Technical Considerations

### Dependencies
- Primary dependencies: pandas, numpy, matplotlib, seaborn
- PyMC Marketing integration for adstock and saturation functions
- Ensure compatibility with Python 3.8+ and common scientific computing environments

### Performance Optimization
- Use vectorized operations where possible
- Minimize memory usage for large datasets
- Provide progress indicators for long-running operations

### Error Handling
- Provide clear error messages for invalid parameter combinations
- Include parameter validation with helpful suggestions
- Handle edge cases gracefully (e.g., zero spend channels)
- Provide warnings for potentially problematic configurations

### Testing Strategy
- Unit tests for individual components
- Integration tests for complete data generation workflows
- Performance benchmarks for scalability testing

## Success Metrics

1. **Reproducibility**: Identical datasets should be generated when using the same seed and parameters

2. **Flexibility**: The module should support at least 80% of common MMM use cases without requiring custom code

3. **Performance**: Data generation should complete within reasonable timeframes (e.g., <30 seconds for typical datasets)

5. **Usability**: Users should be able to generate useful datasets with minimal configuration (using sensible defaults)

6. **Documentation Quality**: Users should be able to understand and use the module effectively through documentation and examples

## Open Questions

1. **Advanced Channel Patterns**: Should the module support more complex channel patterns like step functions, cyclical patterns, or event-driven spikes?

2. **External Factors**: Should the module include generation of external factors (economy, weather, competitor activity) that affect sales?

3. **Seasonality Complexity**: Should the module support multiple seasonal patterns (weekly, monthly, quarterly) beyond the basic annual seasonality?

4. **Validation Metrics**: What specific validation metrics should be included to assess the quality of generated datasets?

5. **Export Formats**: Should the module support additional export formats beyond pandas DataFrames (e.g., CSV, Parquet, JSON)?

6. **Configuration Persistence**: Should the module support saving and loading configuration objects for reproducible research workflows?
