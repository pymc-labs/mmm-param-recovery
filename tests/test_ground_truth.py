"""
Tests for ground truth calculation and validation functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mmm_param_recovery.data_generator.config import (
    MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig
)
from mmm_param_recovery.data_generator.ground_truth import (
    calculate_roas_values,
    calculate_attribution_percentages
)
from mmm_param_recovery.data_generator.validation import (
    validate_ground_truth,
    _validate_transformation_parameters,
    _validate_baseline_components,
    _validate_roas_values,
    _validate_attribution_percentages,
    _validate_transformed_spend
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    channels = [
        ChannelConfig(name="tv", pattern="linear_trend", base_spend=1000),
        ChannelConfig(name="digital", pattern="seasonal", base_spend=500)
    ]
    
    regions = RegionConfig(
        n_regions=2,
        region_names=["US", "EU"],
        base_sales_rate=1000,
        sales_volatility=0.1
    )
    
    transforms = TransformConfig(
        adstock_kwargs={"alpha": 0.5},
        saturation_kwargs={"half_max_effective": 0.5}
    )
    
    return MMMDataConfig(
        n_periods=52,
        channels=channels,
        regions=regions,
        transforms=transforms,
        seed=42
    )


@pytest.fixture
def sample_spend_data():
    """Create sample spend data for testing."""
    dates = pd.date_range('2020-01-01', periods=52, freq='W')
    regions = ["US", "EU"]
    
    # Create multi-index
    index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
    
    data = {
        'x1_tv': np.random.uniform(800, 1200, len(index)),
        'x2_digital': np.random.uniform(400, 600, len(index))
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    dates = pd.date_range('2020-01-01', periods=52, freq='W')
    regions = ["US", "EU"]
    
    # Create multi-index
    index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
    
    data = {
        'contribution_x1_tv': np.random.uniform(50, 150, len(index)),
        'contribution_x2_digital': np.random.uniform(20, 80, len(index))
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_ground_truth(sample_config, sample_spend_data, sample_transformed_data):
    """Create sample ground truth data for testing."""
    # Calculate ROAS and attribution
    roas_values = calculate_roas_values(sample_spend_data, sample_transformed_data, sample_config)
    attribution_percentages = calculate_attribution_percentages(sample_transformed_data, sample_config)
    
    # Create transformation parameters
    transformation_params = {
        'channels': {
            'tv': {
                'US': {'adstock_rate': 0.5, 'saturation_params': {'half_max_effective': 0.5}},
                'EU': {'adstock_rate': 0.6, 'saturation_params': {'half_max_effective': 0.6}}
            },
            'digital': {
                'US': {'adstock_rate': 0.4, 'saturation_params': {'half_max_effective': 0.4}},
                'EU': {'adstock_rate': 0.5, 'saturation_params': {'half_max_effective': 0.5}}
            }
        }
    }
    
    # Create baseline components
    dates = pd.date_range('2020-01-01', periods=52, freq='W')
    regions = ["US", "EU"]
    index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
    
    baseline_components = pd.DataFrame({
        'base_sales': np.random.uniform(800, 1200, len(index)),
        'trend': np.random.uniform(-10, 10, len(index)),
        'seasonal': np.random.uniform(-50, 50, len(index)),
        'baseline_sales': np.random.uniform(900, 1100, len(index))
    }, index=index)
    
    return {
        'transformation_parameters': transformation_params,
        'baseline_components': baseline_components,
        'roas_values': roas_values,
        'attribution_percentages': attribution_percentages,
        'transformed_spend': sample_transformed_data
    }


def test_calculate_roas_values(sample_config, sample_spend_data, sample_transformed_data):
    """Test ROAS calculation."""
    roas_values = calculate_roas_values(sample_spend_data, sample_transformed_data, sample_config)
    
    # Check structure
    assert isinstance(roas_values, dict)
    assert "US" in roas_values
    assert "EU" in roas_values
    
    # Check that all channels are present
    for region in roas_values:
        assert "tv" in roas_values[region]
        assert "digital" in roas_values[region]
        
        # Check that ROAS values are positive
        for channel, roas in roas_values[region].items():
            assert isinstance(roas, (int, float))
            assert roas >= 0


def test_calculate_attribution_percentages(sample_config, sample_transformed_data):
    """Test attribution percentage calculation."""
    attribution_percentages = calculate_attribution_percentages(sample_transformed_data, sample_config)
    
    # Check structure
    assert isinstance(attribution_percentages, dict)
    assert "US" in attribution_percentages
    assert "EU" in attribution_percentages
    
    # Check that all channels are present
    for region in attribution_percentages:
        assert "tv" in attribution_percentages[region]
        assert "digital" in attribution_percentages[region]
        
        # Check that percentages sum to approximately 100%
        total_attribution = sum(attribution_percentages[region].values())
        assert 99.5 <= total_attribution <= 100.5
        
        # Check individual percentage ranges
        for channel, percentage in attribution_percentages[region].items():
            assert isinstance(percentage, (int, float))
            assert 0 <= percentage <= 100


def test_validate_ground_truth_complete(sample_config, sample_ground_truth):
    """Test ground truth validation with complete data."""
    issues = validate_ground_truth(sample_ground_truth, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_ground_truth_missing_keys(sample_config, sample_ground_truth):
    """Test ground truth validation with missing keys."""
    # Remove a key
    incomplete_ground_truth = sample_ground_truth.copy()
    del incomplete_ground_truth['roas_values']
    
    issues = validate_ground_truth(incomplete_ground_truth, sample_config)
    
    # Should detect missing key
    assert any("Missing ground truth key: roas_values" in issue for issue in issues)


def test_validate_transformation_parameters_valid(sample_config, sample_ground_truth):
    """Test transformation parameters validation with valid data."""
    transform_params = sample_ground_truth['transformation_parameters']
    issues = _validate_transformation_parameters(transform_params, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_transformation_parameters_invalid(sample_config):
    """Test transformation parameters validation with invalid data."""
    # Create invalid transformation parameters
    invalid_transform_params = {
        'channels': {
            'tv': {
                'US': {'adstock_rate': 1.5, 'saturation_params': {'half_max_effective': -0.5}}  # Invalid values
            }
        }
    }
    
    issues = _validate_transformation_parameters(invalid_transform_params, sample_config)
    
    # Should detect invalid values
    assert any("Invalid adstock_rate" in issue for issue in issues)
    assert any("Invalid half_max_effective" in issue for issue in issues)


def test_validate_baseline_components_valid(sample_config, sample_ground_truth):
    """Test baseline components validation with valid data."""
    baseline_data = sample_ground_truth['baseline_components']
    issues = _validate_baseline_components(baseline_data, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_baseline_components_invalid(sample_config):
    """Test baseline components validation with invalid data."""
    # Create invalid baseline data
    dates = pd.date_range('2020-01-01', periods=52, freq='W')
    regions = ["US", "EU"]
    index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
    
    invalid_baseline_data = pd.DataFrame({
        'base_sales': np.random.uniform(800, 1200, len(index)),
        'trend': np.random.uniform(-10, 10, len(index)),
        'seasonal': np.random.uniform(-50, 50, len(index)),
        # Missing 'baseline_sales' column
    }, index=index)
    
    issues = _validate_baseline_components(invalid_baseline_data, sample_config)
    
    # Should detect missing column
    assert any("Missing baseline component columns" in issue for issue in issues)


def test_validate_roas_values_valid(sample_config, sample_ground_truth):
    """Test ROAS values validation with valid data."""
    roas_values = sample_ground_truth['roas_values']
    issues = _validate_roas_values(roas_values, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_roas_values_invalid(sample_config):
    """Test ROAS values validation with invalid data."""
    # Create invalid ROAS values
    invalid_roas_values = {
        'US': {
            'tv': -0.5,  # Negative ROAS
            'digital': float('inf')  # Infinite ROAS
        }
    }
    
    issues = _validate_roas_values(invalid_roas_values, sample_config)
    
    # Should detect invalid values
    assert any("Negative ROAS value" in issue for issue in issues)
    assert any("Infinite ROAS value" in issue for issue in issues)


def test_validate_attribution_percentages_valid(sample_config, sample_ground_truth):
    """Test attribution percentages validation with valid data."""
    attribution_values = sample_ground_truth['attribution_percentages']
    issues = _validate_attribution_percentages(attribution_values, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_attribution_percentages_invalid(sample_config):
    """Test attribution percentages validation with invalid data."""
    # Create invalid attribution percentages
    invalid_attribution_values = {
        'US': {
            'tv': 150.0,  # > 100%
            'digital': -10.0  # Negative
        }
    }
    
    issues = _validate_attribution_percentages(invalid_attribution_values, sample_config)
    
    # Should detect invalid values
    assert any("Attribution percentage" in issue and "> 100%" in issue for issue in issues)
    assert any("Negative attribution percentage" in issue for issue in issues)


def test_validate_transformed_spend_valid(sample_config, sample_ground_truth):
    """Test transformed spend validation with valid data."""
    transformed_data = sample_ground_truth['transformed_spend']
    issues = _validate_transformed_spend(transformed_data, sample_config)
    
    # Should have no issues with valid data
    assert len(issues) == 0


def test_validate_transformed_spend_invalid(sample_config):
    """Test transformed spend validation with invalid data."""
    # Create invalid transformed data
    dates = pd.date_range('2020-01-01', periods=52, freq='W')
    regions = ["US", "EU"]
    index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
    
    invalid_transformed_data = pd.DataFrame({
        'contribution_x1_tv': np.random.uniform(-50, 150, len(index)),  # Some negative values
        'contribution_x2_digital': np.random.uniform(20, 80, len(index))
    }, index=index)
    
    issues = _validate_transformed_spend(invalid_transformed_data, sample_config)
    
    # Should detect negative values
    assert any("negative contribution values" in issue for issue in issues)
