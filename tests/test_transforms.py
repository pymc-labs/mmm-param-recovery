"""
Unit tests for adstock and saturation transformations.
"""

import pytest
import numpy as np
from mmm_param_recovery.data_generator.transforms import (
    apply_transformations,
)
from mmm_param_recovery.data_generator.config import TransformConfig, ChannelConfig
from mmm_param_recovery.data_generator.transforms import apply_transform
from pymc_marketing.mmm import transformers


class TestTransformationFunctions:
    """Test individual transformation functions."""
    
    def test_apply_transform(self):
        """Test geometric adstock transformation."""
        # Create test data
        spend_data = np.array([100, 200, 300, 400, 500])
        
        # Apply geometric adstock
        result = apply_transform(spend_data, transformers.geometric_adstock, alpha=0.5)
        
        # Check that result is not None and has correct shape
        assert result is not None
        assert len(result) == len(spend_data)
    

class TestTransformConfig:
    """Test TransformConfig with per-channel parameters."""
    
    def test_single_parameter_config(self):
        """Test TransformConfig with single parameter values."""
        config = TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun="hill_function",
            saturation_kwargs={"kappa": 2000.0}
        )
        
        # Test getter methods
        assert config.get_adstock_kwargs(0) == {"alpha": 0.6}
        assert config.get_saturation_kwargs(0) == {"kappa": 2000.0}
    
    def test_list_parameter_config(self):
        """Test TransformConfig with list parameter values."""
        config = TransformConfig(
            adstock_fun=["geometric_adstock", "geometric_adstock", "geometric_adstock"],
            adstock_kwargs=[{"alpha": 0.6}, {"alpha": 0.7}, {"alpha": 0.8}],
            saturation_fun=["hill_function", "hill_function", "hill_function"],
            saturation_kwargs=[{"kappa": 2000.0}, {"kappa": 3000.0}, {"kappa": 4000.0}]
        )
        
        # Test getter methods with cycling
        assert config.get_adstock_kwargs(0) == {"alpha": 0.6}
        assert config.get_adstock_kwargs(1) == {"alpha": 0.7}
        assert config.get_adstock_kwargs(2) == {"alpha": 0.8}
        assert config.get_adstock_kwargs(3) == {"alpha": 0.6}  # Should cycle
        
        assert config.get_saturation_kwargs(0) == {"kappa": 2000.0}
        assert config.get_saturation_kwargs(1) == {"kappa": 3000.0}
        assert config.get_saturation_kwargs(2) == {"kappa": 4000.0}  # Should cycle


class TestApplyTransformations:
    """Test the main apply_transformations function."""
    
    def test_geometric_adstock_transformation(self):
        """Test geometric adstock transformation."""
        spend_data = np.array([100, 200, 300, 400, 500])
        config = TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun=None
        )
        
        result = apply_transformations(spend_data, config)
        assert result is not None
        assert len(result) == len(spend_data)
    
    def test_hill_saturation_transformation(self):
        """Test Hill saturation transformation."""
        spend_data = np.array([100, 200, 300, 400, 500])
        config = TransformConfig(
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.5, "kappa": 300.0}
        )
        
        result = apply_transformations(spend_data, config)
        assert result is not None
        assert len(result) == len(spend_data)
    
    def test_combined_adstock_and_saturation(self):
        """Test combined adstock and saturation transformations."""
        spend_data = np.array([100, 200, 300, 400, 500])
        config = TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.5, "kappa": 300.0}
        )
        
        result = apply_transformations(spend_data, config)
        assert result is not None
        assert len(result) == len(spend_data)
    
    def test_per_channel_parameters(self):
        """Test per-channel parameter selection."""
        spend_data = np.array([100, 200, 300, 400, 500])
        config = TransformConfig(
            adstock_fun=["geometric_adstock", "weibull_adstock"],
            adstock_kwargs=[{"alpha": 0.6}, {"lam": 2.0, "k": 3.0}],
            saturation_fun=["hill_function", "hill_function"],
            saturation_kwargs=[{"slope": 1.0, "kappa": 2000.0}, {"slope": 1.5, "kappa": 3000.0}]
        )
        
        # Test with different channel indices
        result1 = apply_transformations(spend_data, config, channel_idx=0)
        result2 = apply_transformations(spend_data, config, channel_idx=1)
        
        assert result1 is not None
        assert result2 is not None
        assert len(result1) == len(spend_data)
        assert len(result2) == len(spend_data)


if __name__ == "__main__":
    pytest.main([__file__]) 