#!/usr/bin/env python3
"""
Test script to verify that all presets pass schema validation.
"""

from mmm_param_recovery.data_generator.presets import (
    get_preset_config, 
    list_available_presets, 
    customize_preset
)
from mmm_param_recovery.data_generator.config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig


def test_all_presets_pass_validation():
    """Test that all available presets pass schema validation."""
    available_presets = list_available_presets()
    
    print(f"Testing {len(available_presets)} presets for schema validation...")
    
    for preset_name, description in available_presets.items():
        print(f"Testing preset: {preset_name} - {description}")
        
        try:
            # Get the preset configuration
            config = get_preset_config(preset_name)
            
            # Verify it's a valid MMMDataConfig instance
            assert isinstance(config, MMMDataConfig), f"Preset {preset_name} should return MMMDataConfig"
            
            # Verify all nested configurations are valid
            assert isinstance(config.regions, RegionConfig), f"Preset {preset_name} should have valid RegionConfig"
            assert isinstance(config.transforms, TransformConfig), f"Preset {preset_name} should have valid TransformConfig"
            
            # Verify channels are valid
            assert len(config.channels) > 0, f"Preset {preset_name} should have at least one channel"
            for i, channel in enumerate(config.channels):
                assert isinstance(channel, ChannelConfig), f"Preset {preset_name} channel {i} should be ChannelConfig"
            
            # Verify basic properties
            assert config.n_periods > 0, f"Preset {preset_name} should have positive n_periods"
            assert config.regions.n_regions > 0, f"Preset {preset_name} should have positive n_regions"
            
            print(f"  ✅ {preset_name} passed validation")
            
        except Exception as e:
            print(f"  ❌ {preset_name} failed validation: {str(e)}")
            raise AssertionError(f"Preset {preset_name} failed validation: {str(e)}")
    
    print("✅ All presets passed schema validation!")


def test_preset_channel_names_compliance():
    """Test that all presets use valid channel names (no underscores)."""
    available_presets = list_available_presets()
    
    print("Testing channel name compliance (no underscores)...")
    
    for preset_name in available_presets.keys():
        try:
            config = get_preset_config(preset_name)
            
            for i, channel in enumerate(config.channels):
                if "_" in channel.name:
                    raise ValueError(
                        f"Preset {preset_name} channel {i} '{channel.name}' contains underscore. "
                        f"Use hyphens instead."
                    )
            
            print(f"  ✅ {preset_name} channel names are compliant")
            
        except Exception as e:
            print(f"  ❌ {preset_name} channel names failed: {str(e)}")
            raise
    
    print("✅ All presets have compliant channel names!")


def test_preset_data_generation():
    """Test that all presets can generate data successfully."""
    available_presets = list_available_presets()
    
    print("Testing data generation for all presets...")
    
    # Import here to avoid circular imports
    from mmm_param_recovery.data_generator import generate_mmm_dataset
    
    for preset_name in available_presets.keys():
        try:
            config = get_preset_config(preset_name)
            
            # Generate dataset
            result = generate_mmm_dataset(config)
            
            # Verify result structure
            assert 'data' in result, f"Preset {preset_name} should return data"
            assert 'config' in result, f"Preset {preset_name} should return config"
            
            data = result['data']
            
            # Verify basic data properties
            assert len(data) > 0, f"Preset {preset_name} should generate non-empty data"
            assert 'date' in data.columns, f"Preset {preset_name} should have date column"
            assert 'geo' in data.columns, f"Preset {preset_name} should have geo column"
            assert 'y' in data.columns, f"Preset {preset_name} should have y column"
            
            # Verify expected number of rows
            expected_rows = config.n_periods * config.regions.n_regions
            assert len(data) == expected_rows, (
                f"Preset {preset_name} should have {expected_rows} rows, got {len(data)}"
            )
            
            print(f"  ✅ {preset_name} data generation successful")
            
        except Exception as e:
            print(f"  ❌ {preset_name} data generation failed: {str(e)}")
            raise AssertionError(f"Preset {preset_name} data generation failed: {str(e)}")
    
    print("✅ All presets can generate data successfully!")


def test_customize_preset():
    """Test that preset customization works correctly."""
    print("Testing preset customization...")
    
    try:
        # Test basic customization
        config = customize_preset('basic', n_periods=100, seed=999)
        assert config.n_periods == 100, "Customization should override n_periods"
        assert config.seed == 999, "Customization should override seed"
        
        # Test nested customization
        config = customize_preset('basic', **{
            'transforms.adstock_alpha': 0.8,
            'regions.n_regions': 3
        })
        assert config.transforms.adstock_alpha == 0.8, "Nested customization should work"
        assert config.regions.n_regions == 3, "Nested customization should work"
        
        print("  ✅ Preset customization works correctly")
        
    except Exception as e:
        print(f"  ❌ Preset customization failed: {str(e)}")
        raise
    
    print("✅ Preset customization test passed!")


def test_preset_edge_cases():
    """Test edge cases and error handling for presets."""
    print("Testing preset edge cases...")
    
    # Test invalid preset name
    try:
        get_preset_config('nonexistent_preset')
        raise AssertionError("Should have raised ValueError for invalid preset")
    except ValueError as e:
        assert "Unknown preset" in str(e), "Should provide clear error message"
        print("  ✅ Invalid preset name handled correctly")
    
    # Test invalid customization
    try:
        customize_preset('basic', invalid_param=123)
        raise AssertionError("Should have raised ValueError for invalid override")
    except ValueError as e:
        assert "Invalid override key" in str(e), "Should provide clear error message"
        print("  ✅ Invalid customization handled correctly")
    
    print("✅ Preset edge cases handled correctly!")