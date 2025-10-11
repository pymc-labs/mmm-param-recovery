"""
Tests for visualization functions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta

from mmm_data_generator.visualization import (
    plot_channel_spend,
    plot_channel_contributions,
    plot_roas_comparison,
    plot_regional_comparison,
    plot_data_quality
)


class TestChannelContributionPlotting:
    """Test channel contribution plotting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample data with contributions
        dates = pd.date_range('2020-01-01', periods=52, freq='W')
        regions = ['US', 'EU']
        
        # Create multi-index
        index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
        
        # Sample contribution data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'contribution_x1_tv': np.random.normal(100, 20, len(index)),
            'contribution_x2_digital': np.random.normal(80, 15, len(index)),
            'contribution_x3_radio': np.random.normal(60, 10, len(index)),
            'baseline_sales': np.random.normal(500, 50, len(index)),
            'y': np.random.normal(800, 100, len(index))
        }, index=index).reset_index()
    
    def test_plot_channel_contributions_basic(self):
        """Test basic channel contribution plotting."""
        fig = plot_channel_contributions(self.test_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    def test_plot_channel_contributions_with_channel_filter(self):
        """Test plotting with channel filtering."""
        fig = plot_channel_contributions(
            self.test_data, 
            channels=['tv', 'digital']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_channel_contributions_with_region_filter(self):
        """Test plotting with region filtering."""
        fig = plot_channel_contributions(
            self.test_data, 
            regions=['US']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_channel_contributions_different_styles(self):
        """Test plotting with different styles."""
        styles = ['line', 'area', 'bar']
        
        for style in styles:
            fig = plot_channel_contributions(
                self.test_data, 
                style=style
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plot_channel_contributions_no_baseline(self):
        """Test plotting without baseline."""
        fig = plot_channel_contributions(
            self.test_data, 
            show_baseline=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_channel_contributions_invalid_data(self):
        """Test error handling for invalid data."""
        with pytest.raises(ValueError, match="data must be a pandas DataFrame"):
            plot_channel_contributions("invalid_data")  # type: ignore
    
    def test_plot_channel_contributions_missing_columns(self):
        """Test error handling for missing required columns."""
        invalid_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_channel_contributions(invalid_data)
    
    def test_plot_channel_contributions_no_contribution_columns(self):
        """Test error handling when no contribution columns found."""
        data_no_contributions = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'geo': ['US'] * 10,
            'other_col': np.random.randn(10)
        })
        
        with pytest.raises(ValueError, match="No contribution columns"):
            plot_channel_contributions(data_no_contributions)
    
    def test_plot_channel_contributions_invalid_channels(self):
        """Test error handling for invalid channel names."""
        with pytest.raises(ValueError, match="No matching contribution columns found"):
            plot_channel_contributions(self.test_data, channels=['nonexistent'])
    
    def test_plot_channel_contributions_invalid_regions(self):
        """Test error handling for invalid region names."""
        with pytest.raises(ValueError, match="No data found for specified regions"):
            plot_channel_contributions(self.test_data, regions=['nonexistent'])


class TestChannelSpendPlotting:
    """Test channel spend plotting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample spend data
        dates = pd.date_range('2020-01-01', periods=52, freq='W')
        regions = ['US', 'EU']
        
        # Create multi-index
        index = pd.MultiIndex.from_product([dates, regions], names=['date', 'geo'])
        
        # Sample spend data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'x1_tv': np.random.normal(1000, 200, len(index)),
            'x2_digital': np.random.normal(800, 150, len(index)),
            'x3_radio': np.random.normal(600, 100, len(index)),
            'y': np.random.normal(800, 100, len(index))
        }, index=index).reset_index()
    
    def test_plot_channel_spend_basic(self):
        """Test basic channel spend plotting."""
        fig = plot_channel_spend(self.test_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    def test_plot_channel_spend_with_filters(self):
        """Test plotting with channel and region filters."""
        fig = plot_channel_spend(
            self.test_data, 
            channels=['tv', 'digital'],
            regions=['US']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_channel_spend_different_styles(self):
        """Test plotting with different styles."""
        styles = ['line', 'area', 'bar']
        
        for style in styles:
            fig = plot_channel_spend(
                self.test_data, 
                style=style
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestPlaceholderFunctions:
    """Test placeholder visualization functions."""
    
    def test_plot_roas_comparison(self):
        """Test ROAS comparison plotting placeholder."""
        ground_truth = {'roas_values': {'US': {'tv': 2.5, 'digital': 1.8}}}
        fig = plot_roas_comparison(ground_truth)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_regional_comparison(self):
        """Test regional comparison plotting placeholder."""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'geo': ['US', 'EU'] * 5,
            'y': np.random.randn(10)
        })
        fig = plot_regional_comparison(data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_data_quality(self):
        """Test data quality plotting placeholder."""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'geo': ['US'] * 10,
            'y': np.random.randn(10)
        })
        fig = plot_data_quality(data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig) 