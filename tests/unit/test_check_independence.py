"""
test_check_independence.py
Test suite for the check_independence module.

This module contains unit tests for the check_independence function,
including input validation, edge cases, and functional correctness tests.
"""

import pytest
import pandas as pd
import numpy as np
from lrassume import check_independence


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_invalid_df_type_list(self):
        """Test that passing a list instead of DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            check_independence([1, 2, 3], "y")
    
    def test_invalid_df_type_dict(self):
        """Test that passing a dict instead of DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            check_independence({"x": [1, 2], "y": [3, 4]}, "y")
    
    def test_invalid_target_type(self):
        """Test that passing non-string target raises TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with pytest.raises(TypeError, match="must be a string"):
            check_independence(df, 123)
    
    def test_target_not_in_dataframe(self):
        """Test that specifying non-existent target column raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with pytest.raises(ValueError, match="not found in DataFrame"):
            check_independence(df, "z")
    
    def test_target_not_numeric(self):
        """Test that non-numeric target column raises TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        with pytest.raises(TypeError, match="Target column must be numeric"):
            check_independence(df, "y")
    
    def test_no_numeric_features(self):
        """Test that DataFrame with no numeric features raises ValueError."""
        df = pd.DataFrame({"cat": ["a", "b", "c"], "y": [1, 2, 3]})
        with pytest.raises(ValueError, match="No numeric feature columns found"):
            check_independence(df, "y")
    
    def test_missing_values_in_features(self):
        """Test that missing values in feature columns raise ValueError."""
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [4, 5, 6]})
        with pytest.raises(ValueError, match="contains missing values"):
            check_independence(df, "y")
    
    def test_missing_values_in_target(self):
        """Test that missing values in target column raise ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, np.nan, 6]})
        with pytest.raises(ValueError, match="contains missing values"):
            check_independence(df, "y")


class TestFunctionality:
    """Tests for correct functionality and calculations."""
    
    def test_basic_functionality(self):
        """Test basic function execution with simple data."""
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 5, 7, 8],
            "y": [10, 20, 25, 35, 40]
        })
        result = check_independence(df, target="y")
        
        assert isinstance(result, dict)
        assert 'dw_statistic' in result
        assert 'is_independent' in result
        assert 'message' in result
    
    def test_return_types(self):
        """Test that return values have correct types."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10]
        })
        result = check_independence(df, target="y")
        
        assert isinstance(result['dw_statistic'], float)
        assert isinstance(result['is_independent'], bool)
        assert isinstance(result['message'], str)
    
    def test_dw_statistic_range(self):
        """Test that Durbin-Watson statistic is in valid range [0, 4]."""
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randn(50)
        })
        result = check_independence(df, target="y")
        
        assert 0 <= result['dw_statistic'] <= 4
    
    def test_independent_residuals(self):
        """Test with data that should have independent residuals."""
        np.random.seed(42)
        x = np.arange(100)
        y = 2 * x + 3 + np.random.randn(100) * 5  # Random noise
        df = pd.DataFrame({"x": x, "y": y})
        
        result = check_independence(df, target="y")
        
        # Should detect independence (DW near 2)
        assert result['is_independent'] == True
        assert 1.5 <= result['dw_statistic'] <= 2.5
    
    def test_positive_autocorrelation(self):
        """Test with data exhibiting positive autocorrelation."""
        np.random.seed(42)
        n = 100
        x = np.arange(n)
        # Create autocorrelated errors
        errors = np.zeros(n)
        errors[0] = np.random.randn()
        for i in range(1, n):
            errors[i] = 0.8 * errors[i-1] + np.random.randn() * 0.5
        y = 2 * x + errors
        df = pd.DataFrame({"x": x, "y": y})
        
        result = check_independence(df, target="y")
        
        # Should detect positive autocorrelation (DW < 1.5)
        assert result['dw_statistic'] < 1.5
        assert "Positive autocorrelation" in result['message']
    
    def test_multiple_features(self):
        """Test with multiple feature columns."""
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [5, 4, 3, 2, 1],
            "x3": [2, 3, 4, 5, 6],
            "y": [10, 15, 20, 25, 30]
        })
        result = check_independence(df, target="y")
        
        assert isinstance(result['dw_statistic'], float)
        assert isinstance(result['is_independent'], bool)
    
    def test_with_non_numeric_columns_ignored(self):
        """Test that non-numeric columns are properly ignored."""
        df = pd.DataFrame({
            "category": ["a", "b", "c", "d", "e"],
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10]
        })
        result = check_independence(df, target="y")
        
        # Should work without error, ignoring 'category' column
        assert isinstance(result, dict)
        assert 'dw_statistic' in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimum_data_points(self):
        """Test with minimum number of data points."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [2, 4, 6]
        })
        result = check_independence(df, target="y")
        
        # Should still work with 3 points
        assert isinstance(result, dict)
    
    def test_perfect_linear_relationship(self):
        """Test with perfect linear relationship (zero residuals)."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10]  # y = 2*x exactly
        })
        result = check_independence(df, target="y")
        
        # With perfect fit, residuals are near zero
        assert isinstance(result['dw_statistic'], float)
    
    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n)
        })
        result = check_independence(df, target="y")
        
        assert 0 <= result['dw_statistic'] <= 4
    
    def test_single_feature(self):
        """Test with single feature column."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.1, 4.2, 5.9, 8.1, 10.2, 11.8, 14.1, 16.0, 17.9, 20.1]
        })
        result = check_independence(df, target="y")
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['dw_statistic', 'is_independent', 'message'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])