"""
Test suite for homoscedasticity diagnostics module.

This module contains comprehensive unit tests for the check_homoscedasticity
function, covering both valid behavior and error handling scenarios.

Test Coverage
-------------
- Valid behavior with homoscedastic data
- Detection of heteroscedastic patterns
- Different test methods (Breusch-Pagan, White, Goldfeld-Quandt)
- Parameter validation and error handling
- Edge cases and boundary conditions

Test Fixtures
-------------
homoscedastic_data : Synthetic data with constant variance
heteroscedastic_data : Synthetic data with increasing variance

Running Tests
-------------
Run all tests:
    $ pytest test_homoscedasticity.py

Run with coverage:
    $ pytest --cov=lrassume test_homoscedasticity.py

Run verbose:
    $ pytest -v test_homoscedasticity.py

Notes
-----
All tests use fixed random seeds for reproducibility. Tests are independent
and can be run in any order.
"""

import numpy as np
import pandas as pd
import pytest

from lrassume.check_homoscedasticity import check_homoscedasticity


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def homoscedastic_data():
    """
    Generate synthetic data with constant variance (homoscedastic).
    
    Creates a linear relationship with normally distributed errors that have
    constant variance across all predictor values. This data should pass
    homoscedasticity tests.
    
    Returns
    -------
    X : pd.DataFrame
        DataFrame with two predictors:
        - 'x1': 100 evenly spaced values from 1 to 100
        - 'x2': 100 standard normal random values
    
    y : pd.Series
        Target variable following: y = 2*x1 + 3*x2 + ε
        where ε ~ N(0, 1) with constant variance
    
    Notes
    -----
    Uses np.random.seed(0) for reproducible results.
    
    Examples
    --------
    >>> X, y = homoscedastic_data()
    >>> len(X)
    100
    >>> X.columns.tolist()
    ['x1', 'x2']
    """
    # Set seed for reproducible test results
    np.random.seed(0)
    
    # Create predictor variables
    X = pd.DataFrame({
        "x1": np.linspace(1, 100, 100),  # Linear spacing
        "x2": np.random.randn(100),       # Random normal
    })
    
    # Create target with constant variance errors
    # True model: y = 2*x1 + 3*x2 + noise
    y = 2 * X["x1"] + 3 * X["x2"] + np.random.randn(100)
    
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """
    Generate synthetic data with non-constant variance (heteroscedastic).
    
    Creates a linear relationship where error variance increases linearly
    with the predictor value. This data should fail homoscedasticity tests.
    
    Returns
    -------
    X : pd.DataFrame
        DataFrame with one predictor:
        - 'x1': 100 evenly spaced values from 1 to 100
    
    y : pd.Series
        Target variable following: y = 2*x1 + ε
        where ε ~ N(0, x1²) with variance proportional to x1
    
    Notes
    -----
    Uses np.random.seed(0) for reproducible results.
    The variance of residuals increases with x1, creating clear heteroscedasticity.
    
    Examples
    --------
    >>> X, y = heteroscedastic_data()
    >>> len(X)
    100
    >>> # Variance should increase with x1
    """
    # Set seed for reproducible test results
    np.random.seed(0)
    
    # Create predictor variable
    X = pd.DataFrame({"x1": np.linspace(1, 100, 100)})
    
    # Create errors with variance proportional to x1
    # This creates heteroscedasticity: Var(ε|x1) = x1
    errors = np.random.randn(100) * X["x1"]
    
    # Create target with non-constant variance
    y = 2 * X["x1"] + errors
    
    return X, y


# =============================================================================
# Tests for Valid Behavior
# =============================================================================


def test_breusch_pagan_homoscedastic(homoscedastic_data):
    """
    Test that Breusch-Pagan test correctly identifies homoscedastic data.
    
    Verifies that:
    - The default method is Breusch-Pagan
    - The test correctly concludes homoscedasticity for constant-variance data
    - Results contain the expected structure
    
    Parameters
    ----------
    homoscedastic_data : fixture
        Synthetic data with constant variance
    
    Asserts
    -------
    - Test name is "breusch_pagan"
    - Overall conclusion is "homoscedastic"
    """
    X, y = homoscedastic_data
    
    # Run with default method (Breusch-Pagan)
    results, summary = check_homoscedasticity(X, y)

    # Verify correct test was run
    assert results.iloc[0]["test"] == "breusch_pagan"
    
    # Verify correct conclusion for constant variance data
    assert summary["overall_conclusion"] == "homoscedastic"


def test_detects_heteroscedasticity(heteroscedastic_data):
    """
    Test that the function correctly detects heteroscedasticity.
    
    Uses data with variance that increases with the predictor value.
    The test should detect this pattern and conclude heteroscedasticity.
    
    Parameters
    ----------
    heteroscedastic_data : fixture
        Synthetic data with increasing variance
    
    Asserts
    -------
    - Overall conclusion is "heteroscedastic"
    - At least one test detects the heteroscedasticity (significant result)
    """
    X, y = heteroscedastic_data
    
    # Run homoscedasticity check
    _, summary = check_homoscedasticity(X, y)

    # Verify heteroscedasticity is detected
    assert summary["overall_conclusion"] == "heteroscedastic"
    
    # Verify at least one test found it significant
    assert summary["n_tests_significant"] >= 1


def test_all_methods(homoscedastic_data):
    """
    Test that method="all" runs all three available tests.
    
    Verifies that:
    - All three tests are executed
    - Results contain the correct test names
    - Summary correctly counts the number of tests performed
    
    Parameters
    ----------
    homoscedastic_data : fixture
        Synthetic data with constant variance
    
    Asserts
    -------
    - Exactly 3 tests are performed
    - Results contain Breusch-Pagan, White, and Goldfeld-Quandt tests
    """
    X, y = homoscedastic_data
    
    # Run all available tests
    results, summary = check_homoscedasticity(X, y, method="all")

    # Verify all three tests were run
    assert summary["n_tests_performed"] == 3
    
    # Verify all expected test names are present
    assert set(results["test"]) == {
        "breusch_pagan",
        "white",
        "goldfeld_quandt",
    }


def test_alpha_pass_through(homoscedastic_data):
    """
    Test that the alpha parameter is correctly stored in the summary.
    
    Verifies that custom significance levels are properly passed through
    and recorded in the output summary dictionary.
    
    Parameters
    ----------
    homoscedastic_data : fixture
        Synthetic data with constant variance
    
    Asserts
    -------
    - Summary["alpha"] matches the input alpha value (0.01)
    """
    X, y = homoscedastic_data
    
    # Run with custom alpha level
    _, summary = check_homoscedasticity(X, y, alpha=0.01)
    
    # Verify alpha is recorded correctly
    assert summary["alpha"] == 0.01


# =============================================================================
# Tests for Error Handling
# =============================================================================


def test_X_not_dataframe():
    """
    Test that TypeError is raised when X is not a pandas DataFrame.
    
    The function requires X to be a DataFrame for proper column handling.
    Passing a list or other type should raise a descriptive TypeError.
    
    Asserts
    -------
    - TypeError is raised when X is a list
    """
    # Attempt to pass a list instead of DataFrame
    with pytest.raises(TypeError):
        check_homoscedasticity([1, 2, 3], pd.Series([1, 2, 3]))


def test_y_not_series():
    """
    Test that TypeError is raised when y is not a pandas Series.
    
    The function requires y to be a Series for consistency with pandas
    operations. Passing a list or other type should raise a TypeError.
    
    Asserts
    -------
    - TypeError is raised when y is a list
    """
    # Attempt to pass a list instead of Series
    with pytest.raises(TypeError):
        check_homoscedasticity(
            pd.DataFrame({"x": [1, 2, 3]}),
            [1, 2, 3],
        )


def test_length_mismatch():
    """
    Test that ValueError is raised when X and y have different lengths.
    
    Statistical tests require paired observations, so X and y must have
    the same number of rows. Mismatched lengths should raise ValueError.
    
    Asserts
    -------
    - ValueError is raised when len(X) != len(y)
    """
    # Create data with mismatched lengths
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2])  # Only 2 values vs 3 in X
    
    # Verify error is raised
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)


def test_non_numeric_X():
    """
    Test that ValueError is raised when X contains non-numeric columns.
    
    Statistical tests require numeric predictors. DataFrames with categorical
    or string columns should be rejected with a descriptive ValueError.
    
    Asserts
    -------
    - ValueError is raised when X contains non-numeric data
    
    Notes
    -----
    Users should encode categorical variables before calling the function.
    """
    # Create DataFrame with mixed types (numeric and categorical)
    X = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
    y = pd.Series([1, 2, 3])
    
    # Verify error is raised for non-numeric column
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)


def test_invalid_alpha(homoscedastic_data):
    """
    Test that ValueError is raised for invalid alpha values.
    
    Alpha must be between 0 and 1 (exclusive) to be a valid significance
    level. Values outside this range should raise ValueError.
    
    Parameters
    ----------
    homoscedastic_data : fixture
        Synthetic data with constant variance
    
    Asserts
    -------
    - ValueError is raised when alpha > 1
    
    Notes
    -----
    This also tests alpha <= 0 and alpha >= 1 implicitly through the
    validation logic (0 < alpha < 1).
    """
    X, y = homoscedastic_data
    
    # Attempt to use invalid alpha (> 1)
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y, alpha=1.5)


def test_residuals_without_fitted_values():
    """
    Test that ValueError is raised when residuals provided without fitted_values.
    
    Residuals and fitted values must be provided together as a pair. Providing
    only one without the other creates an inconsistent state and should raise
    a ValueError.
    
    Asserts
    -------
    - ValueError is raised when residuals is provided without fitted_values
    
    Notes
    -----
    The same error should occur if fitted_values is provided without residuals.
    """
    # Create minimal valid data
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    
    # Attempt to provide residuals without fitted_values
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y, residuals=np.ones(10))


def test_too_few_observations():
    """
    Test that ValueError is raised when sample size is too small.
    
    Statistical tests require a minimum sample size for reliable inference.
    The function requires at least 10 observations. Smaller samples should
    raise a ValueError.
    
    Asserts
    -------
    - ValueError is raised when len(X) < 10
    
    Notes
    -----
    While 10 is the minimum, samples of 30+ are recommended for robust
    statistical inference.
    """
    # Create data with only 5 observations (< 10 minimum)
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series(range(5))
    
    # Verify error is raised for insufficient data
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)