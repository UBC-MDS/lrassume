"""
Homoscedasticity diagnostics for linear regression.

This module contains utilities to detect heteroscedasticity (non-constant variance)
in residuals for linear regression workflows. Heteroscedasticity violates a key
assumption of ordinary least squares (OLS) regression and can lead to inefficient
estimates and incorrect standard errors.

The module provides the `check_homoscedasticity` function which implements three
widely-used statistical tests: Breusch-Pagan, White, and Goldfeld-Quandt.

Functions
---------
check_homoscedasticity : Test residuals for constant variance

Examples
--------
Basic usage:
    >>> import pandas as pd
    >>> import numpy as np
    >>> from lrassume import check_homoscedasticity
    >>> 
    >>> X = pd.DataFrame({'x1': range(100), 'x2': np.random.randn(100)})
    >>> y = pd.Series(2 * X['x1'] + np.random.randn(100))
    >>> results, summary = check_homoscedasticity(X, y)
    >>> print(summary['overall_conclusion'])
    'homoscedastic'

Notes
-----
All tests assume that residuals come from a linear regression model. If using
non-linear models, interpret results with caution.

References
----------
.. [1] Breusch, T. S., & Pagan, A. R. (1979). A simple test for 
       heteroscedasticity and random coefficient variation. 
       Econometrica, 47(5), 1287-1294.

.. [2] White, H. (1980). A heteroskedasticity-consistent covariance 
       matrix estimator and a direct test for heteroskedasticity. 
       Econometrica, 48(4), 817-838.

.. [3] Goldfeld, S. M., & Quandt, R. E. (1965). Some tests for 
       homoscedasticity. Journal of the American Statistical Association,
       60(310), 539-547.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt,
)

# Type alias for supported test methods
TestMethod = Literal["breusch_pagan", "white", "goldfeld_quandt", "all"]


def check_homoscedasticity(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: TestMethod = "breusch_pagan",
    alpha: float = 0.05,
    fitted_model: Optional[Any] = None,
    residuals: Optional[np.ndarray] = None,
    fitted_values: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Test for homoscedasticity (constant variance) in linear regression residuals.

    Homoscedasticity is the assumption that residuals have constant variance across
    all levels of the independent variables. Violation of this assumption 
    (heteroscedasticity) leads to inefficient coefficient estimates and incorrect
    standard errors in ordinary least squares (OLS) regression.

    This function implements multiple statistical tests to detect heteroscedasticity:
    
    - **Breusch-Pagan test**: Tests whether residual variance depends linearly 
      on predictors. Null hypothesis: homoscedasticity (constant variance).
    
    - **White test**: More general test that allows for non-linear relationships
      between variance and predictors. Includes squared terms and interactions.
      Null hypothesis: homoscedasticity.
    
    - **Goldfeld-Quandt test**: Splits data by a predictor and compares variance
      in two subsets. Useful for detecting variance that increases/decreases
      with a specific predictor.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of predictors (features). Each column is a predictor variable.
        Must contain only numeric columns.

    y : pd.Series
        Target variable (response). Must have the same length as X.

    method : {"breusch_pagan", "white", "goldfeld_quandt", "all"}, default="breusch_pagan"
        Statistical test(s) to perform:
        
        - "breusch_pagan": Breusch-Pagan Lagrange multiplier test
        - "white": White's general heteroscedasticity test
        - "goldfeld_quandt": Goldfeld-Quandt test (splits on first predictor by default)
        - "all": Run all available tests

    alpha : float, default=0.05
        Significance level for hypothesis tests. Common values: 0.01, 0.05, 0.10.
        Must be between 0 and 1 (exclusive).

    fitted_model : optional
        Pre-fitted regression model object with `predict()` method.
        If None, an OLS model will be fitted internally using X and y.
        Useful for avoiding refitting when model already exists.

    residuals : np.ndarray, optional
        Pre-computed residuals (y - y_pred). Must have same length as y.
        If None, residuals will be computed from fitted_model or internal fit.
        Cannot be specified without fitted_values.

    fitted_values : np.ndarray, optional
        Pre-computed fitted values (y_pred). Must have same length as y.
        If None, fitted values will be computed from fitted_model or internal fit.
        Cannot be specified without residuals.

    Returns
    -------
    test_results : pd.DataFrame
        One row per test performed, with columns:
        
        - "test" (str): Name of the test performed
        - "statistic" (float): Test statistic value, rounded to 3 decimals
        - "p_value" (float): P-value for the test, rounded to 4 decimals
        - "conclusion" (str): One of {"homoscedastic", "heteroscedastic"}
        - "significant" (bool): True if p_value < alpha (reject null hypothesis)
        
        Rows are sorted by test name alphabetically.

    summary : dict
        Overall diagnostics containing:
        
        - "overall_conclusion" (str): "homoscedastic" if all tests pass, 
          otherwise "heteroscedastic"
        - "n_tests_performed" (int): Number of tests conducted
        - "n_tests_significant" (int): Number of tests rejecting homoscedasticity
        - "alpha" (float): Echo of significance level used
        - "n_observations" (int): Sample size
        - "n_predictors" (int): Number of predictor variables
        - "recommendation" (str): Suggested action if heteroscedasticity detected

    Raises
    ------
    ValueError
        - If X contains non-numeric columns
        - If X and y have different lengths
        - If alpha is not between 0 and 1
        - If residuals is provided without fitted_values or vice versa
        - If residuals/fitted_values length doesn't match y
        - If fewer than 10 observations are available (insufficient for testing)

    TypeError
        - If fitted_model is provided but lacks predict() method
        - If X is not a pandas DataFrame
        - If y is not a pandas Series

    Notes
    -----
    - All tests assume residuals from a linear regression model.
    - Tests use chi-square or F-distributions depending on the method.
    - The Breusch-Pagan test is most powerful against linear heteroscedasticity.
    - The White test is more general but may have lower power with small samples.
    - Goldfeld-Quandt test requires ordering data, which may be arbitrary for
      multivariate predictors.
    - If heteroscedasticity is detected, consider using robust standard errors
      (e.g., HC3, HC4) or weighted least squares (WLS) regression.
    - Missing values in X or y will raise an error; clean data beforehand.

    Examples
    --------
    Basic usage with internal model fitting:
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame({
    ...     'x1': np.linspace(1, 100, 100),
    ...     'x2': np.random.randn(100)
    ... })
    >>> y = pd.Series(2 * X['x1'] + 3 * X['x2'] + np.random.randn(100))
    >>> test_results, summary = check_homoscedasticity(X, y)
    >>> print(summary["overall_conclusion"])
    'homoscedastic'

    Using a pre-fitted model:
    
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression().fit(X, y)
    >>> test_results, summary = check_homoscedasticity(
    ...     X, y, fitted_model=model
    ... )
    >>> print(test_results)
              test  statistic   p_value      conclusion  significant
    0  breusch_pagan      2.345      0.309  homoscedastic        False

    Running all tests:
    
    >>> test_results, summary = check_homoscedasticity(
    ...     X, y, method="all", alpha=0.01
    ... )
    >>> print(summary["n_tests_performed"])
    3
    >>> print(summary["n_tests_significant"])
    0

    Detecting heteroscedasticity (variance increases with x):
    
    >>> X_hetero = pd.DataFrame({
    ...     'x1': np.linspace(1, 100, 100)
    ... })
    >>> errors = np.random.randn(100) * X_hetero['x1']  # variance increases
    >>> y_hetero = pd.Series(2 * X_hetero['x1'] + errors)
    >>> test_results, summary = check_homoscedasticity(X_hetero, y_hetero)
    >>> print(summary["overall_conclusion"])
    'heteroscedastic'
    >>> print(summary["recommendation"])
    'Consider using robust standard errors (HC3/HC4) or weighted least squares.'

    Using pre-computed residuals and fitted values:
    
    >>> model = LinearRegression().fit(X, y)
    >>> y_pred = model.predict(X)
    >>> resid = y - y_pred
    >>> test_results, summary = check_homoscedasticity(
    ...     X, y, 
    ...     residuals=resid, 
    ...     fitted_values=y_pred
    ... )
    >>> print(test_results)
              test  statistic   p_value      conclusion  significant
    0  breusch_pagan      2.345      0.309  homoscedastic        False

    References
    ----------
    .. [1] Breusch, T. S., & Pagan, A. R. (1979). A simple test for 
           heteroscedasticity and random coefficient variation. 
           Econometrica, 47(5), 1287-1294.
    
    .. [2] White, H. (1980). A heteroskedasticity-consistent covariance 
           matrix estimator and a direct test for heteroskedasticity. 
           Econometrica, 48(4), 817-838.
    
    .. [3] Goldfeld, S. M., & Quandt, R. E. (1965). Some tests for 
           homoscedasticity. Journal of the American Statistical Association,
           60(310), 539-547.
    """
    
    # --------------------
    # Input Validation
    # --------------------
    
    # Validate data types
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    
    # Validate dimensions
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if len(X) < 10:
        raise ValueError("At least 10 observations are required.")
    
    # Validate alpha parameter
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    
    # Validate numeric columns
    if not all(is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("All columns in X must be numeric.")

    # Validate residuals and fitted_values are provided together
    if (residuals is None) ^ (fitted_values is None):
        raise ValueError("residuals and fitted_values must be provided together.")

    # --------------------
    # Compute or Validate Residuals
    # --------------------
    
    if residuals is not None:
        # User provided residuals - validate length
        if len(residuals) != len(y):
            raise ValueError("residuals must match length of y.")
    else:
        # Need to compute residuals
        if fitted_model is not None:
            # Use provided fitted model
            if not hasattr(fitted_model, "predict"):
                raise TypeError("fitted_model must implement predict().")
            fitted_values = fitted_model.predict(X)
            residuals = y.values - fitted_values
        else:
            # Fit an OLS model internally
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
            residuals = model.resid
            fitted_values = model.fittedvalues

    # --------------------
    # Determine Tests to Run
    # --------------------
    
    tests_to_run = []
    if method in ("breusch_pagan", "all"):
        tests_to_run.append("breusch_pagan")
    if method in ("white", "all"):
        tests_to_run.append("white")
    if method in ("goldfeld_quandt", "all"):
        tests_to_run.append("goldfeld_quandt")

    # Add constant column for statsmodels tests
    X_const = sm.add_constant(X)
    rows = []

    # --------------------
    # Execute Statistical Tests
    # --------------------
    
    for test in tests_to_run:
        if test == "breusch_pagan":
            # Breusch-Pagan Lagrange multiplier test
            # Tests if residual variance is a linear function of predictors
            stat, pval, _, _ = het_breuschpagan(residuals, X_const)
            
        elif test == "white":
            # White's general heteroscedasticity test
            # Tests against non-linear forms of heteroscedasticity
            # Includes squared terms and cross-products
            stat, pval, _, _ = het_white(residuals, X_const)
            
        else:  # goldfeld_quandt
            # Goldfeld-Quandt test
            # Splits sample into two groups and compares variances
            # Uses first predictor column for ordering
            x_gq = X.iloc[:, [0]]  # Keep as DataFrame with shape (n, 1)
            stat, pval, _ = het_goldfeldquandt(residuals, x_gq)

        # Determine if result is statistically significant
        significant = pval < alpha
        
        # Store test results
        rows.append({
            "test": test,
            "statistic": round(float(stat), 3),
            "p_value": round(float(pval), 4),
            "conclusion": "heteroscedastic" if significant else "homoscedastic",
            "significant": significant,
        })

    # --------------------
    # Format Results
    # --------------------
    
    # Create DataFrame of test results, sorted alphabetically by test name
    results = (
        pd.DataFrame(rows)
        .sort_values("test")
        .reset_index(drop=True)
    )

    # Count number of significant tests
    n_sig = int(results["significant"].sum())

    # --------------------
    # Generate Summary
    # --------------------
    
    summary = {
        "overall_conclusion": "heteroscedastic" if n_sig > 0 else "homoscedastic",
        "n_tests_performed": len(results),
        "n_tests_significant": n_sig,
        "alpha": alpha,
        "n_observations": len(y),
        "n_predictors": X.shape[1],
        "recommendation": (
            "Consider using robust standard errors (HC3/HC4) or weighted least squares."
            if n_sig > 0
            else "No action needed."
        ),
    }

    return results, summary