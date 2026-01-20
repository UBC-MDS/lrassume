"""
check_independence.py
Module for testing independence of residuals in linear regression models.

This module provides tools for detecting autocorrelation in residuals from a
fitted linear regression model using the Durbin-Watson statistic. Independence
of residuals is a key assumption for valid inference in linear modeling.

Functions
---------
- check_independence(df, target)
    Fits a linear regression model and tests for independence of residuals
    using the Durbin-Watson statistic to detect autocorrelation.
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.linalg import lstsq


def check_independence(df: pd.DataFrame, target: str) -> dict:
    """
    Checks the independence of residuals using the Durbin-Watson statistic.

    This function fits a linear regression model on the provided dataframe,
    calculates the residuals, and then computes the Durbin-Watson score to
    determine if autocorrelation is present in the residuals. Independence
    is a key assumption for valid inference in linear modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with feature columns and the target column.
        Only numeric features will be used as predictors.
    target : str
        Name of the target column. The column must be numeric and the name
        must match the column name in the dataframe.

    Returns
    -------
    dict
        A dictionary containing:
        - 'dw_statistic' (float): The calculated Durbin-Watson value (0 to 4).
        - 'is_independent' (bool): True if the statistic is near 2 (typically 1.5 to 2.5),
          suggesting no significant autocorrelation.
        - 'message' (str): A brief interpretation of the result.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3, 4, 5],
    ...     "x2": [2, 4, 5, 7, 8],
    ...     "y": [10, 20, 25, 35, 40]
    ... })
    >>> check_independence(df, target="y")
    {'dw_statistic': np.float64(0.0727), 'is_independent': False, 'message': 'Positive autocorrelation detected. Residuals may not be independent.'}
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(target, str):
        raise TypeError("Input 'target' must be a string.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if not is_numeric_dtype(df[target]):
        raise TypeError("Target column must be numeric.")

    # Prepare data
    y = df[target].values
    X_cols = df.select_dtypes(include="number").columns.drop(target, errors="ignore")

    if len(X_cols) == 0:
        raise ValueError(
            "No numeric feature columns found in DataFrame (excluding target)."
        )

    X = df[X_cols].values

    # Check for missing values
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError(
            "DataFrame contains missing values. Please handle missing values before using this function."
        )

    # Add intercept term (column of ones)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Fit linear regression using scipy's least squares
    coeffs, _, _, _ = lstsq(X_with_intercept, y)

    # Calculate predicted values and residuals
    y_pred = X_with_intercept @ coeffs
    residuals = y - y_pred

    # Calculate Durbin-Watson statistic
    # DW = Σ(e_i - e_{i-1})² / Σ(e_i)²
    diff_residuals = np.diff(residuals)
    dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2)

    # Determine independence (typically 1.5 to 2.5 indicates independence)
    is_independent = bool(1.5 <= dw_statistic <= 2.5)

    # Generate message
    if is_independent:
        message = "No autocorrelation detected. Residuals appear independent."
    elif dw_statistic < 1.5:
        message = "Positive autocorrelation detected. Residuals may not be independent."
    else:  # dw_statistic > 2.5
        message = "Negative autocorrelation detected. Residuals may not be independent."

    return {
        "dw_statistic": round(dw_statistic, 4),
        "is_independent": is_independent,
        "message": message,
    }
