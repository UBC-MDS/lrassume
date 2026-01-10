"""
Multicollinearity diagnostics (VIF).

This module contains utilities to detect multicollinearity among predictors
for linear regression workflows.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import pandas as pd


CategoricalHandling = Literal["error", "drop"]


def check_multicollinearity_vif(
    X: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    warn_threshold: float = 5.0,
    severe_threshold: float = 10.0,
    categorical: CategoricalHandling = "error",
    drop_constant: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute multicollinearity diagnostics using Variance Inflation Factor (VIF).

    Multicollinearity refers to strong linear dependence among predictor variables.
    It does NOT involve the target variable. VIF is defined for each predictor x_j as:

        VIF_j = 1 / (1 - R_j^2)

    where R_j^2 is the coefficient of determination from regressing x_j on all
    other predictors. High VIF indicates inflated variance of coefficient estimates
    in ordinary least squares (OLS), leading to unstable coefficients.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of predictors (features). Each column is treated as a predictor.
        If the target column is included, specify it via `target_column`.

    target_column : str, optional
        Name of the target column to exclude from VIF calculation.
        If None, assumes X contains only predictors.
        Raises ValueError if specified but not found in X.

    warn_threshold : float, default=5.0
        VIF threshold for flagging features as "warn".
        Common heuristic: VIF > 5 suggests moderate multicollinearity.

    severe_threshold : float, default=10.0
        VIF threshold for flagging features as "severe".
        Common heuristic: VIF > 10 suggests severe multicollinearity.
        Must be >= warn_threshold.

    categorical : {"error", "drop"}, default="error"
        How to handle non-numeric columns:
        
        - "error": Raise ValueError if non-numeric columns are present
        - "drop": Remove non-numeric columns and report in summary

    drop_constant : bool, default=True
        Whether to drop constant columns (where all values are identical).
        
        - If True: Constant columns are removed and reported in summary
        - If False: May raise ValueError during VIF computation due to singularity

    Returns
    -------
    vif_table : pd.DataFrame
        One row per feature with columns:
        
        - "feature" (str): Feature name
        - "vif" (float): VIF value (may be inf for perfect collinearity)
        - "level" (str): One of {"ok", "warn", "severe"}
        
        Rows are sorted by VIF in descending order.

    summary : dict
        Overall diagnostics containing:
        
        - "overall_status" (str): Worst level found ("ok", "warn", or "severe")
        - "n_features" (int): Number of features evaluated
        - "n_warn" (int): Count of features with warn-level VIF
        - "n_severe" (int): Count of features with severe-level VIF
        - "warn_threshold" (float): Echo of input threshold
        - "severe_threshold" (float): Echo of input threshold
        - "dropped_non_numeric" (list[str]): Non-numeric columns dropped (if categorical="drop")
        - "dropped_constant" (list[str]): Constant columns dropped (if drop_constant=True)

    Raises
    ------
    ValueError
        - If target_column is specified but not found in X
        - If categorical="error" and non-numeric columns exist
        - If drop_constant=False and constant columns prevent VIF computation
        - If warn_threshold <= 0 or severe_threshold < warn_threshold
        - If fewer than 2 features remain after dropping columns
        - If predictors are perfectly collinear (rank-deficient design matrix)

    Notes
    -----
    - VIF measures linear dependence among predictors only, not their relationship
      with the target variable.
    - VIF = inf indicates perfect multicollinearity (one predictor is a perfect
      linear combination of others).
    - The auxiliary regressions used to compute R_j^2 include an intercept term.
    - Constant columns have no variance and will cause numerical issues if not dropped.

    Examples
    --------
    Basic usage with only predictors:
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [2, 4, 5, 7, 8],
    ...     'x3': [1, 3, 2, 5, 4]
    ... })
    >>> vif_table, summary = check_multicollinearity_vif(df)
    >>> print(summary["overall_status"])
    'ok'

    With target column:
    
    >>> df['y'] = [10, 20, 15, 25, 22]
    >>> vif_table, summary = check_multicollinearity_vif(df, target_column='y')
    >>> print(vif_table)
       feature       vif level
    0       x1  1.234567    ok
    1       x2  1.456789    ok
    2       x3  1.123456    ok

    Detecting severe multicollinearity:
    
    >>> df_collinear = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [2, 4, 6, 8, 10],  # x2 = 2 * x1 (perfect collinearity)
    ...     'x3': [1, 3, 2, 5, 4]
    ... })
    >>> vif_table, summary = check_multicollinearity_vif(df_collinear)
    >>> print(summary["overall_status"])
    'severe'
    >>> print(summary["n_severe"])
    2

    Handling non-numeric columns:
    
    >>> df_mixed = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [2, 4, 5, 7, 8],
    ...     'category': ['A', 'B', 'A', 'B', 'A']
    ... })
    >>> vif_table, summary = check_multicollinearity_vif(
    ...     df_mixed, 
    ...     categorical='drop'
    ... )
    >>> print(summary["dropped_non_numeric"])
    ['category']
    """
    raise NotImplementedError(
        "Implementation will be added in a later milestone."
    )