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
        - If X contains missing values (NaN/None) in evaluated predictors

    Notes
    -----
    - VIF measures linear dependence among predictors only, not their relationship
      with the target variable.
    - VIF = inf indicates perfect multicollinearity (one predictor is a perfect
      linear combination of others).
    - The auxiliary regressions used to compute R_j^2 include an intercept term.
    - Constant columns have no variance and will cause numerical issues if not dropped.
    """
    import numpy as np

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    if warn_threshold <= 0:
        raise ValueError("warn_threshold must be > 0.")
    if severe_threshold < warn_threshold:
        raise ValueError("severe_threshold must be >= warn_threshold.")

    df = X.copy()

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"target_column='{target_column}' was not found in X.")
        df = df.drop(columns=[target_column])

    dropped_non_numeric: list[str] = []
    dropped_constant: list[str] = []

    # Handle non-numeric columns
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        if categorical == "error":
            raise ValueError(
                "Non-numeric columns present but categorical='error'. "
                f"Non-numeric columns: {non_numeric_cols}"
            )
        if categorical == "drop":
            dropped_non_numeric = non_numeric_cols
            df = df.drop(columns=non_numeric_cols)
        else:
            raise ValueError(f"Invalid categorical handling option: {categorical}")

    if df.shape[1] == 0:
        raise ValueError("No numeric predictor columns remain after preprocessing.")

    # Missing values check (keep it strict and predictable for unit tests)
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "Missing values found in predictors. "
            f"Columns with missing values: {bad_cols}"
        )

    # Drop constant columns if requested
    if drop_constant:
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            dropped_constant = constant_cols
            df = df.drop(columns=constant_cols)
    else:
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            raise ValueError(
                "Constant columns present but drop_constant=False. "
                f"Constant columns: {constant_cols}"
            )

    if df.shape[1] < 2:
        raise ValueError(
            "Fewer than 2 features remain after dropping columns; VIF requires at least 2 predictors."
        )

    # Convert to float matrix
    features = df.columns.to_list()
    Xmat = df.astype(float).to_numpy()
    n, p = Xmat.shape

    # Compute VIF per feature via auxiliary regression with intercept
    vifs: list[float] = []
    eps = 1e-12

    for j in range(p):
        y = Xmat[:, j]
        others = np.delete(Xmat, j, axis=1)

        # Add intercept column
        A = np.column_stack([np.ones(n), others])

        # Least squares fit y ~ intercept + others
        beta, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)
        y_hat = A @ beta

        sse = float(np.sum((y - y_hat) ** 2))
        y_mean = float(np.mean(y))
        sst = float(np.sum((y - y_mean) ** 2))

        # If SST is ~0, the feature is (near) constant and should have been dropped
        if sst <= eps:
            raise ValueError(
                f"Feature '{features[j]}' has (near) zero variance after preprocessing; "
                "set drop_constant=True or remove constant predictors."
            )

        r2 = 1.0 - (sse / sst)

        # Guard against tiny numerical overshoots
        if r2 >= 1.0 - 1e-12:
            vif = float("inf")
        else:
            denom = max(1.0 - r2, eps)
            vif = float(1.0 / denom)

        vifs.append(vif)

    def _level(v: float) -> str:
        if np.isinf(v) or v > severe_threshold:
            return "severe"
        if v > warn_threshold:
            return "warn"
        return "ok"

    levels = [_level(v) for v in vifs]

    vif_table = pd.DataFrame(
        {"feature": features, "vif": vifs, "level": levels}
    ).sort_values(by="vif", ascending=False, kind="mergesort").reset_index(drop=True)

    n_warn = int((vif_table["level"] == "warn").sum())
    n_severe = int((vif_table["level"] == "severe").sum())
    overall_status = "severe" if n_severe > 0 else ("warn" if n_warn > 0 else "ok")

    summary: Dict[str, Any] = {
        "overall_status": overall_status,
        "n_features": int(vif_table.shape[0]),
        "n_warn": n_warn,
        "n_severe": n_severe,
        "warn_threshold": float(warn_threshold),
        "severe_threshold": float(severe_threshold),
        "dropped_non_numeric": dropped_non_numeric,
        "dropped_constant": dropped_constant,
    }

    return vif_table, summary
