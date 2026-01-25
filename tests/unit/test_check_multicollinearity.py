"""
Unit tests for check_multicollinearity_vif function.

This module tests the VIF (Variance Inflation Factor) calculation functionality,
including edge cases like missing values, constant columns, perfect collinearity,
and threshold customization.
"""

import numpy as np
import pandas as pd
import pytest

from lrassume import check_multicollinearity_vif

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def df_numeric_ok():
    """
    Create a DataFrame with low multicollinearity.

    Returns
    -------
    pd.DataFrame
        DataFrame with 3 numeric features showing minimal collinearity.
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "x3": [5, 3, 2, 4, 1],
        }
    )


@pytest.fixture
def df_with_target():
    """
    Create a DataFrame with features and a target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 2 features (x1, x2) and target (y).
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "y": [10, 20, 15, 25, 22],
        }
    )


@pytest.fixture
def df_with_nan():
    """
    Create a DataFrame with missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN in the x1 column.
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, np.nan, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "x3": [5, 3, 2, 4, 1],
        }
    )


@pytest.fixture
def df_with_non_numeric():
    """
    Create a DataFrame with mixed numeric and categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 2 numeric and 1 categorical column.
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "cat": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def df_with_constant():
    """
    Create a DataFrame with a constant column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 2 varying features and 1 constant column.
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "const": [7, 7, 7, 7, 7],
        }
    )


@pytest.fixture
def df_perfect_collinearity():
    """
    Create a DataFrame with perfect linear collinearity.

    Returns
    -------
    pd.DataFrame
        DataFrame where x2 = 2*x1 (perfect collinearity).
    """
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # x2 = 2*x1
            "x3": [5, 3, 2, 4, 1],
        }
    )


@pytest.fixture
def df_moderate_collinearity():
    """
    Create a DataFrame with moderate multicollinearity.

    Returns
    -------
    pd.DataFrame
        DataFrame designed to produce VIF values between 5 and 10.
    """
    np.random.seed(42)
    x1 = np.arange(1, 21)
    x2 = 2 * x1 + np.random.normal(0, 5, 20)  # High correlation with x1
    x3 = np.random.normal(0, 10, 20)  # Independent

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_check_multicollinearity_vif_basic_runs(df_numeric_ok):
    """
    Test that check_multicollinearity_vif runs successfully with valid input.

    Verifies:
    - Returns DataFrame and dict
    - Correct column names in output
    - All metadata fields present
    - VIF values are >= 1.0
    """
    vif_table, summary = check_multicollinearity_vif(df_numeric_ok)

    assert isinstance(vif_table, pd.DataFrame)
    assert isinstance(summary, dict)

    assert set(vif_table.columns) == {"feature", "vif", "level"}
    assert summary["n_features"] == 3
    assert summary["warn_threshold"] == 5.0
    assert summary["severe_threshold"] == 10.0
    assert summary["dropped_non_numeric"] == []
    assert summary["dropped_constant"] == []

    # VIFs should all be >= 1 for non-degenerate cases
    assert (vif_table["vif"] >= 1.0).all()


# ============================================================================
# Target Column Handling Tests
# ============================================================================


def test_check_multicollinearity_vif_drops_target(df_with_target):
    """
    Test that the target column is correctly excluded from VIF calculation.

    Verifies:
    - Target column is not in VIF results
    - Only features are analyzed
    - Correct feature count
    """
    vif_table, summary = check_multicollinearity_vif(df_with_target, target_column="y")

    assert summary["n_features"] == 2
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert "y" not in vif_table["feature"].tolist()


def test_check_multicollinearity_vif_target_missing_raises(df_with_target):
    """
    Test that specifying a non-existent target column raises ValueError.
    """
    with pytest.raises(ValueError, match="target_column='z'.*not found"):
        check_multicollinearity_vif(df_with_target, target_column="z")


# ============================================================================
# Threshold Validation Tests
# ============================================================================


def test_check_multicollinearity_vif_warn_threshold_invalid(df_numeric_ok):
    """
    Test that invalid warn_threshold (≤ 0) raises ValueError.
    """
    with pytest.raises(ValueError, match="warn_threshold must be > 0"):
        check_multicollinearity_vif(df_numeric_ok, warn_threshold=0)


def test_check_multicollinearity_vif_severe_threshold_invalid(df_numeric_ok):
    """
    Test that severe_threshold less than warn_threshold raises ValueError.
    """
    with pytest.raises(ValueError, match="severe_threshold must be >="):
        check_multicollinearity_vif(df_numeric_ok, warn_threshold=5, severe_threshold=4)


def test_check_multicollinearity_vif_custom_thresholds(df_moderate_collinearity):
    """
    Test that custom thresholds correctly categorize VIF levels.

    Uses stricter thresholds (warn=3, severe=6) and verifies:
    - Thresholds are stored in summary
    - VIF levels are assigned correctly based on custom thresholds
    - Overall status reflects custom thresholds
    """
    vif_table, summary = check_multicollinearity_vif(
        df_moderate_collinearity, warn_threshold=3.0, severe_threshold=6.0
    )

    # Verify custom thresholds are stored
    assert summary["warn_threshold"] == 3.0
    assert summary["severe_threshold"] == 6.0

    # Check that levels are assigned according to custom thresholds
    for _, row in vif_table.iterrows():
        vif_val = row["vif"]
        level = row["level"]

        if vif_val >= 6.0:
            assert level == "severe"
        elif vif_val >= 3.0:
            assert level == "warn"
        else:
            assert level == "ok"

    # Verify overall status uses custom thresholds
    assert summary["overall_status"] in ["ok", "warn", "severe"]


# ============================================================================
# Non-Numeric Column Handling Tests
# ============================================================================


def test_check_multicollinearity_vif_non_numeric_error(df_with_non_numeric):
    """
    Test that non-numeric columns raise ValueError when categorical='error'.
    """
    with pytest.raises(ValueError, match="Non-numeric columns present"):
        check_multicollinearity_vif(df_with_non_numeric, categorical="error")


def test_check_multicollinearity_vif_non_numeric_drop(df_with_non_numeric):
    """
    Test that non-numeric columns are dropped when categorical='drop'.

    Verifies:
    - Categorical columns are listed in dropped_non_numeric
    - Only numeric columns remain
    - Correct feature count
    """
    vif_table, summary = check_multicollinearity_vif(
        df_with_non_numeric, categorical="drop"
    )

    assert summary["dropped_non_numeric"] == ["cat"]
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert summary["n_features"] == 2


# ============================================================================
# Missing Values Tests
# ============================================================================


def test_check_multicollinearity_vif_missing_values_raise(df_with_nan):
    """
    Test that DataFrames with missing values raise ValueError.
    """
    with pytest.raises(ValueError, match="Missing values found in predictors"):
        check_multicollinearity_vif(df_with_nan)


# ============================================================================
# Constant Column Tests
# ============================================================================


def test_check_multicollinearity_vif_constant_dropped(df_with_constant):
    """
    Test that constant columns are dropped when drop_constant=True.

    Verifies:
    - Constant columns are listed in dropped_constant
    - Only varying columns remain
    - Correct feature count
    """
    vif_table, summary = check_multicollinearity_vif(
        df_with_constant, drop_constant=True
    )

    assert summary["dropped_constant"] == ["const"]
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert summary["n_features"] == 2


def test_check_multicollinearity_vif_constant_raises_when_not_dropping(
    df_with_constant,
):
    """
    Test that constant columns raise ValueError when drop_constant=False.
    """
    with pytest.raises(
        ValueError, match="Constant columns present but drop_constant=False"
    ):
        check_multicollinearity_vif(df_with_constant, drop_constant=False)


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_check_multicollinearity_vif_too_few_features_after_drops_raises(
    df_with_non_numeric,
):
    """
    Test that having fewer than 2 features after drops raises ValueError.

    Drops 'x2' and 'cat', leaving only 'x1', which is insufficient for VIF.
    """
    df_one_numeric = df_with_non_numeric.drop(columns=["x2"])
    with pytest.raises(ValueError, match="Fewer than 2 features remain"):
        check_multicollinearity_vif(df_one_numeric, categorical="drop")


def test_check_multicollinearity_vif_perfect_collinearity_infinite(
    df_perfect_collinearity,
):
    """
    Test that perfect collinearity produces infinite VIF values.

    Verifies:
    - Overall status is 'severe'
    - At least one VIF is infinite
    - Infinite VIFs are labeled as 'severe'
    """
    vif_table, summary = check_multicollinearity_vif(df_perfect_collinearity)

    # x1 and x2 should be infinite (or at least one of them), and severe overall
    assert summary["overall_status"] == "severe"
    assert np.isinf(vif_table["vif"]).any()
    assert (vif_table.loc[vif_table["vif"] == np.inf, "level"] == "severe").all()
