import numpy as np
import pandas as pd
import pytest

from lrassume import check_multicollinearity_vif


@pytest.fixture
def df_numeric_ok():
    # Low-ish collinearity, 3 features
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "x3": [5, 3, 2, 4, 1],
        }
    )


@pytest.fixture
def df_with_target():
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "y": [10, 20, 15, 25, 22],
        }
    )


@pytest.fixture
def df_with_nan():
    return pd.DataFrame(
        {
            "x1": [1, 2, np.nan, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "x3": [5, 3, 2, 4, 1],
        }
    )


@pytest.fixture
def df_with_non_numeric():
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "cat": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def df_with_constant():
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 1, 4, 3, 5],
            "const": [7, 7, 7, 7, 7],
        }
    )


@pytest.fixture
def df_perfect_collinearity():
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # x2 = 2*x1
            "x3": [5, 3, 2, 4, 1],
        }
    )


## Tests


def test_check_multicollinearity_vif_basic_runs(df_numeric_ok):
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


def test_check_multicollinearity_vif_drops_target(df_with_target):
    vif_table, summary = check_multicollinearity_vif(df_with_target, target_column="y")

    assert summary["n_features"] == 2
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert "y" not in vif_table["feature"].tolist()


def test_check_multicollinearity_vif_target_missing_raises(df_with_target):
    with pytest.raises(ValueError, match="target_column='z'.*not found"):
        check_multicollinearity_vif(df_with_target, target_column="z")


def test_check_multicollinearity_vif_warn_threshold_invalid(df_numeric_ok):
    with pytest.raises(ValueError, match="warn_threshold must be > 0"):
        check_multicollinearity_vif(df_numeric_ok, warn_threshold=0)


def test_check_multicollinearity_vif_severe_threshold_invalid(df_numeric_ok):
    with pytest.raises(ValueError, match="severe_threshold must be >="):
        check_multicollinearity_vif(df_numeric_ok, warn_threshold=5, severe_threshold=4)


def test_check_multicollinearity_vif_non_numeric_error(df_with_non_numeric):
    with pytest.raises(ValueError, match="Non-numeric columns present"):
        check_multicollinearity_vif(df_with_non_numeric, categorical="error")


def test_check_multicollinearity_vif_non_numeric_drop(df_with_non_numeric):
    vif_table, summary = check_multicollinearity_vif(
        df_with_non_numeric, categorical="drop"
    )

    assert summary["dropped_non_numeric"] == ["cat"]
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert summary["n_features"] == 2


def test_check_multicollinearity_vif_missing_values_raise(df_with_nan):
    with pytest.raises(ValueError, match="Missing values found in predictors"):
        check_multicollinearity_vif(df_with_nan)


def test_check_multicollinearity_vif_constant_dropped(df_with_constant):
    vif_table, summary = check_multicollinearity_vif(
        df_with_constant, drop_constant=True
    )

    assert summary["dropped_constant"] == ["const"]
    assert set(vif_table["feature"].tolist()) == {"x1", "x2"}
    assert summary["n_features"] == 2


def test_check_multicollinearity_vif_constant_raises_when_not_dropping(
    df_with_constant,
):
    with pytest.raises(
        ValueError, match="Constant columns present but drop_constant=False"
    ):
        check_multicollinearity_vif(df_with_constant, drop_constant=False)


def test_check_multicollinearity_vif_too_few_features_after_drops_raises(
    df_with_non_numeric,
):
    # Drops 'cat', leaving only one numeric column => should raise
    df_one_numeric = df_with_non_numeric.drop(columns=["x2"])
    with pytest.raises(ValueError, match="Fewer than 2 features remain"):
        check_multicollinearity_vif(df_one_numeric, categorical="drop")


def test_check_multicollinearity_vif_perfect_collinearity_infinite(
    df_perfect_collinearity,
):
    vif_table, summary = check_multicollinearity_vif(df_perfect_collinearity)

    # x1 and x2 should be infinite (or at least one of them), and severe overall
    assert summary["overall_status"] == "severe"
    assert np.isinf(vif_table["vif"]).any()
    assert (vif_table.loc[vif_table["vif"] == np.inf, "level"] == "severe").all()


@pytest.fixture
def df_two_features_correlated():
    # Simple 2-feature case with known correlation
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 3, 5, 7, 11],
        }
    )


def test_check_multicollinearity_vif_two_feature_matches_correlation(
    df_two_features_correlated,
):
    vif_table, summary = check_multicollinearity_vif(df_two_features_correlated)

    # For two predictors: VIF = 1 / (1 - r^2)
    r = np.corrcoef(
        df_two_features_correlated["x1"].to_numpy(),
        df_two_features_correlated["x2"].to_numpy(),
    )[0, 1]
    expected_vif = 1.0 / (1.0 - r**2)

    vifs = dict(zip(vif_table["feature"], vif_table["vif"]))

    assert pytest.approx(vifs["x1"], rel=1e-10) == expected_vif
    assert pytest.approx(vifs["x2"], rel=1e-10) == expected_vif
    assert summary["n_features"] == 2
