import numpy as np
import pandas as pd
import pytest

from lrassume.check_homoscedasticity import check_homoscedasticity


@pytest.fixture
def homoscedastic_data():
    np.random.seed(0)
    X = pd.DataFrame({
        "x1": np.linspace(1, 100, 100),
        "x2": np.random.randn(100),
    })
    y = 2 * X["x1"] + 3 * X["x2"] + np.random.randn(100)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    np.random.seed(0)
    X = pd.DataFrame({"x1": np.linspace(1, 100, 100)})
    errors = np.random.randn(100) * X["x1"]
    y = 2 * X["x1"] + errors
    return X, y


# -----------------------------
# Valid behavior
# -----------------------------
def test_breusch_pagan_homoscedastic(homoscedastic_data):
    X, y = homoscedastic_data
    results, summary = check_homoscedasticity(X, y)

    assert results.iloc[0]["test"] == "breusch_pagan"
    assert summary["overall_conclusion"] == "homoscedastic"


def test_detects_heteroscedasticity(heteroscedastic_data):
    X, y = heteroscedastic_data
    _, summary = check_homoscedasticity(X, y)

    assert summary["overall_conclusion"] == "heteroscedastic"
    assert summary["n_tests_significant"] >= 1


def test_all_methods(homoscedastic_data):
    X, y = homoscedastic_data
    results, summary = check_homoscedasticity(X, y, method="all")

    assert summary["n_tests_performed"] == 3
    assert set(results["test"]) == {
        "breusch_pagan",
        "white",
        "goldfeld_quandt",
    }


def test_alpha_pass_through(homoscedastic_data):
    X, y = homoscedastic_data
    _, summary = check_homoscedasticity(X, y, alpha=0.01)
    assert summary["alpha"] == 0.01


# -----------------------------
# Error handling
# -----------------------------
def test_X_not_dataframe():
    with pytest.raises(TypeError):
        check_homoscedasticity([1, 2, 3], pd.Series([1, 2, 3]))


def test_y_not_series():
    with pytest.raises(TypeError):
        check_homoscedasticity(
            pd.DataFrame({"x": [1, 2, 3]}),
            [1, 2, 3],
        )


def test_length_mismatch():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)


def test_non_numeric_X():
    X = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)


def test_invalid_alpha(homoscedastic_data):
    X, y = homoscedastic_data
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y, alpha=1.5)


def test_residuals_without_fitted_values():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y, residuals=np.ones(10))


def test_too_few_observations():
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series(range(5))
    with pytest.raises(ValueError):
        check_homoscedasticity(X, y)
