# lrassume

|        |        |
|--------|--------|
| Package | [![Latest PyPI Version](https://img.shields.io/pypi/v/lrassume.svg)](https://test.pypi.org/project/lrassume/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/lrassume.svg)](https://pypi.org/project/lrassume/)  |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) |


**lrassume** (Linear Regression Assumption Validator) is a Python package for validating the core assumptions of linear regression models. It provides statistical tests and diagnostic tools to assess independence, linearity, multicollinearity, and homoscedasticity in your regression workflows.

## Features

- **Independence Testing**: Durbin-Watson test to detect autocorrelation in residuals
- **Linearity Assessment**: Pearson correlation analysis to identify linear relationships with the target
- **Multicollinearity Detection**: Variance Inflation Factor (VIF) calculation with configurable thresholds
- **Homoscedasticity Testing**: Multiple statistical tests (Breusch-Pagan, White, Goldfeld-Quandt) to detect heteroscedasticity

---
## Installation

### User Setup

This option is recommended if you want to use `lrassume` in your own projects and do not need to modify the source code.

```bash
pip install -i https://test.pypi.org/simple/ lrassume
```

---

### Development Setup (Recommended)

This option is recommended if you want to develop, modify, or contribute to `lrassume`.

This project uses **Conda** to manage the Python environment and **pip** to install project dependencies.

---

#### 1. Clone the Repository and Navigate to the Project Directory

```bash
git clone https://github.com/UBC-MDS/lrassume.git
cd lrassume
```

---

#### 2. Create and Activate the Conda Environment

From the project root directory:

```bash
conda env create -f environment.yml
conda activate lrassume
```

> The `environment.yml` file installs Python only.
> All runtime dependencies are specified in `pyproject.toml`.

---

#### 3. Install the Package in Editable Mode

```bash
pip install -e .
```

---

### Alternative: Development Without Conda

If you prefer not to use Conda, you can install the package directly using pip:

```bash
git clone https://github.com/UBC-MDS/lrassume.git
cd lrassume
pip install -e .
```

---

## Running the Test Suite Locally (Developers)

The test suite is executed using pytest. In CI this is managed via Hatch,
but tests can also be run locally using pytest.

---

### Install pytest

```bash
conda install pytest
# or
pip install pytest
```

---

### Run the tests

```bash
pytest
```

---

## Continuous Integration (Automated Testing)

This project uses **GitHub Actions** to automatically run the test suite.

The tests are executed automatically on:
- Pull requests
- Pushes to the `main` branch
- A scheduled weekly run

The test suite is executed using **Hatch**, which runs the project’s
configured `pytest` test environment across multiple operating systems
and Python versions.

No manual action is required to trigger these tests.

The GitHub Actions workflow responsible for running the test suite is located at:

```
.github/workflows/test.yml
```

## Documentation

The full package documentation is built with **Quartodoc** and deployed automatically to **GitHub Pages**.

**Live documentation:**


```
INSERT LINK once DEPLOYED
```

---

### Build Documentation Locally (Developers)

To build the documentation locally:

1. Ensure you are in the development environment:

```bash
conda activate lrassume
```

2. Install documentation dependencies (if not already installed):

```bash
pip install -e ".[docs]"
```

3. Build the documentation:

```bash
quartodoc build
quarto render docs
```

4. Preview the documentation locally:

```bash
quarto preview docs
```

This will open the documentation site in your browser.

---

### Deploy Documentation (Automated)

Documentation deployment is **fully automated** using **GitHub Actions**.

On every push to the `main` branch:

1. GitHub Actions builds the documentation using Quarto and Quartodoc
2. The rendered site is deployed to **GitHub Pages**

No manual deployment steps are required.

The workflow file can be found at:

```
.github/workflows/docs.yml
```

## Quick Start

### Check Independence

This function fits a linear model and checks for autocorrelation in the residuals.
```python
import pandas as pd
from lrassume import check_independence

# Create sample data
df = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5],
    "x2": [2, 4, 5, 7, 8],
    "y": [10, 20, 25, 35, 40]
})

# Check independence of residuals
result = check_independence(df, target="y")

# View results
print(result['dw_statistic'])    
print(result['is_independent'])  
print(result['message'])         
```

**Interpreting the Durbin-Watson statistic:**
- **1.5 to 2.5**: No significant autocorrelation (residuals are independent) ✓
- **< 1.5**: Positive autocorrelation detected
- **> 2.5**: Negative autocorrelation detected

**Note:** The function automatically uses all numeric columns (except the target) as predictors and handles the intercept term internally.

### Check Linearity

Identify features with strong linear relationships to the target:

```python
import pandas as pd
from lrassume import check_linearity

df = pd.DataFrame({
    "sqft": [500, 700, 900, 1100],
    "num_rooms": [1, 2, 1, 3],
    "age": [40, 25, 20, 5],
    "price": [150, 210, 260, 320]
})

linear_features = check_linearity(df, target="price", threshold=0.7)
print(linear_features)
#  feature  correlation
# 0    sqft        0.999
# 1     age       -0.990
```

### Check Multicollinearity

Compute Variance Inflation Factors to detect multicollinearity:

```python
import pandas as pd
from lrassume import check_multicollinearity_vif

X = pd.DataFrame({"sqft": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
     "bedrooms": [1, 2, 1, 3, 2, 4, 3, 5],
     "age": [30, 5, 40, 10, 25, 15, 35, 20]
})

vif_table, summary = check_multicollinearity_vif(X, warn_threshold=5.0)
print(summary['overall_status'])  # 'ok', 'warn', or 'severe'
# severe
print(vif_table)
#    feature        vif   level
# 0  bedrooms  11.100000  severe
# 1     sqft   9.402273    warn
# 2      age   3.102273      ok
```

### Check Homoscedasticity

Test for constant variance in residuals:

```python
import pandas as pd
import numpy as np
from lrassume import check_homoscedasticity
np.random.seed(123)
X = pd.DataFrame({
    'x1': np.linspace(1, 100, 100),
    'x2': np.random.randn(100)
})
y = pd.Series(2 * X['x1'] + 3 * X['x2'] + np.random.randn(100))

test_results, summary = check_homoscedasticity(X, y, method="breusch_pagan")
print(summary['overall_conclusion'])  # 'homoscedastic' 
print(test_results)
#            test  statistic  p_value     conclusion  significant
# 0  breusch_pagan      1.111   0.5737  homoscedastic        False
```

## Core Assumptions Tested

### 1. Independence
Residuals should be independent of each other (no autocorrelation). Violations occur in time-series or spatially correlated data.

### 2. Linearity
The relationship between predictors and the target should be approximately linear. Non-linear relationships may require transformations or non-linear models.

### 3. Multicollinearity
Predictors should not be highly correlated with each other. High multicollinearity inflates standard errors and makes coefficient estimates unstable.

### 4. Homoscedasticity
Residuals should have constant variance across all levels of predictors. Heteroscedasticity leads to inefficient estimates and incorrect standard errors.

## Advanced Usage

### Working with Pre-fitted Models

```python
from sklearn.linear_model import LinearRegression
from lrassume import check_homoscedasticity

model = LinearRegression().fit(X, y)
test_results, summary = check_homoscedasticity(
    X, y, 
    fitted_model=model,
    method="all"  # Run all tests
)
```

### Handling Categorical Variables

```python
from lrassume import check_multicollinearity_vif

# Automatically drop non-numeric columns
vif_table, summary = check_multicollinearity_vif(
    df, 
    target_column='price',
    categorical='drop'
)
print(summary['dropped_non_numeric'])  # Lists dropped columns
```

### Custom Thresholds

```python
# Stricter multicollinearity detection
vif_table, summary = check_multicollinearity_vif(
    X, 
    warn_threshold=3.0,
    severe_threshold=5.0
)

# More conservative homoscedasticity testing
test_results, summary = check_homoscedasticity(
    X, y, 
    alpha=0.01  # 99% confidence level
)
```
## Function Reference

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `check_independence()` | Durbin-Watson test for autocorrelation | `df`, `target` |
| `check_linearity()` | Pearson correlation analysis | `df`, `target`, `threshold` |
| `check_multicollinearity_vif()` | VIF calculation | `X`, `warn_threshold`, `severe_threshold` |
| `check_homoscedasticity()` | Heteroscedasticity testing | `X`, `y`, `method`, `alpha` |


## Interpretation Guidelines

### VIF Thresholds
- **VIF < 5**: No concerning multicollinearity
- **5 ≤ VIF < 10**: Moderate multicollinearity (warning)
- **VIF ≥ 10**: Severe multicollinearity (action recommended)

### Durbin-Watson Statistic
- **DW ≈ 2**: No autocorrelation (independence satisfied)
- **DW < 1.5**: Positive autocorrelation
- **DW > 2.5**: Negative autocorrelation

### Homoscedasticity Tests
- **p-value > α**: Fail to reject null hypothesis (homoscedastic)
- **p-value ≤ α**: Reject null hypothesis (heteroscedastic)

## Contributing

Contributions are welcome! Please see our [Code of Conduct](CODE_OF_CONDUCT.md) for community guidelines.

## License

Copyright © 2026 CHOT.

Free software distributed under the [MIT License](./LICENSE).

## Support

For bug reports and feature requests, please open an issue on [GitHub](https://github.com/UBC-MDS/lrassume/issues).
