import pytest
import pandas as pd
from lrassume.check_linearity import check_linearity

df_example = pd.DataFrame({
    "sqft": [500, 600, 700, 800, 900],
    "num_rooms": [1, 1, 2, 2, 3],
    "age": [50, 40, 30, 20, 10],
    "school_score": [60, 65, 55, 75, 70],
    "random_noise": [42, 17, 88, 55, 63],
    "neighbourhood": ["A", "B", "A", "B", "A"],
    "price": [150, 180, 210, 240, 270]
})
# -------------------------------------
# Test Valid Cases
# ------------------------------------
def test_check_linearity_basic():
    """Basic threshold 0.7 test case."""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms'],
        'correlation': [-1.000, 1.000, 0.945]
    })
    result = check_linearity(df_example, target="price", threshold=0.7)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_threshold_default():
    """Test default threshold 0.7 behavior."""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms'],
        'correlation': [-1.000, 1.000, 0.945]
    })
    result = check_linearity(df_example, target="price")
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_threshold_one():
    """Test threshold edge case 1.0 (features equal to threshold are included)."""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft'],
        'correlation': [-1.000, 1.000]
    })
    result = check_linearity(df_example, target="price", threshold=1.0)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_threshold_zero():
    """Test threshold edge case 0.0 returns all numeric features"""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms', 'school_score', 'random_noise'],
        'correlation': [-1.000, 1.000, 0.945, 0.600, 0.483]
    })
    result = check_linearity(df_example, target="price", threshold=0.0)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_no_numeric_features():
    """Test when no numeric features exist → empty DataFrame"""
    df_no_numeric = df_example[['neighbourhood', 'price']]
    expected = pd.DataFrame({
        'feature': pd.Series(dtype='object'),
        'correlation': pd.Series(dtype='float')
    })
    result = check_linearity(df_no_numeric, target="price", threshold=0.5)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_no_feature_target_correlation():
    """Test when no features exceed threshold → empty DataFrame"""
    df_example_2 = df_example[['school_score', 'random_noise', 'price']]
    expected = pd.DataFrame({
        'feature': pd.Series(dtype='object'),
        'correlation': pd.Series(dtype='float')
    })
    result = check_linearity(df_example_2, target="price", threshold=0.8)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_tie_break():
    """Test alphabetical tie-break when multiple features have the same absolute correlation.

    Creates a duplicate feature with the same correlation as an existing feature. 
    Verifies that the output is sorted first by absolute correlation descending, 
    then alphabetically by feature name.
    """
    df_tie = df_example.copy()
    # Duplicate 'num_rooms' to create tie in absolute correlation
    df_tie['num_rooms_clone'] = df_tie['num_rooms']
    
    result = check_linearity(df_tie, target="price", threshold=0.7)
    
    # Check correlations
    expected_corrs = [-1.000, 1.000, 0.945, 0.945]
    assert result['correlation'].tolist() == expected_corrs, "Correlation values are incorrect."
    
    # Check alphabetical ordering for tie-break
    expected_order = ['age', 'sqft', 'num_rooms', 'num_rooms_clone']
    assert result['feature'].tolist() == expected_order, "Tie-break alphabetical ordering failed."

#------------------------------------- 
# Test Error Cases
# ------------------------------------
import pytest
import pandas as pd
from lrassume.check_linearity import check_linearity

df_example = pd.DataFrame({
    "sqft": [500, 600, 700, 800, 900],
    "num_rooms": [1, 1, 2, 2, 3],
    "age": [50, 40, 30, 20, 10],
    "school_score": [60, 65, 55, 75, 70],
    "random_noise": [42, 17, 88, 55, 63],
    "neighbourhood": ["A", "B", "A", "B", "A"],
    "price": [150, 180, 210, 240, 270]
})
# -------------------------------------
# Test Error Cases
# -------------------------------------
def test_check_linearity_invalid_target():
    """Raise ValueError if the target column does not exist in the DataFrame."""
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found in DataFrame."):
        check_linearity(df_example, target="nonexistent", threshold=0.7)


def test_check_linearity_non_numeric_target():
    """Raise TypeError if the target column exists but is not numeric."""
    df_copy = df_example.copy()
    df_copy["price"] = df_copy["price"].astype(str)
    with pytest.raises(TypeError, match="Target column must be numeric."):
        check_linearity(df_copy, target="price")


def test_check_linearity_invalid_threshold():
    """Raise ValueError if the threshold is not between 0 and 1."""
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        check_linearity(df_example, target="price", threshold=1.5)
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        check_linearity(df_example, target="price", threshold=-0.1)


def test_check_linearity_df_type():
    """Raise TypeError if the input df is not a pandas DataFrame."""
    not_a_df = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
        check_linearity(not_a_df, target="price", threshold=0.7)

    with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
        check_linearity("not a dataframe", target="price", threshold=0.7)


def test_check_linearity_target_type():
    """Raise TypeError if the target argument is not a string."""
    with pytest.raises(TypeError, match="Input 'target' must be a string."):
        check_linearity(df_example, target=123, threshold=0.7)

    with pytest.raises(TypeError, match="Input 'target' must be a string."):
        check_linearity(df_example, target=None, threshold=0.7)


def test_check_linearity_threshold_type():
    """Raise TypeError if the threshold argument is not numeric."""
    with pytest.raises(TypeError, match="Input 'threshold' must be a numeric value."):
        check_linearity(df_example, target="price", threshold="high")

    with pytest.raises(TypeError, match="Input 'threshold' must be a numeric value."):
        check_linearity(df_example, target="price", threshold=None)


def test_check_linearity_missing_arguments():
    """Raise TypeError when df or target is None; threshold is optional."""
    # df is None
    with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
        check_linearity(df=None, target="price")

    # target is None
    with pytest.raises(TypeError, match="Input 'target' must be a string."):
        check_linearity(df=df_example, target=None)

    # Threshold omitted is fine
    result = check_linearity(df=df_example, target="price")
    assert not result.empty, "Function failed when threshold argument was omitted."
