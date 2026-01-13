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
    """Basic threshold 0.7 with alphabetical tie-breaks"""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms'],
        'correlation': [-1.000, 1.000, 0.945]
    })
    result = check_linearity(df_example, target="price", threshold=0.7)
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

def test_check_linearity_no_features():
    """Test when no features exceed threshold → empty DataFrame"""
    df_example_2 = df_example[['school_score', 'random_noise', 'price']]
    expected = pd.DataFrame({
        'feature': pd.Series(dtype='object'),
        'correlation': pd.Series(dtype='float')
    })
    result = check_linearity(df_example_2, target="price", threshold=0.8)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_tie_break():
    """Test alphabetical tie-break when multiple features have the same absolute correlation."""
    df_tie = df_example.copy()
    # To match correlation, create copy of a feature with different name
    df_tie['num_rooms_clone'] = df_tie['num_rooms']
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms', 'num_rooms_clone'],
        'correlation': [-1.000, 1.000, 0.945, 0.945]
    })
    result = check_linearity(df_tie, target="price", threshold=0.7)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_rounding():
    """Test that correlations returned are rounded to three decimal places"""
    result = check_linearity(df_example, target='price', threshold=0.0)
    assert all(result['correlation'].apply(lambda x: round(x, 3) == x))
#------------------------------------- 
# Test Error Cases for Invalid Inputs
# ------------------------------------
def test_check_linearity_invalid_target():
    """Test invalid target column raises ValueError"""
    with pytest.raises(ValueError):
        check_linearity(df_example, target="nonexistent", threshold=0.7)

def test_check_linearity_non_numeric_target():
    """Test non-numeric target column raises TypeError"""
    df_copy = df_example.copy()
    df_copy["price"] = df_copy["price"].astype(str)
    with pytest.raises(TypeError):
        check_linearity(df_copy, target="price")

def test_check_linearity_invalid_threshold():
    """Test invalid threshold raises ValueError"""
    with pytest.raises(ValueError):
        check_linearity(df_example, target="price", threshold=1.5)
    with pytest.raises(ValueError):
        check_linearity(df_example, target="price", threshold=-0.1) 

def test_check_linearity_df_type():
    """Test that passing a non-DataFrame raises TypeError"""
    not_a_df = [1, 2, 3, 4]  # List instead of DataFrame
    with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
        check_linearity(not_a_df, target="price", threshold=0.7)

    with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
        check_linearity("not a dataframe", target="price", threshold=0.7)

def test_check_linearity_target_type():
    """Test that passing a non-string target raises TypeError"""
    with pytest.raises(TypeError, match="Input 'target' must be a string."):
        check_linearity(df_example, target=123, threshold=0.7)

    with pytest.raises(TypeError, match="Input 'target' must be a string."):
        check_linearity(df_example, target=None, threshold=0.7)

def test_check_linearity_threshold_type():
    """Test that passing a non-numeric threshold raises TypeError"""
    with pytest.raises(TypeError, match="Input 'threshold' must be a numeric value."):
        check_linearity(df_example, target="price", threshold="high")

    with pytest.raises(TypeError, match="Input 'threshold' must be a numeric value."):
        check_linearity(df_example, target="price", threshold=None)

