import pytest
import pandas as pd
from lrassume.check_linearity import check_linearity

df_eg = pd.DataFrame({
    "sqft": [500, 700, 900, 1100],
    "num_rooms": [1, 2, 1, 3],
    "age": [40, 25, 20, 5], 
    "distance_to_city": [10, 12, 11, 13],
    "price": [150, 210, 260, 320]
})

def test_check_linearity():
    expected_result = pd.DataFrame({
        'feature': ['sqft', 'age', 'distance_to_city'], 
        'correlation': [0.999, -0.990, 0.821]  # rounded to 3 decimals
    })
    
    actual = check_linearity(df=df_eg, target="price", threshold=0.7)
    actual['correlation'] = actual['correlation'].round(3)  # round actual values
    
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected_result)


#def test_check_linearity_no_strong_correlations():
   
#def test_check_linearity_errors():

