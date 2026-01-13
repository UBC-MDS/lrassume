import pandas as pd
from pandas.api.types import is_numeric_dtype

def check_linearity(df: pd.DataFrame, target: str, threshold: float = 0.7) -> pd.DataFrame: 
  """Identify features with a specified strength of linear relationship to the target.
  
  This function identifies all of the numeric features in a DataFrame and computes the Pearson correlation coefficient between each numeric feature in the DataFrame and the specified numeric target column.
  It returns a DataFrame containing the features whose absolute correlation with the target is greater than or equal to the given threshold along with their correlation values.
  
  Parameters
  ----------
  df : pandas.DataFrame
    Input DataFrame with feature columns and the target column. Only numeric features will be considered. 
  target : str
    Name of the target column. The column must be numeric and the name must match the column name in data. 
  threshold : float, optional
    Minimum absolute Pearson correlation required for a feature to be considered strongly correlated with the target. Must be between 0 and 1. Default is 0.7.
  
  Returns
  -------
  pandas.DataFrame
    A Dataframe with the following columns:
      - feature : str
        Name of the feature column
      - correlation : float
        Pearson correlation coefficient between the feature and the target.
    The DataFrame is sorted by absolute correlation in descending order. 
  
  Examples
  --------
  >>> df_example = pd.DataFrame({
    "sqft": [500, 700, 900, 1100],
    "num_rooms": [1, 2, 1, 3],
    "age": [40, 25, 20, 5], 
    "distance_to_city": [10, 12, 11, 13],
    "price": [150, 210, 260, 320]
  })
  >>> check_linearity(df=df_example, target="price", threshold=0.7)
          feature  correlation
  0          sqft     0.994
  1            age    -0.952
  2      num_rooms     0.703

  """
  if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
  if not isinstance(target, str):
        raise TypeError("Input 'target' must be a string.") 
  if not isinstance(threshold, (int, float)):
        raise TypeError("Input 'threshold' must be a numeric value.")
  if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
  if not is_numeric_dtype(df[target]):
        raise TypeError("Target column must be numeric.")
  if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    # Compute correlations for all numeric columns at once
  numeric_cols = df.select_dtypes(include='number').columns.drop(target, errors='ignore')
  corrs = df[numeric_cols].corrwith(df[target]).round(3)

    # Filter correlations by threshold
  corrs = corrs[abs(corrs) >= threshold]

    # If no features meet the threshold, return empty DataFrame
  if corrs.empty:
      return pd.DataFrame({
          "feature": pd.Series(dtype="object"),
          "correlation": pd.Series(dtype="float")
        })

    # Convert Series to DataFrame
  result = corrs.reset_index()
  result.columns = ['feature', 'correlation']

    # Sort by absolute correlation descending, then alphabetically by feature for tie-breaks
  result['abs_corr'] = result['correlation'].abs()
  result = result.sort_values(by=['abs_corr', 'feature'], ascending=[False, True]).drop(columns='abs_corr').reset_index(drop=True)

  return result
