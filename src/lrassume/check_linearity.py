def check_linearity(df: pandas.DataFrame, target: str, threshold: float = 0.7) -> pandas.DataFrame: 
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
