import pandas as pd
from typing import List

def get_recursive_features(data: List[pd.DataFrame], features = [], n_back = 1):
  result = []
  rest_features = list(set(data[0].columns) - set(features))
  assert len(rest_features + features) == len(data[0].columns)
  
  for acdata in data:
    for pos in [1, 2]:
      df = acdata[acdata['pos'] == pos].copy().reset_index()
      if df.shape[0] == 0:
        continue
      X = df[features]
      X_aug = X.copy()
      for offset in range(1, n_back + 1):
        features_back = [f"{i}_{offset}" for i in features]
        X_aug.loc[0:offset, features_back] =  X.iloc[0,:].to_numpy()
        X_aug.loc[offset:, features_back] = X.iloc[:-offset,:].to_numpy()
      
      X_aug.loc[:, rest_features] = df[rest_features]
      result.append(X_aug)
  
  result = pd.concat(result) \
              .sort_values('reportts' if 'reportts' in rest_features else 'pos') \
              .reset_index() \
              .drop(columns=['index'])
  return result

def build_dataset(fleet: List[pd.DataFrame], y_cols, meta_cols, features, n_back=1):
  return get_recursive_features(
    [df[y_cols + meta_cols + features] for df in fleet],
    features, 
    n_back
  )

