import pandas as pd
import matplotlib.pyplot as plt
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

# Get exponential rolling average with smothing factor alpha
def smooth(x: pd.Series, alpha=0.5):
  return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_list()

def plot_predictions(data, acnum, pos, train_i, predicted_test, predicted_train, is_smooth=True, figsize=(14, 7), title=None):
  data.loc[:train_i-1, 'pred_train'] = predicted_train
  data.loc[train_i:, 'pred_test'] = predicted_test

  sub = data[(data['acnum'] == acnum) & (data['pos'] == pos)]
  train_i2 = sub['pred_train'].count()

  plt.figure(figsize=figsize)

  if is_smooth:
    plt.plot(sub['reportts'][:train_i2], smooth(sub['pred_train'][:train_i2], alpha=1/10), '-')
    plt.plot(sub['reportts'], smooth(sub['pred_test'], alpha=1/10), '-')
  else:
    plt.scatter(sub['reportts'][:train_i2], sub['pred_train'][:train_i2], s=2)
    plt.scatter(sub['reportts'], sub['pred_test'], s=2)

  plt.plot(sub['reportts'], sub['egtm'], '-', color='#2ca02c')

  plt.title(f'Linear model of EGTM on {acnum} engine {pos}, Gas path params' if title is None else title)
  plt.legend(['train_pred', 'test_pred', 'true'])
  plt.show()