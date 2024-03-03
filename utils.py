import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_model(X, y, model = 'linreg'):
  assert len(X) == len(y)
  train_i = int(len(X) * 75 / 100)
  X_train, y_train = X[0:train_i], y[0:train_i]
  X_test, y_test = X[train_i:], y[train_i:]

  model = LinearRegression()

  model.fit(X_train, y_train)

  predicted_train = model.predict(X_train)

  predicted_test = model.predict(X_test)
  mse = mean_squared_error(y_test, predicted_test, squared=False)
  mae = mean_absolute_error(y_test, predicted_test)
  r2 = r2_score(y_test, predicted_test)

  return mse, mae, r2, model, predicted_train, predicted_test, train_i, y_test


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

def plot_predictions_for_print(
    data, acnum, pos, train_i, predicted_test, predicted_train, 
    figsize=(14, 7), split_date=pd.to_datetime('2019-08-06'), title=None
):
  data.loc[:train_i-1, 'pred_train'] = predicted_train
  data.loc[train_i:, 'pred_test'] = predicted_test

  sub = data[(data['acnum'] == acnum) & (data['pos'] == pos)]
  train_i2 = sub['pred_train'].count()

  plt.figure(figsize=figsize)

  plt.plot(sub['reportts'], sub['egtm'], '-', color='blue')

  smooth_train = smooth(sub['pred_train'][:train_i2], alpha=1/10)
  smooth_test = smooth(sub['pred_test'], alpha=1/10)

  plt.plot(sub['reportts'][:train_i2], smooth_train, '--', linewidth=2, color='red')
  plt.plot(sub['reportts'], smooth_test, '--', linewidth=2, color='red')
  
  plt.plot((split_date, split_date), (16, 40), '-', color='black')

  plt.text(split_date + pd.tseries.offsets.Day(2),  40 - 1, 'Test')
  plt.text(split_date - pd.tseries.offsets.Day(15),  40 - 1, 'Train')

  if title:
    plt.title(title)
  plt.legend(['Факт', 'Прогноз'])
  plt.show()
