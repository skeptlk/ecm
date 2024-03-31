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
      df = acdata[acdata['pos'] == pos].copy().reset_index(drop=True)
      if df.shape[0] == 0:
        continue
      X = df[features]
      X_aug = X.copy()

      for offset in range(1, n_back + 1):
        features_back = [f"{i}_{offset}" for i in features]
        X_aug.loc[0:offset, features_back] =  X.iloc[0].to_numpy()
        X_aug.loc[offset:, features_back] = X.iloc[:-offset,:].to_numpy()
      
      X_aug = pd.concat([df[rest_features], X_aug], axis=1)
      result.append(X_aug)

  result = pd.concat(result)

  if ('reportts' in rest_features):
    result = result.sort_values('reportts') \
              .reset_index(drop=True)
  
  return result


def build_dataset(fleet: List[pd.DataFrame], y_cols, meta_cols, features, n_back=1, data: pd.DataFrame = None):
  if data is None:
    return get_recursive_features(
      [df[y_cols + meta_cols + features] for df in fleet],
      features, 
      n_back
    )
  else:
    return get_recursive_features(
      [data[data['acnum'] == acnum][y_cols + meta_cols + features] for acnum in fleet],
      features, 
      n_back
    )


# Get exponential rolling average with smothing factor alpha
def smooth(x: pd.Series, alpha=0.5):
  return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_list()


def plot_predictions(data, acnum, pos, train_i, predicted_test, predicted_train, is_smooth=True, figsize=(14, 7), title=None, alpha=1/10):
  data = pd.concat([
    data.drop(columns=['pred_test', 'pred_train'], errors='ignore'), 
    predicted_test.rename(columns={'pred': 'pred_test'}), 
    predicted_train.rename(columns={'pred': 'pred_train'})
  ], axis=1)

  sub = data.query(f'acnum=="{acnum}" and pos=={pos}')

  train_i2 = sub['pred_train'].count()

  plt.figure(figsize=figsize)

  if is_smooth:
    series = sub['pred_train'].dropna().to_list() + sub['pred_test'].dropna().to_list()
    smoothed = smooth(series, alpha=alpha)
    plt.plot(sub['reportts'][:train_i2], smoothed[:train_i2], '-')
    plt.plot(sub['reportts'][train_i2:], smoothed[train_i2:], '-')
  else:
    plt.scatter(sub['reportts'], sub['pred_train'], s=1)
    plt.scatter(sub['reportts'], sub['pred_test'], s=1)

  plt.plot(sub['reportts'], sub['egtm'], '-', color='#2ca02c')

  plt.title(f'Linear model of EGTM on {acnum} engine {pos}, Gas path params' if title is None else title)
  plt.legend(['train_pred', 'test_pred', 'true'])
  plt.show()


def plot_predictions_for_print(
    data, acnum, pos, train_i, predicted_test, predicted_train, 
    figsize=(14, 7), split_date=pd.to_datetime('2019-08-06'), title=None,
    baseline_test=None, baseline_train=None, 
):
  data.loc[:train_i-1, 'pred_train'] = predicted_train
  data.loc[train_i:, 'pred_test'] = predicted_test

  if baseline_train is not None:  
    data.loc[:train_i-1, 'baseline_train'] = baseline_train
    data.loc[train_i:, 'baseline_test'] = baseline_test

  sub = data[(data['acnum'] == acnum) & (data['pos'] == pos)]
  train_i2 = sub['pred_train'].count()
  
  plt.figure(figsize=figsize)
  plt.plot(sub['reportts'], sub['egtm'], '-', color='blue')

  if baseline_train is not None:
    smooth_baseline = smooth(list(sub['baseline_train'][:train_i2]) + list(sub['baseline_test'][train_i2:]), alpha=1/10)
    plt.plot(sub['reportts'], smooth_baseline, '-.', color='green')

  smooth_pred = smooth(list(sub['pred_train'][:train_i2]) + list(sub['pred_test'][train_i2:]), alpha=1/10)
  plt.plot(sub['reportts'], smooth_pred, '--', linewidth=1.5, color='red')

  plt.plot((split_date, split_date), (16, 40), '-', color='black')
  plt.text(split_date + pd.tseries.offsets.Day(2),  40 - 1, 'Test')
  plt.text(split_date - pd.tseries.offsets.Day(15),  40 - 1, 'Train')

  if title:
    plt.title(title)
  plt.legend(['Факт', 'Прогноз исходной модели', 'Прогноз итоговой модели'])
  plt.show()
