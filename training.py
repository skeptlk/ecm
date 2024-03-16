from sklearn.metrics import mean_squared_error as mse, \
  mean_absolute_error as mae, \
  mean_absolute_percentage_error as mape, \
  r2_score as r2

def get_metrics(y, y_pred, round_n=5):
  return (
    {
      'rmse': round(mse(y, y_pred, squared=False), round_n),
      'mae': round(mae(y, y_pred), round_n),
      'r2': round(r2(y, y_pred), round_n),
      'mape': round(mape(y, y_pred), round_n),
    }
  )

def correct(data):
  datak = data.copy()
  alpha = 0.5
  theta = (data['oat'] + 273.15) / 288.15
  delta = (data['p2e'] * 68.948) / 1013.25

  datak['nfk'] = data['nf'] / (theta ** alpha)
  datak['n1ak'] = data['n1a'] / (theta ** alpha)
  datak['n1k'] = data['n1'] / (theta ** alpha)
  datak['n2ak'] = data['n2a'] / (theta ** alpha)
  datak['egtk'] = (data['egt'] + 273.15) / (theta ** 0.84)
  datak['ffk'] = (data['ff']) / (delta * (theta ** alpha))

  return datak