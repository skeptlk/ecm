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
  alpha_2 = 0.84

  theta = (data['t2'] + 273.16) / (288.16)
  delta = data['p2e'] / 29.92

  datak['nfk'] = data['nf'] / (theta ** alpha)
  datak['n1ak'] = data['n1a'] / (theta ** alpha)
  datak['n1k'] = data['n1'] / (theta ** alpha)
  datak['n2ak'] = data['n2a'] / (theta ** alpha)
  datak['egtk'] = (data['egt'] + 273.16) / (theta ** alpha_2)
  datak['egtk_2'] = (data['egt'] + 273.16) / theta
  datak['ffk'] = (data['ff']) / (delta * (theta ** 0.59))

  datak['egt_peak_k'] = (data['egt_peak'] + 273.16) / (theta ** alpha_2)
  datak['n1a_peak_k'] = data['n1a_peak'] / (theta ** alpha)

  return datak