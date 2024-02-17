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
