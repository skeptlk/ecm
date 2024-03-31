import pandas as pd 
from sklearn.linear_model import LinearRegression
from utils import * 
from training import * 

def train_engine_baseline(points: pd.DataFrame, x_param='n1ak', y_param='egtk'):
  model = LinearRegression()
  model.fit(points[[x_param]], points[y_param])
  return model

def compute_egtm(points: pd.DataFrame, model: LinearRegression, x_param='n1ak'):
  offset = 30
  delta = model.predict(points[[x_param]]) - points['egtk']
  return delta + offset


def add_egt_delta_to_dataset(
    dataset: pd.DataFrame, 
    bleed_param = 'prv', 
    fleet = [], 
    early = False, 
    interval = pd.to_timedelta('30D'),
    acms_data: pd.DataFrame = None
):
  for acnum in fleet: 
    for pos in [1, 2]:
      for bleed in [0, 1]:
        subset_index = (dataset['pos'] == pos) & (dataset['acnum'] == acnum) & (dataset[bleed_param] == bleed)
        subset = dataset[subset_index]
        
        if subset.shape[0] == 0:
          continue

        if acms_data is None: 
          early_end = subset.iloc[0]['reportts'] + interval
          early_filter = subset['reportts'] <= early_end
          baseline = train_engine_baseline(subset[early_filter] if early else subset)
        else:
          acms_index = (acms_data['pos'] == pos) & (acms_data['acnum'] == acnum) & (acms_data[bleed_param] == bleed)
          acms_subset = acms_data[acms_index]
          early_end = acms_subset.iloc[0]['reportts'] + interval
          acms_early = acms_subset[acms_subset['reportts'] <= early_end]
          print(acms_subset.iloc[0]['reportts'], acms_early.shape)
          baseline = train_engine_baseline(correct(acms_early))

        egt_delta = compute_egtm(subset, baseline)
        dataset.loc[subset_index, 'egt_delta'] = egt_delta

      subset_index = (dataset['pos'] == pos) & (dataset['acnum'] == acnum) 
      
      if dataset[subset_index].shape[0] == 0:
        continue
      
      egt_delta = dataset.loc[subset_index, 'egt_delta']
      egt_delta_smooth = smooth(egt_delta, 0.05)
      dataset.loc[subset_index, 'egt_delta_smooth'] = egt_delta_smooth

  return dataset
