import pandas as pd 
from sklearn.linear_model import LinearRegression
from utils import * 

def train_engine_baseline(points: pd.DataFrame, x_param='n1ak', y_param='egtk'):
  model = LinearRegression()
  model.fit(points[[x_param]], points[y_param])
  return model

def compute_egtm(points: pd.DataFrame, model: LinearRegression, x_param='n1ak'):
  offset = 30
  delta = model.predict(points[[x_param]]) - points['egtk']
  return delta + offset


def add_egt_delta_to_dataset(dataset: pd.DataFrame, bleed_param='prv', fleet=[]):
  for acnum in fleet: 
    for pos in [1, 2]:
      for bleed in [0, 1]:
        subset_index = (dataset['pos'] == pos) & (dataset['acnum'] == acnum) & (dataset[bleed_param] == bleed)
        if dataset[subset_index].shape[0] == 0:
          continue
        baseline = train_engine_baseline(dataset[subset_index])
        egt_delta = compute_egtm(dataset[subset_index], baseline)
        dataset.loc[subset_index, 'egt_delta'] = egt_delta

      subset_index = (dataset['pos'] == pos) & (dataset['acnum'] == acnum) 
      
      if dataset[subset_index].shape[0] == 0:
        continue
      
      egt_delta = dataset.loc[subset_index, 'egt_delta']
      egt_delta_smooth = smooth(egt_delta, 0.05)
      dataset.loc[subset_index, 'egt_delta_smooth'] = egt_delta_smooth

  return dataset
