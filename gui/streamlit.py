from sklearn.linear_model import Ridge
import streamlit as st 
import pandas as pd
import altair as alt
import urllib.request
import joblib
from utils import *

st.set_page_config(layout="wide")
st.write('## ECM P&W1000g models')

def predict_boolean_ensemble(models: List[Ridge], X: pd.DataFrame, field='nai'):
  pred = pd.DataFrame(index=X.index.copy(), columns=['pred'])
  for val in [True, False]:
    index = (X[field] == val)
    pred.loc[index, 'pred'] = models[val].predict(X[index])
  return pred

@st.cache_data
def get_data():
  return pd.read_csv(
    "https://drive.google.com/uc?export=view&id=1wCaZH0A-r6QRdMvAnwtKn8L1nUREWcn9", 
    parse_dates=['reportts']
  )


@st.cache_data
def evaluate_model(model, acnum, acms_data):
  acms_corrrected = correct(acms_data)
  X = add_egt_delta_to_dataset(
    acms_corrrected, 
    x_param='n1a_peak_k', 
    y_param='egt_peak_k', 
    fleet=[acnum], 
    early=True
  )
  return predict_boolean_ensemble(model, X, field='prv')

fleet = [
  "VQ-BDU",
  "VQ-BGU",
  # "RA73439",
  # "RA73440",
  # "RA73441",
  # "RA73442",
  # "RA73443",
  # "RA73444",
  # "RA73445",
  # "RA73446",
  # "VQ-BDI",
  # "VQ-BDV",
  # "VQ-BDW",
  # "VQ-BGR",
  # "VQ-BYI",
  # "VQ-BYJ"
]

col1, col2 = st.columns([2, 10])

acnum = col1.selectbox("Aircraft", fleet)
pos = col1.radio("Engine position", [1, 2])

is_clicked = col1.button("Predict")

df = get_data().query(f'acnum=="{acnum}" and pos=={pos}')

leg = alt.Legend(
  title="",
  orient='none',
  legendX=930,
  legendY=30,
)

base = alt.Chart(df, height=700)

egtm = base.mark_line(interpolate="basis") \
  .transform_calculate(line="'EGTM'") \
  .encode(
    x=alt.X('reportts:T'),
    y=alt.Y('egtm', title="EGT Margin"),
    color=alt.Color('line:N', legend=leg)
  )

delta = base.mark_point(filled=True) \
  .transform_calculate(line="'EGT Delta'") \
  .encode(
    x=alt.X('reportts:T'),
    y=alt.Y('egt_delta', title=""),
    opacity=alt.value(0.3),
    color=alt.Color('line:N', legend=leg)
  )

delta_smooth = base.mark_line(interpolate="basis") \
  .transform_calculate(line="'EGT Delta Smooth'") \
  .encode(
    x=alt.X('reportts:T', title="Reported time"),
    y=alt.Y('egt_delta_smooth', title=""),
    color=alt.Color('line:N', legend=leg)
  )

alt.themes.enable("googlecharts")

chart = alt.layer(egtm + delta_smooth + delta)

col2.altair_chart(chart, use_container_width=True, theme=None)
