import streamlit as st 
import pandas as pd

st.set_page_config(layout="wide")
st.write('## ECM P&W1000g models')

fleet = [
  # "RA73439",
  # "RA73440",
  # "RA73441",
  # "RA73442",
  # "RA73443",
  # "RA73444",
  # "RA73445",
  # "RA73446",
  "VQ-BDU",
  "VQ-BGU",
  "VQ-BDI",
  "VQ-BDV",
  "VQ-BDW",
  "VQ-BGR",
  "VQ-BYI",
  "VQ-BYJ"
]

col1, col2 = st.columns([1, 10])

acnum = col1.selectbox("Select aircraft", fleet)
pos = col1.radio("Select engine position", [1, 2])

is_clicked = col1.button("Predict")

data = pd.read_csv("VQ-BDU_VQ-BGU_with_delta.csv", parse_dates=['reportts'])
df = data.query(f'acnum=="{acnum}" and pos=={pos}')

col2.line_chart(df, x='reportts', y='egtm', height=540)
